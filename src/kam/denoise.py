"""Reusable CWF denoising of RELION particle stacks with FastPCA.

Includes the CUDA FLE transforms behind a lazy factory, so importing this module never requires fle_2d or CuPy.

The output is a CTF-corrected stack in unwhitened, amplitude-preserving units.
The radial FastPCA CTF uses mean defocus, optionally quantized into bins.  The
manifest retains the original per-particle CTF metadata and describes both the
astigmatism and defocus-quantization approximations.

No RELION pose, origin, reference map, or clean projection is used, and the images are not shifted.
Downstream moment estimation must treat the saved images as clean latent images:
do not apply CTF correction or noise debiasing a second time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from .utils import ROOT, git_commit

# git submodules checked out next to src/ (git clone --recursive)
FAST_PCA_ROOT = ROOT / "fast-cryoEM-PCA"
FLE_ROOT = ROOT / "fle_2d"
DEFAULT_DENOISE_MAX_ORDER = 40
CTF_COLUMNS = (
    "_rlnVoltage",
    "_rlnDefocusU",
    "_rlnDefocusV",
    "_rlnDefocusAngle",
    "_rlnSphericalAberration",
    "_rlnAmplitudeContrast",
)
RADIAL_CTF_COLUMNS = (
    "_rlnVoltage",
    "_rlnSphericalAberration",
    "_rlnAmplitudeContrast",
    "effective_mean_defocus",
)


@dataclass(frozen=True)
class FastCWFDenoiseConfig:
    """Inputs and numerical settings for :func:`denoise_relion_particles`."""

    dataset_label: str
    star_path: Path
    output_dir: Path
    data_folder: Path | None = None
    stack_prefix: str | None = None
    native_pixel_size: float | None = None
    n_images: int = 0
    seed: int = 0
    denoise_res: int = 64
    batch_size: int = 256
    workers: int = 1
    device: str = "gpu"
    background_radius: float = 0.9
    eps: float = 1e-3
    denoise_max_order: int = DEFAULT_DENOISE_MAX_ORDER
    ctf_mean_defocus_bin: float = 50.0
    flip_sign: bool = True


@dataclass(frozen=True)
class FastCWFDenoiseResult:
    """Paths and key dimensions produced by CWF denoising."""

    output_dir: Path
    stack_path: Path
    manifest_path: Path
    report_path: Path
    n_images: int
    n_total: int
    native_res: int
    denoise_res: int
    native_pixel_size: float
    denoise_pixel_size: float
    elapsed_seconds: float


def select_rows(n_total, n_images, seed):
    """Select rows without replacement and return canonical STAR order.

    ``n_images=0`` selects the complete STAR table without invoking the RNG.
    """
    if n_images == 0:
        return np.arange(n_total, dtype=np.int64)
    rows = np.random.default_rng(seed).choice(n_total, size=n_images, replace=False)
    return np.sort(rows).astype(np.int64)


def group_radial_ctfs(parameters, mean_defocus_bin):
    """Group radial CTFs and return an internal contiguous ordering.

    Groups retain exact voltage, spherical aberration, and amplitude contrast.
    They ignore astigmatism orientation and replace ``(U,V)`` by mean defocus.
    A positive bin width rounds mean defocus to its nearest bin center; zero
    retains the exact per-particle mean.
    """
    parameters = np.asarray(parameters, dtype=np.float64)

    exact_mean = 0.5 * (parameters[:, 1] + parameters[:, 2])
    if mean_defocus_bin == 0:
        effective_mean = exact_mean.copy()
    else:
        effective_mean = mean_defocus_bin * np.rint(exact_mean / mean_defocus_bin)
    keys = np.column_stack((parameters[:, 0], parameters[:, 4],
                            parameters[:, 5], effective_mean))
    unique, ids, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(ids, kind="stable")
    inverse = np.argsort(order)
    starts = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    return unique, ids.astype(np.int64), order, inverse, starts, counts, exact_mean, effective_mean


def ctf_phase_error_bounds(parameters, effective_mean, pixel_size):
    """Return Nyquist phase-error bounds for radialization and quantization."""
    parameters = np.asarray(parameters, dtype=np.float64)
    effective_mean = np.asarray(effective_mean, dtype=np.float64)
    voltage = parameters[:, 0]
    wavelength = 12.2643247 / np.sqrt(voltage * 1e3 + 0.978466 * voltage**2)
    exact_mean = 0.5 * (parameters[:, 1] + parameters[:, 2])
    nyquist_squared = (1.0 / (2.0 * pixel_size)) ** 2
    astigmatism = np.abs(parameters[:, 1] - parameters[:, 2])
    astigmatism_bound = np.pi * wavelength * (0.5 * astigmatism) * nyquist_squared
    quantization_bound = (
        np.pi * wavelength * np.abs(exact_mean - effective_mean) * nyquist_squared
    )
    return astigmatism, astigmatism_bound, quantization_bound


def subtract_background_mean(images, radius=0.9):
    """Subtract each image's background mean without normalizing its STD."""
    images = np.asarray(images)
    side = images.shape[-1]
    axis = (np.arange(side, dtype=np.float64) - side // 2) / (side / 2)
    mask = np.hypot(axis[:, None], axis[None, :]) > radius
    background = images[:, mask].astype(np.float64)
    means = np.mean(background, axis=1)
    stds = np.std(background, axis=1)
    return (images - means[:, None, None]).astype(np.float32), means, stds


def _with_max_order(basis_class, max_order):
    """FLE basis class that keeps only the eigenfunctions of angular order ``|nu| <= max_order``.

    Conjugate pairs +-nu stay together and FastPCA's mean, covariance and Wiener filter then run on the smaller basis."""

    class MaxOrderBasis(basis_class):
        def lap_eig_disk(self, *args, **kwargs):
            psi, ns, ks, lmds, cs, _ = super().lap_eig_disk(*args, **kwargs)
            keep = np.abs(ns) <= max_order
            return ([f for f, k in zip(psi, keep) if k], ns[keep], ks[keep], lmds[keep], cs[keep], int(keep.sum()))

    return MaxOrderBasis


def denoise_fastpca_batches(fast_pca, mean_est, covar_est, batch_size):
    """Apply FastPCA CWF in full batches, including mixed CTF groups.

    The upstream implementation restarts the FLE transform for every CTF
    group.  Real data can contain hundreds of groups, many smaller than one
    batch.  This equivalent formulation transforms each full image batch once
    and supplies local filter identifiers to the same Wiener filter.
    """
    from utils_cwf_fast_batch import wiener_filter

    basis = fast_pca.basis
    source = fast_pca.src

    count = int(source.n)
    side = int(source.L)
    filter_ids = np.asarray(source.filter_indices, dtype=np.int64)
    whiten_radial = 1.0 / np.sqrt(fast_pca.noise_psd)
    whiten_eigen = basis.expand_radial_vec(whiten_radial)
    whiten = basis.to_angular_order(whiten_eigen.T).T
    filters = source.unique_filters
    block_zero = slice(basis.blk_ind[0], basis.blk_ind[1])
    coefficient_one = basis.to_angular_order(
        basis.evaluate_t(np.ones((side, side))).reshape(-1))
    coefficient_one_zero = coefficient_one[block_zero]

    images = np.empty((count, side, side), dtype=np.float32)
    contrast = np.empty(count, dtype=np.float64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        batch = source.images[start:end].asnumpy()
        if fast_pca.flip_sign:
            batch = -batch
        transform_started = time.monotonic()
        coefficients_eigen = basis.evaluate_t(batch)
        fast_pca.t_basis_expansion += time.monotonic() - transform_started
        coefficients = basis.to_angular_order(coefficients_eigen.T).T

        batch_filter_ids = filter_ids[start:end]
        unique_ids, local_ids = np.unique(
            batch_filter_ids, return_inverse=True)
        voltage = np.asarray([filters[index].voltage for index in unique_ids])
        spherical_aberration = np.asarray(
            [filters[index].Cs for index in unique_ids])
        amplitude_contrast = np.asarray(
            [filters[index].alpha for index in unique_ids])
        defocus = np.asarray(
            [filters[index].defocus_mean for index in unique_ids])
        transfer = basis.expand_ctf(
            voltage, spherical_aberration, amplitude_contrast, defocus,
            fast_pca.pixel_size)
        transfer = basis.to_angular_order(transfer.T).T
        coefficients *= whiten
        transfer *= whiten

        wiener_started = time.monotonic()
        estimated = wiener_filter(
            coefficients, mean_est, covar_est, fast_pca.noise_var,
            transfer, local_ids, basis)
        fast_pca.t_denoise += time.monotonic() - wiener_started
        contrast[start:end] = estimated[:, block_zero] @ coefficient_one_zero
        estimated_eigen = basis.to_eigen_order(estimated.T).T
        transform_started = time.monotonic()
        images[start:end] = np.asarray(
            basis.evaluate(estimated_eigen), dtype=np.float32)
        fast_pca.t_basis_expansion += time.monotonic() - transform_started
        print(f"FastPCA denoised {end:,}/{count:,}", flush=True)

    return images, contrast


def denoise_relion_particles(config: FastCWFDenoiseConfig) -> FastCWFDenoiseResult:
    """Denoise a reproducibly selected RELION particle subset.

    The function writes ``denoised_images.mrcs``, ``manifest.npz``,
    ``report.json``, and ``denoising_preview.pdf`` to a new output directory. Returned
    images are unwhitened latent-image estimates with their amplitude and
    FastPCA contrast estimate retained.
    """
    args = config
    fast_pca_root = FAST_PCA_ROOT.resolve()
    fle_root = FLE_ROOT.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory; previous results are preserved")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path[:0] = [str(ROOT / "src"), str(fast_pca_root), str(fle_root / "src")]

    import aspire
    from aspire.image import Image
    from fast_cryo_pca import FastPCA
    from fle_2d import FLEBasis2D
    import mrcfile
    from .moments import centered_ifft2
    from .preprocessing import RelionParticles, downsample_fourier_images
    from utils_cwf_fast_batch import estimate_radial_psd

    started = time.monotonic()
    particles = RelionParticles(
        args.star_path,
        data_folder=args.data_folder,
        stack_prefix=args.stack_prefix or None,
        pixel_size=args.native_pixel_size,
    )
    source_rows = select_rows(particles.n, args.n_images, args.seed)
    n_images = len(source_rows)
    source_fingerprint = particles.fingerprint()
    image_names = particles.image_names[source_rows].copy()
    parameters = np.column_stack(
        [particles.column(key)[source_rows] for key in CTF_COLUMNS]
    )
    phase_shifts = particles.column("_rlnPhaseShift", default=0.0)[source_rows]
    if np.any(phase_shifts != 0):
        raise ValueError("FastPCA radial CTF does not support nonzero STAR phase shifts")
    (radial_parameters, ctf_ids, order, inverse, starts, counts, exact_mean,
     effective_mean) = group_radial_ctfs(parameters, args.ctf_mean_defocus_bin)
    voxel_size = particles.pixel_size * particles.res / args.denoise_res
    astigmatism, astig_bound, quantization_bound = ctf_phase_error_bounds(
        parameters, effective_mean, voxel_size
    )
    filters = [
        SimpleNamespace(
            voltage=row[0],
            Cs=row[1],
            alpha=row[2],
            defocus_mean=row[3],
            pixel_size=voxel_size,
        )
        for row in radial_parameters
    ]
    sampling = (
        "all_STAR_rows" if n_images == particles.n
        else "uniform_without_replacement_then_sort_STAR_rows"
    )
    print(
        f"Selected {n_images:,}/{particles.n:,} particles; sampling={sampling}; "
        f"seed={args.seed}; canonical ascending STAR rows",
        flush=True,
    )
    print(
        f"Native {particles.res}px at {particles.pixel_size:.12g} A/px "
        f"(STAR {particles.metadata_pixel_size:.12g}); denoise {args.denoise_res}px "
        f"at {voxel_size:.12g} A/px; images are not shifted; no pose or origin corrections",
        flush=True,
    )
    print(
        f"Radial CTF: nearest mean-defocus bin={args.ctf_mean_defocus_bin:g} A, "
        f"{len(radial_parameters):,} groups, group size min/median/max="
        f"{counts.min()}/{np.median(counts):g}/{counts.max()}",
        flush=True,
    )
    print(
        "CTF approximation at denoising Nyquist: "
        f"|U-V| median/max={np.median(astigmatism):.4g}/{np.max(astigmatism):.4g} A; "
        f"astigmatism phase bound median/max={np.median(astig_bound):.4g}/"
        f"{np.max(astig_bound):.4g} rad; quantization phase bound median/max="
        f"{np.median(quantization_bound):.4g}/{np.max(quantization_bound):.4g} rad",
        flush=True,
    )
    print(
        "Subtracting background means only; per-image STD and contrast are retained. "
        f"flip_sign={args.flip_sign}. Whitening is internal; output is unwhitened.",
        flush=True,
    )

    basis_started = time.monotonic()
    if args.device == "gpu":
        GPUFLEBasis2D, estimate_radial_psd_gpu = _gpu_fle_components()

        basis = _with_max_order(GPUFLEBasis2D, args.denoise_max_order)(
            args.denoise_res, args.denoise_res, eps=args.eps,
            precision="single")
        def psd_estimator(images):
            return estimate_radial_psd_gpu(
                images, basis, args.background_radius, np.float32)

        fastpca_backend = "cupy_cufinufft_fle"
    else:
        basis = _with_max_order(FLEBasis2D, args.denoise_max_order)(
            args.denoise_res, args.denoise_res, eps=args.eps,
            precision="single")
        def psd_estimator(images):
            return estimate_radial_psd(
                images, basis, args.background_radius, np.float32)

        fastpca_backend = "numpy_scipy_finufft_fle"
    basis_seconds = time.monotonic() - basis_started
    print(
        f"FastPCA backend={fastpca_backend}; FLE precision={basis.precision}; "
        f"basis setup={basis_seconds:.2f} s",
        flush=True,
    )

    observations = np.empty(
        (n_images, args.denoise_res, args.denoise_res), dtype=np.float32
    )
    background_means = np.empty(n_images, dtype=np.float64)
    background_stds = np.empty(n_images, dtype=np.float64)
    noise_psd_sum = np.zeros(basis.pts.shape[0], dtype=np.float64)
    load_psd_started = time.monotonic()

    def load_chunk(positions):
        raw = particles.read(source_rows[positions])
        fourier = downsample_fourier_images(raw, args.denoise_res)
        lowres = centered_ifft2(fourier)
        return positions, lowres.real

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, n_images, args.batch_size):
            end = min(start + args.batch_size, n_images)
            chunks = [
                part for part in np.array_split(np.arange(start, end), args.workers)
                if len(part)
            ]
            for positions, images in pool.map(load_chunk, chunks):
                observations[positions] = images
            images, means, stds = subtract_background_mean(observations[start:end], args.background_radius)
            observations[start:end] = images
            background_means[start:end] = means
            background_stds[start:end] = stds
            noise_psd_sum += np.sum(psd_estimator(observations[start:end]), axis=0, dtype=np.float64)
            print(f"Loaded/downsampled/background/PSD {end:,}/{n_images:,}", flush=True)
    noise_psd = noise_psd_sum / n_images
    load_psd_seconds = time.monotonic() - load_psd_started
    if np.any(background_stds <= 0):
        raise FloatingPointError("selected-image background noise has zero variance")

    n_preview = min(8, n_images)
    preview_input = (-observations[:n_preview] if args.flip_sign
                     else observations[:n_preview].copy())
    source = SimpleNamespace(
        n=n_images,
        L=args.denoise_res,
        images=Image(observations[order]),
        unique_filters=filters,
        filter_indices=ctf_ids[order],
    )
    del observations
    options = {
        "whiten": True,
        "noise_psd": noise_psd,
        "store_noise_psd": True,
        "radius": args.background_radius,
        "batch_size": args.batch_size,
        "single_pass": True,
        "single_whiten_filter": True,
        "flip_sign": args.flip_sign,
        "correct_contrast": True,
        "subtract_background": False,
        "dtype": np.float32,
        "verbose": True,
    }
    fast_pca = FastPCA(source, basis, options)
    covariance_started = time.monotonic()
    mean_est, covar_est = fast_pca.estimate_mean_covar()
    covariance_seconds = time.monotonic() - covariance_started
    denoise_started = time.monotonic()
    denoised_grouped, contrast_grouped = denoise_fastpca_batches(
        fast_pca, mean_est, covar_est, args.batch_size)
    denoise_seconds = time.monotonic() - denoise_started
    denoised = np.asarray(denoised_grouped, dtype=np.float32)[inverse]
    contrast_est = np.asarray(contrast_grouped, dtype=np.float64)[inverse]
    if not np.all(np.isfinite(denoised)) or not np.all(np.isfinite(contrast_est)):
        raise FloatingPointError("FastPCA returned nonfinite images or contrasts")

    write_started = time.monotonic()
    stack_path = output / "denoised_images.mrcs"
    with mrcfile.new_mmap(stack_path, shape=denoised.shape, mrc_mode=2) as stack:
        stack.set_image_stack()
        stack.voxel_size = voxel_size
        stack.data[:] = denoised
        stack.update_header_stats()
    configured_data_folder = "" if args.data_folder is None else str(args.data_folder.resolve())
    stack_prefix = args.stack_prefix or ""
    manifest_path = output / "manifest.npz"
    np.savez_compressed(
        manifest_path,
        dataset=args.dataset_label,
        star_path=str(particles.star_path),
        data_folder=configured_data_folder,
        stack_prefix=stack_prefix,
        resolved_stack_paths=np.asarray([str(path.resolve()) for path in particles.stack_paths]),
        source_fingerprint=source_fingerprint,
        source_rows=source_rows,
        image_names=image_names,
        n_total=particles.n,
        n_images=n_images,
        seed=args.seed,
        sampling=sampling,
        native_res=particles.res,
        denoise_res=args.denoise_res,
        vox_native=particles.pixel_size,
        metadata_pixel_size=particles.metadata_pixel_size,
        native_pixel_size_overridden=args.native_pixel_size is not None,
        vox_denoise=voxel_size,
        denoised_stack=str(stack_path),
        ctf_parameter_names=np.asarray(CTF_COLUMNS),
        ctf_parameters=parameters,
        ctf_phase_shifts=phase_shifts,
        exact_mean_defocus=exact_mean,
        effective_mean_defocus=effective_mean,
        mean_defocus_quantization_error=effective_mean - exact_mean,
        ctf_mean_defocus_bin=args.ctf_mean_defocus_bin,
        effective_radial_ctf_parameter_names=np.asarray(RADIAL_CTF_COLUMNS),
        effective_radial_ctf_parameters=radial_parameters,
        filter_indices=ctf_ids,
        grouped_to_selected=order,
        selected_to_grouped=inverse,
        grouped_source_rows=source_rows[order],
        run_starts=starts,
        run_sizes=counts,
        run_filter_ids=np.arange(len(radial_parameters)),
        astigmatism_abs_u_minus_v=astigmatism,
        astigmatism_phase_error_bound_at_denoise_nyquist=astig_bound,
        quantization_phase_error_bound_at_denoise_nyquist=quantization_bound,
        combined_phase_error_bound_at_denoise_nyquist=astig_bound + quantization_bound,
        noise_psd=noise_psd,
        noise_psd_sampling_points=basis.pts,
        flip_sign=args.flip_sign,
        background_radius=args.background_radius,
        background_means=background_means,
        background_stds=background_stds,
        contrast_est=contrast_est,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        fastpca_backend=fastpca_backend,
        fle_precision=basis.precision,
        eps=args.eps,
        python_version=sys.version,
        numpy_version=np.__version__,
        aspire_version=aspire.__version__,
        fast_pca_commit=git_commit(fast_pca_root),
        fle_2d_commit=git_commit(fle_root),
        kam_cryoem_commit=git_commit(ROOT),
    )

    import matplotlib.pyplot as plt
    from .utils import show_particle_image

    fig, axes = plt.subplots(2, n_preview, figsize=(2.1 * n_preview, 4.2), squeeze=False)
    for row, images in enumerate((preview_input, denoised[:n_preview])):
        for column, image in enumerate(images):
            show_particle_image(axes[row, column], image)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(output / "denoising_preview.pdf", bbox_inches="tight")
    plt.close(fig)
    write_seconds = time.monotonic() - write_started

    report = {
        "dataset": args.dataset_label,
        "device": args.device,
        "fastpca_backend": fastpca_backend,
        "fle_precision": basis.precision,
        "fle_max_order": int(args.denoise_max_order),
        "fle_n_coefficients": int(np.sum(basis.blk_size)),
        "n_images": n_images,
        "n_total": particles.n,
        "seed": args.seed,
        "sampling": sampling,
        "star_path": str(particles.star_path),
        "data_folder_argument": configured_data_folder,
        "stack_prefix_argument": stack_prefix,
        "absolute_stack_paths_in_star": bool(
            all(Path(name.split("@", 1)[1]).is_absolute() for name in image_names)
        ),
        "source_fingerprint": source_fingerprint,
        "native_res": particles.res,
        "denoise_res": args.denoise_res,
        "native_pixel_size": particles.pixel_size,
        "star_pixel_size": particles.metadata_pixel_size,
        "native_pixel_size_overridden": args.native_pixel_size is not None,
        "denoise_pixel_size": voxel_size,
        "ctf_mean_defocus_bin_angstrom": args.ctf_mean_defocus_bin,
        "ctf_group_count": len(radial_parameters),
        "ctf_group_size_min": int(counts.min()),
        "ctf_group_size_median": float(np.median(counts)),
        "ctf_group_size_max": int(counts.max()),
        "astigmatism_abs_u_minus_v_median": float(np.median(astigmatism)),
        "astigmatism_abs_u_minus_v_max": float(np.max(astigmatism)),
        "astigmatism_phase_bound_nyquist_median": float(np.median(astig_bound)),
        "astigmatism_phase_bound_nyquist_max": float(np.max(astig_bound)),
        "quantization_phase_bound_nyquist_median": float(np.median(quantization_bound)),
        "quantization_phase_bound_nyquist_max": float(np.max(quantization_bound)),
        "combined_phase_bound_nyquist_max": float(
            np.max(astig_bound + quantization_bound)
        ),
        "flip_sign": args.flip_sign,
        "output_stack": str(stack_path),
        "manifest": str(manifest_path),
        "elapsed_seconds": time.monotonic() - started,
        "timings_seconds": {
            "basis_setup": basis_seconds,
            "load_downsample_and_noise_psd": load_psd_seconds,
            "mean_and_covariance": covariance_seconds,
            "denoise": denoise_seconds,
            "write_stack_manifest_and_preview": write_seconds,
            "fastpca_basis_expansion_counter": float(
                fast_pca.t_basis_expansion),
            "fastpca_ctf_expansion_counter": float(
                fast_pca.t_ctf_expansion),
            "fastpca_covariance_counter": float(fast_pca.t_cov_est),
            "fastpca_wiener_counter": float(fast_pca.t_denoise),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Saved {n_images:,} row-audited, unwhitened denoised images: {stack_path}",
        flush=True,
    )
    print(
        f"Manifest: {manifest_path}; elapsed {report['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return FastCWFDenoiseResult(
        output_dir=output,
        stack_path=stack_path,
        manifest_path=manifest_path,
        report_path=report_path,
        n_images=n_images,
        n_total=particles.n,
        native_res=particles.res,
        denoise_res=args.denoise_res,
        native_pixel_size=particles.pixel_size,
        denoise_pixel_size=voxel_size,
        elapsed_seconds=report["elapsed_seconds"],
    )


# ===========================================================================
# CUDA FLE transforms for FastPCA
# ===========================================================================

def _cuda_modules():
    """Import the CUDA dependencies lazily."""
    import cupy as cp
    import cupyx.scipy.fft as cufft
    import cupyx.scipy.sparse as cusparse
    import cufinufft
    return cp, cufft, cusparse, cufinufft


_GPU_FLE = {}


def _gpu_fle_components():
    """Define the CUDA FLE basis on first use.

    ``fle_2d`` must already be importable; :func:`denoise_relion_particles`
    puts the configured roots on ``sys.path`` before calling this.
    """
    if _GPU_FLE:
        return _GPU_FLE["basis"], _GPU_FLE["psd"]
    from collections import OrderedDict  # noqa: F401 - used by the class body
    import threading  # noqa: F401 - used by the class body
    from fle_2d import FLEBasis2D

    class GPUFLEBasis2D(FLEBasis2D):
        """FLE basis with CUDA forward and adjoint evaluation.

        Inputs and outputs of :meth:`evaluate_t` and :meth:`evaluate` are NumPy
        arrays, matching ``FLEBasis2D`` and FastPCA.  Intermediate arrays stay on
        ``device_id``.  cuFINUFFT plans are cached by transform type and batch size;
        the cache is bounded because each plan owns device-side work buffers.

        Parameters other than ``device_id`` and ``max_cached_plans`` are passed to
        ``FLEBasis2D`` unchanged.  Both FLE precisions are supported.
        """

        def __init__(self, *args, device_id=0, max_cached_plans=8, **kwargs):
            self.device_id = int(device_id)
            self.max_cached_plans = int(max_cached_plans)
            cp, cufft, cusparse, cufinufft = _cuda_modules()
            self._cp = cp
            self._cufft = cufft
            self._cusparse = cusparse
            self._cufinufft = cufinufft

            # FLE precomputation is CPU-only and establishes the canonical basis.
            with cp.cuda.Device(self.device_id):
                super().__init__(*args, **kwargs)
                self._real_dtype = (
                    np.dtype(np.float32) if self.precision == "single"
                    else np.dtype(np.float64)
                )
                self._complex_dtype = (
                    np.dtype(np.complex64) if self.precision == "single"
                    else np.dtype(np.complex128)
                )
                self._cufinufft_dtype = self._complex_dtype.name
                self._upload_operators()

            # The parent creates single-transform CPU FINUFFT plans.  They are not
            # used by this subclass and retaining them wastes host memory.
            self.plan1 = None
            self.plan2 = None
            self._plan_cache = OrderedDict()
            self._plan_lock = threading.RLock()

        @property
        def gpu_real_dtype(self):
            """NumPy real dtype expected by the GPU transform."""
            return self._real_dtype

        def _upload_operators(self):
            cp = self._cp
            sparse = self._cusparse
            self._grid_x_gpu = cp.asarray(self.grid_x, dtype=self._real_dtype)
            self._grid_y_gpu = cp.asarray(self.grid_y, dtype=self._real_dtype)
            self._disk_outside_gpu = cp.asarray(self.idx, dtype=cp.bool_)
            self._nus_gpu = cp.asarray(self.nus, dtype=cp.int64)
            self._forward_phase_gpu = cp.asarray(
                (1j) ** self.nus, dtype=self._complex_dtype
            ).reshape(-1, 1)
            self._adjoint_phase_gpu = cp.asarray(
                (-1j) ** self.nus, dtype=self._complex_dtype
            ).reshape(1, 1, -1)
            self._cs_gpu = cp.asarray(self.cs, dtype=cp.float64)
            self._idx_list_gpu = [cp.asarray(index, dtype=cp.int64)
                                  for index in self.idx_list]
            self._a3_gpu = [sparse.csr_matrix(matrix) for matrix in self.A3]
            self._a3_t_gpu = [sparse.csr_matrix(matrix) for matrix in self.A3_T]
            self._c2r_nus_gpu = sparse.csr_matrix(self.c2r_nus)
            self._r2c_nus_gpu = sparse.csr_matrix(self.r2c_nus)
            self._c2r_gpu = sparse.csr_matrix(self.c2r)
            self._r2c_gpu = sparse.csr_matrix(self.r2c)
            self._dense_b_gpu = None
            if self.L < 16:
                self._dense_b_gpu = cp.asarray(self.B)

        def _plan(self, nufft_type, n_trans):
            n_trans = int(n_trans)
            key = (nufft_type, n_trans)
            plan = self._plan_cache.get(key)
            if plan is not None:
                self._plan_cache.move_to_end(key)
                return plan
            isign = 1 if nufft_type == 1 else -1
            plan = self._cufinufft.Plan(
                nufft_type,
                (self.L, self.L),
                n_trans=n_trans,
                isign=isign,
                eps=self.eps,
                dtype=self._cufinufft_dtype,
                gpu_device_id=self.device_id,
            )
            plan.setpts(self._grid_x_gpu, self._grid_y_gpu)
            self._plan_cache[key] = plan
            while len(self._plan_cache) > self.max_cached_plans:
                self._plan_cache.popitem(last=False)
            return plan

        def _execute_plan(self, nufft_type, data):
            """Execute a cached plan while preventing concurrent plan reuse."""
            cp = self._cp
            n_trans = int(data.shape[0])
            with self._plan_lock, cp.cuda.Device(self.device_id):
                plan = self._plan(nufft_type, n_trans)
                output = plan.execute(cp.ascontiguousarray(data))
                # cuFINUFFT execution may be asynchronous.  A cached plan must
                # not be reused by another Python thread until its work has
                # completed on the current stream.
                cp.cuda.get_current_stream().synchronize()
                return output

        def _forward_gpu(self, images):
            """Apply the real FLE adjoint to a batch already on the GPU."""
            cp = self._cp
            cufft = self._cufft
            n_images = int(images.shape[0])
            images = cp.asarray(images, dtype=self._complex_dtype)
            polar_half = self._execute_plan(2, images) * self.h**2
            polar_half = polar_half.reshape(
                n_images, self.n_radial, self.n_angular // 2
            )
            polar = cp.empty(
                (n_images, self.n_radial, self.n_angular),
                dtype=self._complex_dtype,
            )
            polar[:, :, : self.n_angular // 2] = polar_half
            polar[:, :, self.n_angular // 2 :] = cp.conj(polar_half)

            angular = cp.fft.fft(polar, axis=2) / self.n_angular
            angular = angular[:, :, self._nus_gpu]
            angular = cp.swapaxes(angular, 0, 2).reshape(
                len(self.nus), self.n_radial * n_images
            )
            angular *= self._forward_phase_gpu
            angular = self._c2r_nus_gpu @ angular
            angular = cp.real(
                cp.swapaxes(
                    angular.reshape(-1, self.n_radial, n_images), 0, 2
                )
            )

            if self.n_interp > self.n_radial:
                angular = (
                    cufft.dct(angular, axis=1, type=2) / (2 * self.n_radial)
                )
                angular = cp.concatenate((angular, cp.zeros_like(angular)), axis=1)
                angular = (
                    cufft.idct(angular, axis=1, type=2) * 2 * angular.shape[1]
                )

            angular = cp.moveaxis(angular, 0, -1)
            coefficients = cp.zeros(
                (self.ne, n_images), dtype=self._real_dtype
            )
            for order in range(self.ndmax + 1):
                coefficients[self._idx_list_gpu[order]] = (
                    self._a3_gpu[order] @ angular[:, order, :]
                )
            coefficients = coefficients.T
            # FLE stores ``cs`` in float64 even for a single-precision basis, so
            # this deliberately matches the reference output promotion.
            coefficients = coefficients * self._cs_gpu / self.h
            if self.complexmode:
                coefficients = (self._r2c_gpu @ coefficients.T).T
            return coefficients

        def _adjoint_gpu(self, coefficients):
            """Apply the real FLE synthesis map to GPU coefficients."""
            cp = self._cp
            cufft = self._cufft
            n_images = int(coefficients.shape[0])
            if self.complexmode:
                coefficients = (self._c2r_gpu @ coefficients.T).T
            coefficients = cp.real(coefficients)

            scaled = (coefficients * self.h * self._cs_gpu).T
            radial = cp.zeros(
                (self.n_interp, 2 * self.nmax + 1, n_images),
                dtype=self._real_dtype,
                order="F",
            )
            for order in range(self.ndmax + 1):
                radial[:, order, :] = (
                    self._a3_t_gpu[order] @ scaled[self._idx_list_gpu[order]]
                )
            radial = cp.moveaxis(radial, -1, 0)
            if self.n_interp > self.n_radial:
                radial = cufft.dct(radial, axis=1, type=2)
                radial = radial[:, : self.n_radial, :]
                radial = cufft.idct(radial, axis=1, type=2)

            angular = cp.swapaxes(radial, 0, 2).reshape(
                2 * self.nmax + 1, self.n_radial * n_images
            )
            angular = self._r2c_nus_gpu @ angular
            angular = cp.swapaxes(
                angular.reshape(-1, self.n_radial, n_images), 0, 2
            )
            polar = cp.zeros(
                (n_images, self.n_radial, self.n_angular),
                dtype=self._complex_dtype,
            )
            polar[:, :, self._nus_gpu] = self._adjoint_phase_gpu * angular
            polar = cp.fft.ifft(polar, axis=2).astype(self._complex_dtype)

            images = self._execute_plan(
                1, polar[:, :, : self.n_angular // 2].reshape(n_images, -1)
            )
            images = cp.real(images + cp.conj(images)).reshape(
                n_images, self.L, self.L
            )
            images[:, self._disk_outside_gpu] = 0
            return images

        def evaluate_t(self, f):
            """Evaluate FLE analysis coefficients on CUDA and return NumPy data."""
            array = np.asarray(f)
            single = array.size == self.L1**2
            n_images = array.size // self.L1**2
            array = np.asarray(array.real).reshape(n_images, self.L1, self.L1)
            if self.L > self.L1:
                array = np.pad(array, ((0, 0), (1, 0), (1, 0)))

            cp = self._cp
            with cp.cuda.Device(self.device_id):
                images = cp.asarray(array, dtype=self._real_dtype)
                if self.L < 16:
                    result = (
                        self._dense_b_gpu.T
                        @ images.reshape(n_images, self.L**2).T
                    ).T
                else:
                    images[:, self._disk_outside_gpu] = 0
                    result = self._forward_gpu(images)
                result = cp.asnumpy(result)
            return np.squeeze(result) if single else result

        def evaluate(self, a):
            """Synthesize images from FLE coefficients on CUDA and return NumPy data."""
            array = np.asarray(a)
            single = array.size == self.ne
            n_images = array.size // self.ne
            array = array.reshape(n_images, self.ne)

            cp = self._cp
            with cp.cuda.Device(self.device_id):
                coefficients = cp.asarray(array)
                if self.L < 16:
                    if self.complexmode:
                        coefficients = (self._c2r_gpu @ coefficients.T).T
                    images = (
                        self._dense_b_gpu @ cp.real(coefficients).T
                    ).T.reshape(n_images, self.L, self.L)
                else:
                    images = self._adjoint_gpu(coefficients)
                if self.L > self.L1:
                    images = images[:, 1:, 1:]
                result = cp.asnumpy(images)
            return np.squeeze(result) if single else result

        def nufft_type2(self, images):
            """Evaluate the basis polar type-2 NUFFT for a real/complex batch.

            This low-level method is used by :func:`estimate_radial_psd_gpu` and
            intentionally returns a CuPy array.
            """
            cp = self._cp
            with cp.cuda.Device(self.device_id):
                values = cp.asarray(images, dtype=self._complex_dtype)
                return self._execute_plan(2, values)

    def estimate_radial_psd_gpu(imgs, basis, bgRadius=0.9, dtype=None):
        """Estimate radial background PSD with the basis' cached cuFINUFFT plan.

        The formula and output layout match
        ``fast-cryoEM-PCA.utils_cwf_fast_batch.estimate_radial_psd``.  ``basis``
        must be a :class:`GPUFLEBasis2D`; this requirement prevents an accidental
        CPU fallback in a requested GPU run.
        """
        images = np.asarray(imgs)
        requested_dtype = np.dtype(np.float64 if dtype is None else dtype)

        side = basis.L
        axis = np.arange(-side // 2, side // 2, dtype=requested_dtype) / (side // 2)
        radius = np.hypot(axis[:, None], axis[None, :])
        mask = radius >= bgRadius
        denominator = int(np.count_nonzero(mask))

        cp = basis._cp
        with cp.cuda.Device(basis.device_id):
            masked = cp.asarray(images.real, dtype=requested_dtype)
            masked *= cp.asarray(mask, dtype=requested_dtype)
            polar = basis.nufft_type2(masked)
            polar = polar.reshape(
                len(images), basis.n_radial, basis.n_angular // 2
            )
            psd = cp.mean(cp.abs(polar) ** 2 / denominator, axis=2)
            result = cp.asnumpy(psd)
        return result

    _GPU_FLE["basis"] = GPUFLEBasis2D
    _GPU_FLE["psd"] = estimate_radial_psd_gpu
    return GPUFLEBasis2D, estimate_radial_psd_gpu
