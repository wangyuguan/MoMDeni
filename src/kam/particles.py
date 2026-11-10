"""Particle images: reading RELION stacks, CWF denoising and the selection of the least crowded particles.

:class:`RelionParticles` reads the images of a STAR file without applying poses or intensity corrections. :func:`denoise_relion_particles`
Fourier-crops them, denoises them by FastPCA covariance Wiener filtering (radial CTF at the mean defocus) and writes CTF-corrected,
amplitude-preserving images together with a manifest of their CTF parameters and provenance. :func:`select_particles` keeps the particles
whose denoised images show the least signal of neighbouring particles. No poses, reference maps or image shifts are used.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle
import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .utils import ROOT, centered_fft2, centered_ifft2, git_commit, sha256_file, show_particle_image

# git submodules checked out next to src/ (git clone --recursive)
FAST_PCA_ROOT = ROOT / "fast-cryoEM-PCA"
FLE_ROOT = ROOT / "fle_2d"
DEFAULT_DENOISE_RES = 64
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


class RelionParticles:
    """Read particle pixels without applying STAR poses or intensity corrections.

    Stack paths are relative to ``data_folder`` (the STAR directory by default).
    ``stack_prefix`` optionally strips one exact leading relative directory
    prefix from every STAR stack path, for datasets stored in a flattened root.
    It must be slash-separated, without a trailing slash or ``.``/``..`` parts.
    Original image names and metadata are retained unchanged.
    ``pixel_size`` explicitly overrides the sampling in Angstroms; the STAR
    value remains available as ``metadata_pixel_size`` and in ``metadata``.
    All particles must share a square grid and pixel size. Optics metadata are
    merged into the particle table with ASPIRE's RELION STAR reader.
    """

    def __init__(self, star_path, data_folder=None, stack_prefix=None, pixel_size=None):
        from aspire.utils import RelionStarFile

        self.star_path = Path(star_path).resolve(strict=True)
        root = self.star_path.parent if data_folder is None else Path(data_folder).resolve(strict=True)
        self.metadata = RelionStarFile(str(self.star_path)).get_merged_data_block()
        self.image_names = np.asarray(self.metadata["_rlnImageName"], dtype=str)
        self.n = len(self.image_names)
        parts = [name.split("@", 1) for name in self.image_names]
        self.stack_indices = np.array([int(part[0]) - 1 for part in parts])
        names, self.stack_ids = np.unique([part[1] for part in parts], return_inverse=True)
        if stack_prefix is not None:
            names = [name[len(stack_prefix) + 1:] for name in names]
        self.stack_paths = [root / name for name in names]
        if "_rlnImagePixelSize" in self.metadata:
            self.metadata_pixel_size = float(self.column("_rlnImagePixelSize")[0])
        else:
            self.metadata_pixel_size = float(1e4 * self.column("_rlnDetectorPixelSize")[0] / self.column("_rlnMagnification")[0])
        self.pixel_size = self.metadata_pixel_size if pixel_size is None else float(pixel_size)
        with mrcfile.mmap(self.stack_paths[0], mode="r") as stack:
            self.res = int(stack.data.shape[-1])

    def column(self, name, default=None):
        """A numeric metadata column, or ``default`` for every particle if the STAR file does not have it."""
        if name not in self.metadata and default is not None:
            return np.full(self.n, default, dtype=np.float64)
        return np.asarray(self.metadata[name], dtype=np.float64)

    def read(self, indices):
        """Read zero-based STAR rows in the requested order, using memory maps."""
        indices = np.asarray(indices)
        images = np.empty((len(indices), self.res, self.res), dtype=np.float32)
        for stack_id in np.unique(self.stack_ids[indices]):
            positions = np.flatnonzero(self.stack_ids[indices] == stack_id)
            stack_indices = self.stack_indices[indices[positions]]
            with mrcfile.mmap(self.stack_paths[stack_id], mode="r") as stack:
                data = stack.data
                if data.ndim == 2:
                    data = data[None]
                images[positions] = data[stack_indices]
        return images

    def fingerprint(self):
        """Identify STAR contents and the referenced stacks without hashing all pixels."""
        digest = hashlib.sha256(self.star_path.read_bytes())
        for path in self.stack_paths:
            stat = path.stat()
            digest.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()


def downsample_fourier_images(images, ds_res):
    """Fourier crop raw particles, preserving their field of view and amplitude.

    The output contains centered, unnormalized 2-D Fourier coefficients.
    Multiplication by ``(ds_res/native_res)**2`` keeps the amplitude of the cropped coefficients.
    For even sizes, zero the unpaired Nyquist row/column; the reconstruction
    disk excludes these edges. No mean subtraction or normalization is applied.
    """
    images = np.asarray(images)
    native_res = images.shape[-1]
    ds_res = int(ds_res)
    start = native_res // 2 - ds_res // 2
    # NumPy 2 preserves float32 FFT precision; promote before transforming so
    # the cached complex128 moments retain an accurately accumulated DC term.
    fourier = centered_fft2(np.asarray(images, dtype=np.float64))
    cropped = fourier[:, start:start + ds_res, start:start + ds_res].copy()
    cropped *= (ds_res / native_res) ** 2
    if ds_res % 2 == 0:
        cropped[:, 0, :] = 0
        cropped[:, :, 0] = 0
    return cropped


# ===========================================================================
# CWF denoising
# ===========================================================================

@dataclass(frozen=True)
class FastCWFDenoiseConfig:
    """Inputs and numerical settings for :func:`denoise_relion_particles`."""

    dataset_label: str
    star_path: Path
    output_dir: Path
    data_folder: Path | None = None
    stack_prefix: str | None = None
    native_pixel_size: float | None = None
    denoise_res: int = DEFAULT_DENOISE_RES
    batch_size: int = 256
    workers: int = 1
    device: str = "gpu"
    background_radius: float = 0.9
    eps: float = 1e-3
    denoise_max_order: int = DEFAULT_DENOISE_MAX_ORDER
    ctf_mean_defocus_bin: float = 50.0
    flip_sign: bool = True


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


def denoise_relion_particles(config: FastCWFDenoiseConfig):
    """Denoise all particles of a RELION STAR file.

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
    from utils_cwf_fast_batch import estimate_radial_psd

    started = time.monotonic()
    particles = RelionParticles(
        args.star_path,
        data_folder=args.data_folder,
        stack_prefix=args.stack_prefix,
        pixel_size=args.native_pixel_size,
    )
    n_images = particles.n
    source_rows = np.arange(n_images, dtype=np.int64)
    source_fingerprint = particles.fingerprint()
    image_names = particles.image_names[source_rows].copy()
    parameters = np.column_stack(
        [particles.column(key)[source_rows] for key in CTF_COLUMNS]
    )
    phase_shifts = particles.column("_rlnPhaseShift", default=0.0)[source_rows]
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
    print(f"Denoising all {n_images:,} particles of the STAR file", flush=True)
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
        n_images=n_images,
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


# ===========================================================================
# Low-crowding particle selection (reference-free: no poses or maps)
# ===========================================================================

# Fixed crowding score: Gaussian smoothing (pixels of the denoised images), the central disk and the outer annulus (fractions of the image
# radius), the top fractions of their positive pixels whose RMS is compared, and the number of contrast bins of the stratified allocation.
SMOOTH_SIGMA = 1.1
INNER_RADIUS = 0.50
OUTER_RADIUS = (0.62, 0.94)
INNER_TOP_FRACTION = 0.20
OUTER_TOP_FRACTION = 0.10
CONTRAST_BINS = 4
DEFAULT_N_SELECT = 10_000
DEFAULT_NEIGHBOR_FRACTION = 280.0 / 380.0


@dataclass(frozen=True)
class LowCrowdingSelection:
    """Outputs and diagnostics from :func:`select_low_crowding_particles`.

    Positions index the input image stack.  ``geometry_pass`` records the
    joint nearest-neighbor and crowding-score gate before stratified sampling.
    The remaining arrays make the micrograph-by-contrast allocation fully
    reproducible and auditable.
    """

    selected_positions: np.ndarray
    rejected_positions: np.ndarray
    geometry_pass: np.ndarray
    crowding_scores: np.ndarray
    inner_strength: np.ndarray
    outer_strength: np.ndarray
    score_ceiling: float
    contrast_bin_boundaries: np.ndarray
    contrast_bin_ids: np.ndarray
    micrograph_ids: np.ndarray
    stratum_codes: np.ndarray
    unique_stratum_codes: np.ndarray
    stratum_population: np.ndarray
    stratum_geometry_capacity: np.ndarray
    stratum_selection_quota: np.ndarray


def nearest_particle_distances(micrograph_names, coordinate_x, coordinate_y):
    """Return the nearest distinct pick distance within each micrograph.

    Parameters are one-dimensional arrays in the native particle-table order.
    Distances are in the same units as the coordinates.  A particle that is
    alone on its micrograph receives an infinite distance.
    """

    names = np.asarray(micrograph_names, dtype=str)
    x = np.asarray(coordinate_x, dtype=np.float64)
    y = np.asarray(coordinate_y, dtype=np.float64)

    _, group_ids = np.unique(names, return_inverse=True)
    distances = np.full(len(names), np.inf, dtype=np.float64)
    for group_id in range(int(group_ids.max()) + 1):
        rows = np.flatnonzero(group_ids == group_id)
        if len(rows) < 2:
            continue
        coordinates = np.column_stack((x[rows], y[rows]))
        values, _ = cKDTree(coordinates).query(coordinates, k=2)
        distances[rows] = values[:, 1]
    return distances


def select_low_crowding_particles(
        images, micrograph_names, nearest_neighbor_distances, contrast,
        n_select, *, minimum_neighbor_distance,
        max_score_quantile=0.95, auto_score_ceiling=False, batch_size=256,
        progress=None):
    """Select a low-crowding subset without poses or a reference volume.

    Each image is Gaussian-smoothed, and its crowding score is the top-RMS
    positive signal in an outer annulus divided by the corresponding signal in
    a central disk.  Images must pass both a nearest-pick-distance gate and a
    global score-quantile gate.  Among passing images, the requested count is
    allocated proportionally across micrograph-by-contrast-bin strata, subject
    to the capacity of each stratum, and the lowest scores are retained.

    ``images`` may be an in-memory array or an array-like memory map with shape
    ``(n, side, side)``.  ``nearest_neighbor_distances`` should normally be
    computed from the complete particle table with
    :func:`nearest_particle_distances` and then indexed into the image-stack
    order.  ``progress``, when supplied, is called as ``progress(done, total)``
    after periodic scoring batches.

    ``auto_score_ceiling=True`` ignores ``max_score_quantile`` and uses the
    tightest score ceiling that still leaves ``n_select`` neighbor-gate
    candidates, i.e. the ``n_select``-th smallest score among them.  When fewer
    images pass the gates than ``n_select``, all passing images are selected.

    The selection is deterministic: ties are resolved by input position and
    returned positions are sorted.  No random sampling is performed here.
    """

    count, side, _ = map(int, images.shape)
    names = np.asarray(micrograph_names, dtype=str)
    distances = np.asarray(nearest_neighbor_distances, dtype=np.float64)
    contrast = np.asarray(contrast, dtype=np.float64)
    n_select, contrast_bins, batch_size = int(n_select), CONTRAST_BINS, int(batch_size)

    coordinates = np.arange(side, dtype=np.float64) - (side - 1) / 2
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt(xx ** 2 + yy ** 2) / (side / 2)
    inner_mask = radius <= INNER_RADIUS
    outer_mask = (radius >= OUTER_RADIUS[0]) & (radius <= OUTER_RADIUS[1])

    scores = np.empty(count, dtype=np.float64)
    inner_strength = np.empty(count, dtype=np.float64)
    outer_strength = np.empty(count, dtype=np.float64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        batch = np.array(images[start:end], dtype=np.float32, copy=True)
        smooth = gaussian_filter(
            batch, sigma=(0, SMOOTH_SIGMA, SMOOTH_SIGMA), mode="nearest")
        positive = np.maximum(smooth, 0.0)
        central = _top_rms(positive[:, inner_mask], INNER_TOP_FRACTION)
        peripheral = _top_rms(positive[:, outer_mask], OUTER_TOP_FRACTION)
        floor = max(float(np.median(central)) * 1e-6,
                    np.finfo(np.float32).tiny)
        scores[start:end] = peripheral / np.maximum(central, floor)
        inner_strength[start:end] = central
        outer_strength[start:end] = peripheral
        if progress is not None and (end == count or end % (10 * batch_size) == 0):
            progress(end, count)

    if auto_score_ceiling:
        candidates = np.sort(scores[distances >= minimum_neighbor_distance])
        n_select = min(n_select, len(candidates))
        score_ceiling = float(candidates[n_select - 1])
    else:
        score_ceiling = float(np.quantile(scores, max_score_quantile))
    geometry_pass = ((distances >= minimum_neighbor_distance)
                     & (scores <= score_ceiling))
    n_select = min(n_select, int(geometry_pass.sum()))

    boundaries = np.quantile(
        contrast, np.arange(1, contrast_bins) / contrast_bins)
    contrast_ids = np.searchsorted(boundaries, contrast, side="right")
    _, micrograph_ids = np.unique(names, return_inverse=True)
    stratum_codes = micrograph_ids * contrast_bins + contrast_ids
    unique_strata, inverse_strata = np.unique(stratum_codes, return_inverse=True)
    population = np.bincount(inverse_strata)
    capacity = np.bincount(
        inverse_strata[geometry_pass], minlength=len(unique_strata))
    quotas = _capped_proportional_quotas(population, capacity, n_select)

    selected_parts = []
    for stratum, quota in enumerate(quotas):
        if quota == 0:
            continue
        candidates = np.flatnonzero(geometry_pass & (inverse_strata == stratum))
        rank = np.lexsort((candidates, scores[candidates]))
        selected_parts.append(candidates[rank[:quota]])
    selected_positions = np.sort(np.concatenate(selected_parts))
    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[selected_positions] = True

    return LowCrowdingSelection(
        selected_positions=selected_positions,
        rejected_positions=np.flatnonzero(~selected_mask),
        geometry_pass=geometry_pass,
        crowding_scores=scores,
        inner_strength=inner_strength,
        outer_strength=outer_strength,
        score_ceiling=score_ceiling,
        contrast_bin_boundaries=boundaries,
        contrast_bin_ids=contrast_ids,
        micrograph_ids=micrograph_ids,
        stratum_codes=stratum_codes,
        unique_stratum_codes=unique_strata,
        stratum_population=population,
        stratum_geometry_capacity=capacity,
        stratum_selection_quota=quotas,
    )


def _top_rms(values, fraction):
    count = max(1, int(np.ceil(fraction * values.shape[1])))
    partitioned = np.partition(values, values.shape[1] - count, axis=1)
    return np.sqrt(np.mean(partitioned[:, -count:] ** 2, axis=1))


def _capped_proportional_quotas(population, capacity, total):
    """Allocate proportional integer quotas subject to stratum capacities."""

    population = np.asarray(population, dtype=np.int64)
    capacity = np.asarray(capacity, dtype=np.int64)
    if total == 0:
        return np.zeros_like(population)
    lower, upper = 0.0, max(1.0, total / max(int(population.sum()), 1))
    while np.minimum(capacity, upper * population).sum() < total:
        upper *= 2
    for _ in range(80):
        middle = (lower + upper) / 2
        if np.minimum(capacity, middle * population).sum() < total:
            lower = middle
        else:
            upper = middle
    desired = np.minimum(capacity, upper * population)
    quotas = np.floor(desired).astype(np.int64)
    remaining = total - int(quotas.sum())
    if remaining:
        eligible = np.flatnonzero(quotas < capacity)
        priority = desired[eligible] - quotas[eligible]
        order = eligible[np.lexsort((eligible, -priority))]
        quotas[order[:remaining]] += 1
    return quotas



def scalar(manifest, name):
    """One scalar of an NPZ manifest."""
    return np.asarray(manifest[name]).reshape(()).item()


def optional_text(manifest, name):
    """An optional path or prefix of a denoising manifest, which stores None as an empty string."""
    return str(scalar(manifest, name)) or None


def save_selection_diagnostic(images, selected, path, scores=None):
    """Save a random audit montage with a check or cross beside each image."""
    images = np.asarray(images)
    selected = np.asarray(selected, dtype=bool)
    if scores is not None:
        scores = np.asarray(scores, dtype=float)

    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))
    figure = plt.figure(figsize=(1.45 * columns, 1.25 * rows))
    outer = figure.add_gridspec(
        rows, columns, left=0.005, right=0.995, bottom=0.005, top=0.995,
        wspace=0.08, hspace=0.08)
    for position in range(rows * columns):
        row, column = divmod(position, columns)
        if position >= len(images):
            axis = figure.add_subplot(outer[row, column])
            axis.set_axis_off()
            continue
        inner = outer[row, column].subgridspec(
            1, 2, width_ratios=(5.0, 1.0), wspace=0.02)
        image_axis = figure.add_subplot(inner[0, 0])
        mark_axis = figure.add_subplot(inner[0, 1])
        image = images[position]
        color = "#16883b" if selected[position] else "#c62828"
        show_particle_image(image_axis, image)
        image_axis.add_patch(Rectangle(
            (-0.5, -0.5), image.shape[1], image.shape[0], fill=False,
            edgecolor=color, linewidth=1.8))
        image_axis.set_axis_off()
        if scores is not None:
            image_axis.text(
                0.03, 0.03, f"{scores[position]:.3f}",
                transform=image_axis.transAxes, fontsize=7, color="white",
                bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none",
                      "pad": 1})
        mark_axis.set_xlim(0.0, 1.0)
        mark_axis.set_ylim(0.0, 1.0)
        mark_axis.set_aspect("equal")
        mark_axis.set_axis_off()
        if selected[position]:
            mark_axis.plot(
                (0.10, 0.38, 0.90), (0.50, 0.20, 0.82), color=color,
                linewidth=3.2, solid_capstyle="round", solid_joinstyle="round")
        else:
            mark_axis.plot(
                (0.16, 0.84), (0.16, 0.84), color=color, linewidth=3.2,
                solid_capstyle="round")
            mark_axis.plot(
                (0.16, 0.84), (0.84, 0.16), color=color, linewidth=3.2,
                solid_capstyle="round")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(figure)


SELECTION_OUTPUT_NAMES = (
    "selection.npz", "selection_report.json", "selected_particles.tsv",
    "selection_diagnostic.pdf", "selection_scores.pdf",
    "selection_selected_sample.pdf",
)


def _require_selection_outputs_absent(output: Path) -> None:
    """Allow denoising files here without overwriting a previous selection."""
    existing = [name for name in SELECTION_OUTPUT_NAMES
                if (output / name).exists()]
    if existing:
        raise FileExistsError(
            f"selection output already exists in {output}: {', '.join(existing)}")


def save_selected_sample(images, scores, path, columns=20):
    """Save a montage of selected particles with the crowding score in the corner."""
    images = np.asarray(images)
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(1.5 * columns, 1.5 * rows + 0.4), squeeze=False)
    for axis in axes.ravel():
        axis.set_axis_off()
    for axis, image, score in zip(axes.ravel(), images, scores):
        show_particle_image(axis, image)
        axis.text(0.03, 0.03, f"{score:.2f}", transform=axis.transAxes,
                  fontsize=6, color="white",
                  bbox={"facecolor": "black", "alpha": 0.6,
                        "edgecolor": "none", "pad": 1})
    figure.suptitle(f"{len(images)} random selected particles "
                    "(corner: crowding score)", y=0.998)
    figure.subplots_adjust(0.003, 0.003, 0.997, 0.97, 0.03, 0.03)
    figure.savefig(path.with_suffix(".pdf"), dpi=100)
    plt.close(figure)


def save_selection_scores(scores, selected, score_ceiling, path):
    """Compare selected and rejected image scores using the same histogram bins."""
    scores = np.asarray(scores, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    edges = np.histogram_bin_edges(scores, bins=60)
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.hist([scores[selected], scores[~selected]], bins=edges, stacked=True,
              color=["#16883b", "#bdbdbd"], label=["Selected", "Not selected"])
    if np.isfinite(score_ceiling):
        axis.axvline(score_ceiling, color="#c62828", linestyle="--",
                     linewidth=1.0, label="Score ceiling")
    axis.set(xlabel="Crowding score (outer / central signal)", ylabel="Images")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def manifest_geometry(manifest, manifest_path):
    """The denoised stack of a denoising manifest with its image count, native and denoised side and native pixel size."""
    stack_path = Path(str(scalar(manifest, "denoised_stack")))
    return (stack_path, int(scalar(manifest, "n_images")), int(scalar(manifest, "native_res")),
            int(scalar(manifest, "denoise_res")), float(scalar(manifest, "vox_native")))


def select_particles(*, denoising_dir, output_dir, n_select, n_preview, seed, neighbor_distance_fraction, max_score_quantile,
                     auto_score_quantile, batch_size):
    """Select the least crowded particles of a denoising output and write ``selection.npz``, ``selection_report.json``,
    ``selected_particles.tsv`` and the diagnostic plots to ``output_dir`` (which may be the denoising directory itself).

    ``auto_score_quantile`` takes the tightest crowding-score ceiling that leaves ``n_select`` particles; otherwise the ceiling is the
    ``max_score_quantile`` quantile of the scores."""
    started = time.monotonic()
    output = Path(output_dir).resolve()
    denoising = Path(denoising_dir).resolve(strict=True)
    _require_selection_outputs_absent(output)
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = (denoising / "manifest.npz").resolve(strict=True)
    with np.load(manifest_path, allow_pickle=False) as saved:
        manifest = {name: saved[name] for name in saved.files}
    stack_path, count, native_res, side, vox_native = manifest_geometry(manifest, manifest_path)
    input_loading_done = time.monotonic()

    source_rows = np.asarray(manifest["source_rows"], dtype=np.int64)
    image_names = np.asarray(manifest["image_names"], dtype=str)

    star_path = Path(str(scalar(manifest, "star_path")))
    data_folder = optional_text(manifest, "data_folder")
    stack_prefix = optional_text(manifest, "stack_prefix")
    particles = RelionParticles(
        star_path, data_folder=data_folder, stack_prefix=stack_prefix,
        pixel_size=vox_native)
    micrograph_names = np.asarray(
        particles.metadata["_rlnMicrographName"], dtype=str)
    # The pick coordinates are in micrograph pixels: their size is the optics column _rlnMicrographPixelSize (written for
    # Fourier-cropped particle stacks), otherwise the particle pixel size. The threshold is the fraction of the field of view.
    coordinate_pixel = float(particles.column("_rlnMicrographPixelSize", vox_native)[0])
    neighbor_distance = neighbor_distance_fraction * native_res * vox_native / coordinate_pixel
    all_neighbor_distances = nearest_particle_distances(
        micrograph_names, particles.column("_rlnCoordinateX"),
        particles.column("_rlnCoordinateY"))
    neighbor_distances = all_neighbor_distances[source_rows]
    contrast = np.asarray(manifest["contrast_est"], dtype=np.float64)
    metadata_setup_done = time.monotonic()
    with mrcfile.mmap(stack_path, mode="r", permissive=False) as stack:
        selection = select_low_crowding_particles(
            stack.data, micrograph_names[source_rows], neighbor_distances,
            contrast, n_select,
            minimum_neighbor_distance=neighbor_distance,
            max_score_quantile=max_score_quantile,
            auto_score_ceiling=auto_score_quantile,
            batch_size=batch_size,
            progress=lambda done, total: print(
                f"  scored {done:,}/{total:,}", flush=True))

    selected_positions = selection.selected_positions
    rejected_positions = selection.rejected_positions
    geometry_pass = selection.geometry_pass
    scores = selection.crowding_scores
    inner_strength = selection.inner_strength
    outer_strength = selection.outer_strength
    score_ceiling = selection.score_ceiling
    boundaries = selection.contrast_bin_boundaries
    contrast_ids = selection.contrast_bin_ids
    micrograph_ids = selection.micrograph_ids
    stratum_codes = selection.stratum_codes
    unique_strata = selection.unique_stratum_codes
    population = selection.stratum_population
    capacity = selection.stratum_geometry_capacity
    quotas = selection.stratum_selection_quota
    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[selected_positions] = True
    selection_done = time.monotonic()

    diagnostic_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x53454C]))
    diagnostic_positions = diagnostic_rng.choice(
        count, n_preview, replace=False)
    diagnostic_selected = selected_mask[diagnostic_positions]
    with mrcfile.mmap(stack_path, mode="r", permissive=False) as stack:
        diagnostic_images = np.array(
            stack.data[diagnostic_positions], dtype=np.float32, copy=True)
    save_selection_diagnostic(
        diagnostic_images, diagnostic_selected,
        output / "selection_diagnostic", scores=scores[diagnostic_positions])
    save_selection_scores(
        scores, selected_mask, score_ceiling, output / "selection_scores")
    sample_positions = np.sort(np.random.default_rng(
        np.random.SeedSequence([seed, 0x53414D])).choice(
            selected_positions, min(200, len(selected_positions)), replace=False))
    with mrcfile.mmap(stack_path, mode="r", permissive=False) as stack:
        sample_images = np.array(
            stack.data[sample_positions], dtype=np.float32, copy=True)
    save_selected_sample(
        sample_images, scores[sample_positions], output / "selection_selected_sample")
    diagnostic_pdf = output / "selection_diagnostic.pdf"

    dataset = optional_text(manifest, "dataset")
    source_fingerprint = optional_text(manifest, "source_fingerprint")
    manifest_hash = sha256_file(manifest_path)
    np.savez_compressed(
        output / "selection.npz",
        selected_positions=selected_positions,
        selected_source_rows=source_rows[selected_positions],
        selected_image_names=image_names[selected_positions],
        rejected_positions=rejected_positions,
        geometry_pass_positions=np.flatnonzero(geometry_pass),
        diagnostic_positions=diagnostic_positions,
        diagnostic_selected=diagnostic_selected,
        crowding_scores=scores,
        inner_strength=inner_strength,
        outer_strength=outer_strength,
        nearest_neighbor_distance_coordinate_pixels=neighbor_distances,
        contrast_bin_boundaries=boundaries,
        contrast_bin_ids=contrast_ids,
        stratum_codes=stratum_codes,
        unique_stratum_codes=unique_strata,
        stratum_population=population,
        stratum_geometry_capacity=capacity,
        stratum_selection_quota=quotas,
        effective_contrast_bins=len(np.unique(contrast_ids)),
        dataset="" if dataset is None else dataset,
        source_fingerprint="" if source_fingerprint is None else source_fingerprint,
        input_manifest=str(manifest_path),
        input_manifest_sha256=manifest_hash,
        input_stack=str(stack_path),
        native_res=native_res,
        native_pixel_size_angstrom=vox_native,
        minimum_neighbor_distance_fraction=neighbor_distance_fraction,
        coordinate_pixel_size_angstrom=coordinate_pixel,
        minimum_neighbor_distance_coordinate_pixels=neighbor_distance,
    )
    with (output / "selected_particles.tsv").open("w") as stream:
        stream.write(
            "denoised_position\tstar_row_zero_based\timage_name\t"
            "crowding_score\tnearest_neighbor_coordinate_pixels\n")
        for position in selected_positions:
            stream.write(
                f"{position}\t{source_rows[position]}\t{image_names[position]}\t"
                f"{scores[position]:.9g}\t{neighbor_distances[position]:.9g}\n")

    selection_path = output / "selection.npz"
    before_report_write = time.monotonic()
    timings = {
        "input_and_manifest_loading": input_loading_done - started,
        "metadata_and_neighbor_setup": (
            metadata_setup_done - input_loading_done),
        "scoring_and_selection": selection_done - metadata_setup_done,
        "diagnostic_and_output_writing": before_report_write - selection_done,
    }
    report = {
        "dataset": dataset,
        "input_stack": str(stack_path),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": manifest_hash,
        "selection_sha256": sha256_file(selection_path),
        "source_fingerprint": source_fingerprint,
        "source_star": str(star_path),
        "source_data_folder": None if data_folder is None else str(data_folder),
        "source_stack_prefix": stack_prefix,
        "n_input": count,
        "n_requested": n_select,
        "n_selected": int(len(selection.selected_positions)),
        "selected_fraction": len(selection.selected_positions) / count,
        "native_box_pixels": native_res,
        "native_pixel_size_angstrom": vox_native,
        "native_box_size_angstrom": native_res * vox_native,
        "smooth_sigma_pixels_at_denoise_resolution": SMOOTH_SIGMA,
        "minimum_neighbor_distance_fraction_of_native_box": neighbor_distance_fraction,
        "coordinate_pixel_size_angstrom": coordinate_pixel,
        "minimum_neighbor_distance_coordinate_pixels": neighbor_distance,
        "score_quantile_mode": "auto" if auto_score_quantile else "fixed",
        "maximum_crowding_score_quantile": (
            float(np.mean(scores <= score_ceiling)) if auto_score_quantile
            else max_score_quantile),
        "maximum_crowding_score": score_ceiling,
        "n_passing_geometry_and_score_gates": int(geometry_pass.sum()),
        "n_micrographs": int(len(np.unique(micrograph_ids))),
        "n_populated_micrograph_contrast_strata": int(len(unique_strata)),
        "contrast_bins": CONTRAST_BINS,
        "effective_contrast_bins": int(len(np.unique(contrast_ids))),
        "contrast_bin_boundaries": boundaries.tolist(),
        "inner_radius_fraction": INNER_RADIUS,
        "outer_radius_fraction": list(OUTER_RADIUS),
        "inner_top_fraction": INNER_TOP_FRACTION,
        "outer_top_fraction": OUTER_TOP_FRACTION,
        "selected_score_quantiles": {
            str(q): float(np.quantile(scores[selected_positions], q))
            for q in (0, 0.1, 0.5, 0.9, 0.99, 1)},
        "score_quantiles": {
            str(q): float(np.quantile(scores, q))
            for q in (0, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 1)},
        "preview_seed": seed,
        "n_preview": n_preview,
        "selection_diagnostic": {
            "seed_sequence": [seed, 0x53454C],
            "n_images": int(len(diagnostic_positions)),
            "n_selected": int(np.sum(diagnostic_selected)),
            "n_not_selected": int(np.sum(~diagnostic_selected)),
            "pdf": str(diagnostic_pdf),
            "pdf_sha256": sha256_file(diagnostic_pdf),
        },
        "score_plot": {
            "pdf": str(output / "selection_scores.pdf"),
        },
        "elapsed_seconds": before_report_write - started,
        "timings_seconds": timings,
        "provenance": {
            "kam_cryoem_commit": git_commit(ROOT),
            "python": sys.version,
            "numpy": np.__version__,
        },
    }
    report_path = output / "selection_report.json"
    # Write once so the measured output stage includes a real JSON
    # serialization and filesystem write, then patch in the final timings.
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report_checkpoint = time.monotonic()
    timings["diagnostic_and_output_writing"] = (
        report_checkpoint - selection_done)
    report["elapsed_seconds"] = report_checkpoint - started
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Selection timing: "
        f"input={timings['input_and_manifest_loading']:.1f}s, "
        f"metadata={timings['metadata_and_neighbor_setup']:.1f}s, "
        f"scoring={timings['scoring_and_selection']:.1f}s, "
        f"outputs={timings['diagnostic_and_output_writing']:.1f}s, "
        f"total={report['elapsed_seconds']:.1f}s",
        flush=True)
    print(json.dumps(report, indent=2), flush=True)


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
