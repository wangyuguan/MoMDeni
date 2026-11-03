"""Reusable CWF denoising of RELION particle stacks with FastPCA.

Includes the CUDA FLE transforms (formerly gpu_fle.py) behind a lazy
factory, so importing this module never requires fle_2d or CuPy.

The output is a CTF-corrected stack in unwhitened, amplitude-preserving units.
The radial FastPCA CTF uses mean defocus, optionally quantized into bins.  The
manifest retains the original per-particle CTF metadata and describes both the
astigmatism and defocus-quantization approximations.

By default no RELION pose, origin, reference map, or clean projection is used.
Optional global centering estimates one shift common to the dataset without STAR origins.
A separate diagnostic opt-in can use STAR origins instead. Both occur before
background or covariance estimation and never apply orientation angles.
Downstream moment estimation must treat the saved images as clean latent images:
do not apply CTF correction or noise debiasing a second time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .centering import METHOD as CENTERING_METHOD, METHOD_VERSION as CENTERING_METHOD_VERSION


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
# git submodules checked out next to src/ (git clone --recursive)
DEFAULT_FAST_PCA_ROOT = ROOT / "fast-cryoEM-PCA"
DEFAULT_FLE_ROOT = ROOT / "fle_2d"
PREPROCESSING_VERSION = "generic_relion_fast_cwf_mean_defocus_v2"
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
CENTERING_PREVIEW_MARKER_SEMANTICS = (
    "pre_translation_estimated_center_coordinate_in_both_rows"
)
CENTERING_PREVIEW_DESCRIPTION = (
    "Same raw Fourier-downsampled observations before and after the estimated "
    "translation; no phase flipping, intensity normalization, background "
    "subtraction, or denoising. The yellow plus marks the pre-translation "
    "estimated-center coordinate in both rows."
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
    denoise_max_order: int | None = None
    ctf_mean_defocus_bin: float = 50.0
    flip_sign: bool = True
    fast_pca_root: Path = DEFAULT_FAST_PCA_ROOT
    fle_root: Path = DEFAULT_FLE_ROOT
    driver_path: Path | None = None
    recenter_from_star: bool = False
    centering: bool = False
    centering_output_dir: Path | None = None

    def __post_init__(self):
        for name in ("star_path", "output_dir", "fast_pca_root", "fle_root"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.data_folder is not None:
            object.__setattr__(self, "data_folder", Path(self.data_folder))
        if self.driver_path is not None:
            object.__setattr__(self, "driver_path", Path(self.driver_path))
        if self.centering_output_dir is not None:
            object.__setattr__(self, "centering_output_dir", Path(self.centering_output_dir))
            if not self.centering:
                raise ValueError("centering_output_dir requires global centering")
        if not isinstance(self.dataset_label, str) or not self.dataset_label.strip():
            raise ValueError("dataset_label must be nonempty")
        if self.n_images < 0 or min(self.batch_size, self.workers) < 1:
            raise ValueError(
                "n_images must be nonnegative; batch_size and workers must be positive"
            )
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")
        if self.denoise_res < 4 or self.denoise_res % 2:
            raise ValueError("denoise_res must be even and at least 4")
        if (self.native_pixel_size is not None
                and (isinstance(self.native_pixel_size, (bool, np.bool_))
                     or not np.isfinite(self.native_pixel_size)
                     or self.native_pixel_size <= 0)):
            raise ValueError("native_pixel_size must be finite and positive")
        if not 0 < self.background_radius < np.sqrt(2):
            raise ValueError("background_radius must be between 0 and sqrt(2)")
        if not np.isfinite(self.eps) or not 0 < self.eps < 1:
            raise ValueError("eps must be between 0 and 1")
        if self.denoise_max_order is not None and (int(self.denoise_max_order) != self.denoise_max_order or self.denoise_max_order < 1):
            raise ValueError("denoise_max_order must be a positive integer or None")
        if (isinstance(self.ctf_mean_defocus_bin, (bool, np.bool_))
                or not np.isfinite(self.ctf_mean_defocus_bin)
                or self.ctf_mean_defocus_bin < 0):
            raise ValueError("ctf_mean_defocus_bin must be finite and nonnegative")
        if not isinstance(self.flip_sign, (bool, np.bool_)):
            raise ValueError("flip_sign must be Boolean")
        if not isinstance(self.recenter_from_star, (bool, np.bool_)):
            raise ValueError("recenter_from_star must be Boolean")
        if not isinstance(self.centering, (bool, np.bool_)):
            raise ValueError("centering must be Boolean")
        if self.centering and self.recenter_from_star:
            raise ValueError("global centering and RELION-origin recentering are mutually exclusive")


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
    if n_total < 1:
        raise ValueError("n_total must be positive")
    if n_images == 0:
        return np.arange(n_total, dtype=np.int64)
    if not 1 <= n_images <= n_total:
        raise ValueError(f"n_images must be zero or in [1, {n_total}]")
    rows = np.random.default_rng(seed).choice(n_total, size=n_images, replace=False)
    return np.sort(rows).astype(np.int64)


def read_relion_origin_shifts(particles):
    """Read explicit STAR origins as native-pixel ``(x, y)`` displacements.

    Angstrom origins take precedence and are divided by the configured native
    pixel size. Pixel-unit origins already refer to the native particle grid.
    Missing/incomplete coordinate pairs are errors; origin priors are not used.
    If both unit systems are present, require agreement within 1e-4 native
    pixels (allowing the decimal rounding used by STAR writers).

    RELION recenters with ASPIRE ``Image.shift(-origin_xy)``. ASPIRE's shift
    uses a positive Fourier phase, so this corresponds to a negative phase
    and a positive ``(x, y)`` displacement of image content on the array grid.
    """
    angstrom_columns = ("_rlnOriginXAngst", "_rlnOriginYAngst")
    pixel_columns = ("_rlnOriginX", "_rlnOriginY")
    present_pairs = []
    for columns in (angstrom_columns, pixel_columns):
        present = [column in particles.metadata for column in columns]
        if any(present) and not all(present):
            raise ValueError(f"incomplete RELION origin pair: {columns}")
        if all(present):
            present_pairs.append(columns)
    if not present_pairs:
        raise ValueError("RELION recentering requires explicit X/Y origin columns")
    pixel_size = particles.pixel_size
    if (isinstance(pixel_size, (bool, np.bool_))
            or not np.isfinite(pixel_size) or pixel_size <= 0):
        raise ValueError("native pixel size must be finite and positive")
    origins = {}
    for columns in present_pairs:
        values = np.column_stack([particles.column(name) for name in columns])
        if values.shape != (particles.n, 2) or not np.all(np.isfinite(values)):
            raise ValueError("RELION origins must be finite X/Y values for every particle")
        origins[columns] = (values / pixel_size
                            if columns == angstrom_columns else values)
    if len(origins) == 2 and not np.allclose(
            origins[angstrom_columns], origins[pixel_columns], rtol=0, atol=1e-4):
        raise ValueError("Angstrom and pixel RELION origin columns disagree")
    columns = present_pairs[0]
    source = {
        "columns": list(columns),
        "units": "angstrom" if columns == angstrom_columns else "native_pixels",
        "native_pixel_size_angstrom": float(pixel_size),
        "both_unit_systems_present": len(origins) == 2,
        "dual_unit_consistency_atol_native_pixels": 1e-4,
        "origin_priors_used": False,
    }
    return np.asarray(origins[columns], dtype=np.float64), source


def apply_relion_origin_phase(fourier, origins_xy_native_pixels, native_res):
    """Recenter cropped Fourier images, preserving the periodic field of view.

    Input coefficients use the centered grid from ``downsample_fourier_images``.
    Origins are ordered ``(x, y)`` in native pixels; columns are x and rows are y.
    Multiplication by ``exp(-2*pi*i*(fx*origin_x + fy*origin_y))`` uses origins
    scaled to the cropped grid and is equivalent to translation before Fourier
    cropping. No angle, reference, interpolation kernel, or padding is applied.
    """
    fourier = np.asarray(fourier)
    origins = np.asarray(origins_xy_native_pixels)
    if (fourier.ndim != 3 or fourier.shape[-1] != fourier.shape[-2]
            or not np.iscomplexobj(fourier) or not np.all(np.isfinite(fourier))):
        raise ValueError("expected finite centered complex square Fourier images")
    side = fourier.shape[-1]
    if (isinstance(native_res, (bool, np.bool_))
            or not isinstance(native_res, (int, np.integer))
            or side < 1 or native_res < side):
        raise ValueError("native resolution must be an integer at least the Fourier side")
    if (origins.shape != (len(fourier), 2) or np.iscomplexobj(origins)
            or not np.all(np.isfinite(origins))):
        raise ValueError("expected one finite native-pixel (x, y) origin per image")
    origins = origins.astype(np.float64, copy=False)
    scaled_origins = origins * (side / native_res)
    frequencies = np.fft.fftshift(np.fft.fftfreq(side))
    phase_argument = (
        scaled_origins[:, 0, None, None] * frequencies[None, None, :]
        + scaled_origins[:, 1, None, None] * frequencies[None, :, None]
    )
    shifted = fourier * np.exp(-2j * np.pi * phase_argument)
    if not np.all(np.isfinite(shifted)):
        raise ValueError("RELION origin phase produced nonfinite coefficients")
    return shifted


def group_radial_ctfs(parameters, mean_defocus_bin):
    """Group radial CTFs and return an internal contiguous ordering.

    Groups retain exact voltage, spherical aberration, and amplitude contrast.
    They ignore astigmatism orientation and replace ``(U,V)`` by mean defocus.
    A positive bin width rounds mean defocus to its nearest bin center; zero
    retains the exact per-particle mean.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != len(CTF_COLUMNS):
        raise ValueError("expected one six-column CTF row per selected particle")
    if not len(parameters) or not np.all(np.isfinite(parameters)):
        raise ValueError("CTF parameters must be nonempty and finite")
    if (np.any(parameters[:, 0] <= 0) or np.any(parameters[:, 4] < 0)
            or np.any(np.abs(parameters[:, 5]) > 1)):
        raise ValueError("invalid voltage, spherical aberration or amplitude contrast")
    if (not np.isfinite(mean_defocus_bin) or mean_defocus_bin < 0
            or isinstance(mean_defocus_bin, (bool, np.bool_))):
        raise ValueError("mean_defocus_bin must be finite and nonnegative")

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
    np.testing.assert_array_equal(ids[order], np.repeat(np.arange(len(unique)), counts))
    np.testing.assert_array_equal(order[inverse], np.arange(len(parameters)))
    return unique, ids.astype(np.int64), order, inverse, starts, counts, exact_mean, effective_mean


def ctf_phase_error_bounds(parameters, effective_mean, pixel_size):
    """Return Nyquist phase-error bounds for radialization and quantization."""
    parameters = np.asarray(parameters, dtype=np.float64)
    effective_mean = np.asarray(effective_mean, dtype=np.float64)
    if (parameters.ndim != 2 or parameters.shape[1] != len(CTF_COLUMNS)
            or effective_mean.shape != (len(parameters),)):
        raise ValueError("incompatible CTF parameter and effective-defocus arrays")
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be finite and positive")
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


def fastpca_radial_ctf_grid(side, pixel_size, radial_parameters):
    """Evaluate FastPCA's radial CTF on a centered Cartesian Fourier grid.

    Columns are voltage (kV), Cs (mm), amplitude contrast, mean defocus (A).
    The wavelength constants and sign agree with FLE ``expand_ctf`` exactly;
    in particular H(0)=-alpha. This is used only to phase flip estimator copies
    for image-based centering, not to CTF-correct the observations twice.
    """
    parameters = np.asarray(radial_parameters, dtype=np.float64)
    if (isinstance(side, (bool, np.bool_))
            or not isinstance(side, (int, np.integer)) or side < 1
            or not np.isfinite(pixel_size) or pixel_size <= 0
            or parameters.ndim != 2 or parameters.shape[1] != 4
            or not len(parameters) or not np.all(np.isfinite(parameters))):
        raise ValueError("expected a positive grid/pixel size and finite four-column radial CTF rows")
    voltage, cs, alpha, defocus = parameters.T
    if np.any(voltage <= 0) or np.any(cs < 0) or np.any(np.abs(alpha) > 1):
        raise ValueError("invalid radial CTF parameters")
    frequency = np.fft.fftshift(np.fft.fftfreq(side)) / pixel_size
    radius_squared = frequency[:, None]**2 + frequency[None, :]**2
    wavelength = 12.2643247 / np.sqrt(voltage * 1e3 + 0.978466 * voltage**2)
    gamma = (
        (-np.pi * wavelength * defocus)[:, None, None] * radius_squared
        + (0.5 * np.pi * cs * 1e7 * wavelength**3)[:, None, None]
        * radius_squared**2
    )
    return (np.sqrt(1. - alpha**2)[:, None, None] * np.sin(gamma)
            - alpha[:, None, None] * np.cos(gamma))


def subtract_background_mean(images, radius=0.9):
    """Subtract each image's background mean without normalizing its STD."""
    images = np.asarray(images)
    if (images.ndim != 3 or images.shape[1] != images.shape[2]
            or np.iscomplexobj(images) or not np.all(np.isfinite(images))):
        raise ValueError("expected finite real square images")
    if not 0 < radius < np.sqrt(2):
        raise ValueError("background radius must be between 0 and sqrt(2)")
    side = images.shape[-1]
    axis = (np.arange(side, dtype=np.float64) - side // 2) / (side / 2)
    mask = np.hypot(axis[:, None], axis[None, :]) > radius
    if not np.any(mask):
        raise ValueError("empty background mask")
    background = images[:, mask].astype(np.float64)
    means = np.mean(background, axis=1)
    stds = np.std(background, axis=1)
    return (images - means[:, None, None]).astype(np.float32), means, stds


def verify_fastpca_order(results, starts, counts):
    """Verify that FastPCA neither dropped nor duplicated a grouped run."""
    np.testing.assert_array_equal(results["image_indices_start"], starts)
    np.testing.assert_array_equal(results["image_indices_number"], counts)
    np.testing.assert_array_equal(
        results["image_indices_list"], np.r_[0, np.cumsum(counts)]
    )


def _with_max_order(basis_class, max_order):
    """FLE basis class that keeps only the eigenfunctions of angular order ``|nu| <= max_order`` (all of them if ``None``).

    Conjugate pairs +-nu stay together and FastPCA's mean, covariance and Wiener filter then run on the smaller basis."""
    if max_order is None:
        return basis_class

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
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not fast_pca.whiten or fast_pca.noise_psd is None:
        raise ValueError("batched denoising requires a fixed whitening PSD")
    if fast_pca.noise_psd.shape[0] != 1:
        raise ValueError("batched denoising currently requires one pooled PSD")
    if not fast_pca.correct_contrast:
        raise ValueError("batched denoising expects contrast correction enabled")
    if fast_pca.subtract_background:
        raise ValueError(
            "background subtraction must occur before batched denoising")

    count = int(source.n)
    side = int(source.L)
    filter_ids = np.asarray(source.filter_indices, dtype=np.int64)
    if filter_ids.shape != (count,):
        raise ValueError("FastPCA filter indices are incompatible with the stack")
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

    if not np.all(np.isfinite(images)) or not np.all(np.isfinite(contrast)):
        raise FloatingPointError("FastPCA returned nonfinite images or contrasts")
    fast_pca.contrast_est = contrast
    return images, contrast


def _code_identity(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _commit(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_centering_preview(before, after, path, *, estimated_centers_yx):
    """Plot paired observations with the pre-translation center marked in both."""
    import matplotlib.pyplot as plt

    before = np.asarray(before)
    after = np.asarray(after)
    centers = np.asarray(estimated_centers_yx, dtype=np.float64)
    if (before.ndim != 3 or before.shape != after.shape or len(before) < 1
            or not np.all(np.isfinite(before)) or not np.all(np.isfinite(after))):
        raise ValueError("centering preview requires finite, matching image stacks")
    if (centers.shape != (len(before), 2)
            or not np.all(np.isfinite(centers))):
        raise ValueError(
            "estimated_centers_yx must contain one finite (row, column) center per image"
        )
    height, width = before.shape[1:]
    if np.any((centers < 0) | (centers > np.array((height - 1, width - 1)))):
        raise ValueError("estimated centers must lie inside the image grid")
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("centering preview must be saved as a PDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2, len(before), figsize=(2.1 * len(before), 4.2), squeeze=False
    )
    limits = []
    for column, (original, centered) in enumerate(zip(before, after)):
        vmin, vmax = np.percentile(
            np.concatenate((original.ravel(), centered.ravel())), [1, 99]
        )
        if vmax <= vmin:
            vmax = vmin + max(1.0, abs(float(vmin))) * 1e-6
        limits.append([float(vmin), float(vmax)])
        for row, image in enumerate((original, centered)):
            axis = axes[row, column]
            axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax,
                        interpolation="nearest")
            # Keep the marker at the original estimated-center coordinate in
            # both rows so the displacement of that location is explicit.
            marker_y, marker_x = centers[column]
            # A dark halo keeps the marker visible on either bright or dark pixels.
            axis.plot(marker_x, marker_y, marker="+", linestyle="none",
                      color="black", markersize=10, markeredgewidth=3.0,
                      zorder=3)
            axis.plot(marker_x, marker_y, marker="+", linestyle="none",
                      color="#ffcc00", markersize=9, markeredgewidth=1.4,
                      zorder=4)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    axes[0, 0].set_ylabel("Before")
    axes[1, 0].set_ylabel("After")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return limits


def denoise_relion_particles(config: FastCWFDenoiseConfig) -> FastCWFDenoiseResult:
    """Denoise a reproducibly selected RELION particle subset.

    The function writes ``denoised_images.mrcs``, ``manifest.npz``,
    ``report.json``, and ``denoising_preview.pdf`` to a new output directory. Returned
    images are unwhitened latent-image estimates with their amplitude and
    FastPCA contrast estimate retained.
    """
    if not isinstance(config, FastCWFDenoiseConfig):
        raise TypeError("config must be a FastCWFDenoiseConfig")
    args = config
    fast_pca_root = args.fast_pca_root.resolve()
    fle_root = args.fle_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory; previous results are preserved")
    for path in (
        fast_pca_root / "fast_cryo_pca.py",
        fle_root / "src/fle_2d/fle_2d.py",
        args.star_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
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
    if args.denoise_res > particles.res:
        raise ValueError("denoise resolution cannot exceed native particle size")
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
    selected_origins = np.empty((0, 2), dtype=np.float64)
    origin_source = None
    image_centering = bool(args.centering)
    centering_method = (CENTERING_METHOD if image_centering else
                        "relion_origins" if args.recenter_from_star else "none")
    recentering = {
        "enabled": bool(args.recenter_from_star or image_centering),
        "method": centering_method,
        "origin_source": None,
        "orientation_angles_used": False,
        "relion_origins_used": bool(args.recenter_from_star),
        "reference_map_used": False,
    }
    if args.recenter_from_star:
        all_origins, origin_source = read_relion_origin_shifts(particles)
        selected_origins = all_origins[source_rows].copy()
        shift_norms = np.linalg.norm(selected_origins, axis=1)
        recentering.update({
            "origin_source": origin_source,
            "applied_phase": "exp(-2*pi*i*(fx*origin_x+fy*origin_y))",
            "applied_array_displacement": "positive_origin_xy; columns=x, rows=y",
            "equivalent_aspire_shift": "Image.shift(-origin_xy)",
            "boundary_condition": "periodic Fourier translation; no padding or cropping beyond downsampling",
            "stage": "after Fourier crop, before inverse FFT, background subtraction, PSD and CWF",
            "shift_norm_native_pixels": {
                "minimum": float(np.min(shift_norms)),
                "median": float(np.median(shift_norms)),
                "maximum": float(np.max(shift_norms)),
            },
            "maximum_absolute_component_native_pixels": float(
                np.max(np.abs(selected_origins))),
        })
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
        f"at {voxel_size:.12g} A/px; "
        + ("reference-free image centering; no pose or origin corrections" if image_centering
           else "STAR-origin recentering; no orientation corrections"
           if args.recenter_from_star else "no pose or origin corrections"),
        flush=True,
    )
    if args.recenter_from_star:
        print(
            "Recentering from " + ", ".join(origin_source["columns"])
            + "; periodic Fourier translation before background/PSD/CWF; "
            + f"native-pixel shift norm median/max={np.median(shift_norms):.4g}/"
            + f"{np.max(shift_norms):.4g}; no orientation angles or reference used",
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
    applied_shifts_denoise = selected_origins * (args.denoise_res / particles.res)
    centering_seconds = 0.0
    centering_setup_seconds = 0.0
    centering_diagnostic = None
    centering_inference = None
    if image_centering:
        from .centering import (CenteringConfig, apply_image_shifts,
                               estimate_dataset_shifts)

        centering_output = (
            args.centering_output_dir.resolve()
            if args.centering_output_dir is not None else output / "centering"
        )
        if centering_output.exists() and any(centering_output.iterdir()):
            raise FileExistsError(
                f"centering diagnostics already exist: {centering_output}"
            )
        centering_output.mkdir(parents=True, exist_ok=True)
        centering_preview_count = min(8, n_images)
        centering_preview_before = np.empty(
            (centering_preview_count, args.denoise_res, args.denoise_res),
            dtype=np.float32,
        )
        centering_preview_after = np.empty_like(centering_preview_before)
        centering_started = time.monotonic()
        centering_ctfs = fastpca_radial_ctf_grid(
            args.denoise_res, voxel_size, radial_parameters)
        applied_shifts_denoise = np.zeros((n_images, 2), dtype=np.float64)
        recentering.update({
            "ctf_for_estimator": "FastPCA radial binned mean-defocus; H(0)=-alpha",
            "ctf_wavelength_angstrom": "12.2643247/sqrt(voltage_kV*1e3+0.978466*voltage_kV**2)",
            "phase_flip_applied_to_observations": False,
            "observations_modified_only_by_translation": True,
            "uncertain_estimate_displacement": "zero: the common correction is applied only when it is at least min_shift_pixels and inside the unambiguous range",
            "center_coordinates": "box center minus the applied common XY correction, reversed to YX; not a calibrated physical center",
            "boundary_condition": "periodic Fourier translation; no padding or cropping beyond downsampling",
            "stage": "after all raw images are downsampled; before production background subtraction, PSD and CWF",
        })
        centering_setup_seconds = time.monotonic() - centering_started
        print(
            "Common-shift centering from the CTF-corrected mean image; CTF weighting changes estimator copies only; "
            "no STAR origins, angles, or reference map",
            flush=True,
        )
    load_psd_started = time.monotonic()

    def load_chunk(positions):
        raw = particles.read(source_rows[positions])
        fourier = downsample_fourier_images(raw, args.denoise_res)
        if args.recenter_from_star:
            fourier = apply_relion_origin_phase(
                fourier, selected_origins[positions], particles.res)
        lowres = centered_ifft2(fourier)
        scale = max(1.0, float(np.max(np.abs(lowres.real))))
        if np.max(np.abs(lowres.imag)) > 1e-10 * scale:
            raise FloatingPointError("Fourier downsampling failed the real-image check")
        return positions, lowres.real

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, n_images, args.batch_size):
            end = min(start + args.batch_size, n_images)
            chunks = [
                part for part in np.array_split(np.arange(start, end), args.workers)
                if len(part)
            ]
            for positions, images in pool.map(load_chunk, chunks):
                if image_centering:
                    # One raw stack is retained in place. Dataset-wide calibration
                    # must finish before translating observations or fitting CWF.
                    observations[positions] = images
                else:
                    images, means, stds = subtract_background_mean(images, args.background_radius)
                    observations[positions] = images
                    background_means[positions] = means
                    background_stds[positions] = stds
            if not image_centering:
                batch_psd = psd_estimator(observations[start:end])
                noise_psd_sum += np.sum(batch_psd, axis=0, dtype=np.float64)
            print(f"Loaded/downsampled {end:,}/{n_images:,}", flush=True)
    if image_centering:
        centering_started = time.monotonic()
        centering_preview_before[:] = observations[:centering_preview_count]
        centering_inference, centering_algorithm = estimate_dataset_shifts(
            observations, centering_ctfs, ctf_ids, config=CenteringConfig())
        applied_shifts_denoise = centering_inference["shifts_xy"]
        recentering["algorithm"] = centering_algorithm
        print(
            f"Centering common correction XY={centering_inference['common_shift_xy'].tolist()} "
            f"(estimate {centering_algorithm['estimate_xy_pixels']}, "
            f"not applied because: {centering_algorithm['not_applied_reason']})",
            flush=True)
        centering_seconds += time.monotonic() - centering_started
        for start in range(0, n_images, args.batch_size):
            end = min(start + args.batch_size, n_images)
            centering_started = time.monotonic()
            images = apply_image_shifts(observations[start:end], applied_shifts_denoise[start:end])
            if start < centering_preview_count:
                stop = min(end, centering_preview_count)
                centering_preview_after[start:stop] = images[:stop - start]
            centering_seconds += time.monotonic() - centering_started
            images, means, stds = subtract_background_mean(images, args.background_radius)
            observations[start:end] = images
            background_means[start:end] = means
            background_stds[start:end] = stds
            noise_psd_sum += np.sum(psd_estimator(observations[start:end]), axis=0, dtype=np.float64)
            print(f"Centered/background/PSD {end:,}/{n_images:,}", flush=True)
    noise_psd = noise_psd_sum / n_images
    load_psd_seconds = time.monotonic() - load_psd_started
    if image_centering:
        shift_norms = np.linalg.norm(applied_shifts_denoise, axis=1)
        recentering.update({
            "shift_norm_denoise_pixels": {
                "minimum": float(np.min(shift_norms)),
                "median": float(np.median(shift_norms)),
                "maximum": float(np.max(shift_norms)),
            },
            "elapsed_seconds": centering_seconds,
            "setup_seconds": centering_setup_seconds,
        })
        print(
            f"Global centering: shift norm median/max={np.median(shift_norms):.4g}/{np.max(shift_norms):.4g} "
            f"denoise pixels; estimation/application={centering_seconds:.2f} s",
            flush=True,
        )
        centering_preview_path = centering_output / "before_after.pdf"
        centering_preview_cache = centering_output / "preview.npz"
        centering_report_path = centering_output / "report.json"
        centering_estimates_path = centering_output / "estimates.npz"
        np.savez_compressed(
            centering_estimates_path,
            source_rows=source_rows, image_names=image_names,
            applied_shifts_xy_denoise_pixels=applied_shifts_denoise,
            **centering_inference,
        )
        preview_centers = (
            args.denoise_res // 2
            - applied_shifts_denoise[:centering_preview_count, ::-1])
        preview_limits = save_centering_preview(
            centering_preview_before, centering_preview_after,
            centering_preview_path,
            estimated_centers_yx=preview_centers,
        )
        np.savez_compressed(
            centering_preview_cache,
            before=centering_preview_before,
            after=centering_preview_after,
            source_rows=source_rows[:centering_preview_count],
            image_names=image_names[:centering_preview_count],
            applied_shifts_xy_denoise_pixels=(
                applied_shifts_denoise[:centering_preview_count]),
            estimated_centers_yx_denoise_pixels=preview_centers,
        )
        centering_diagnostic = {
            "directory": str(centering_output),
            "preview": str(centering_preview_path),
            "preview_cache": str(centering_preview_cache),
            "report": str(centering_report_path),
            "estimates": str(centering_estimates_path),
        }
        centering_report_path.write_text(json.dumps({
            "dataset": args.dataset_label,
            "n_total_particles": particles.n,
            "n_images_centered": n_images,
            "all_images_entering_covariance_centered": True,
            "n_preview": centering_preview_count,
            "preview_source_rows_zero_based": source_rows[:centering_preview_count].tolist(),
            "preview_description": CENTERING_PREVIEW_DESCRIPTION,
            "preview_marker_semantics": CENTERING_PREVIEW_MARKER_SEMANTICS,
            "preview_grayscale_limits_per_pair": preview_limits,
            "full_shift_manifest": str(centering_estimates_path),
            "recentering": recentering,
            "diagnostics": centering_diagnostic,
        }, indent=2) + "\n")
        print(f"Centering before/after preview: {centering_preview_path}", flush=True)
    if not np.all(np.isfinite(background_stds)) or np.any(background_stds <= 0):
        raise FloatingPointError("selected-image background noise has invalid/zero variance")
    if not np.all(np.isfinite(noise_psd)) or np.any(noise_psd <= 0):
        raise FloatingPointError("estimated background PSD is not finite and positive")

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
    if not np.all(np.isfinite(mean_est)):
        raise FloatingPointError("FastPCA mean is not finite")
    for block in range(covar_est.nblocks):
        if not np.all(np.isfinite(covar_est[block])):
            raise FloatingPointError("FastPCA covariance is not finite")
    denoise_started = time.monotonic()
    denoised_grouped, contrast_grouped = denoise_fastpca_batches(
        fast_pca, mean_est, covar_est, args.batch_size)
    denoise_seconds = time.monotonic() - denoise_started
    denoised = np.asarray(denoised_grouped, dtype=np.float32)[inverse]
    contrast_est = np.asarray(contrast_grouped, dtype=np.float64)[inverse]
    if denoised.shape != (n_images, args.denoise_res, args.denoise_res):
        raise ValueError(f"unexpected denoised stack shape: {denoised.shape}")
    if not np.all(np.isfinite(denoised)) or not np.all(np.isfinite(contrast_est)):
        raise FloatingPointError("FastPCA returned nonfinite images or contrasts")
    np.testing.assert_array_equal(source_rows[order][inverse], source_rows)

    write_started = time.monotonic()
    stack_path = output / "denoised_images.mrcs"
    with mrcfile.new_mmap(stack_path, shape=denoised.shape, mrc_mode=2) as stack:
        stack.set_image_stack()
        stack.voxel_size = voxel_size
        stack.data[:] = denoised
        stack.update_header_stats()
    stack_stat = stack_path.stat()
    configured_data_folder = "" if args.data_folder is None else str(args.data_folder.resolve())
    stack_prefix = args.stack_prefix or ""
    manifest_path = output / "manifest.npz"
    np.savez_compressed(
        manifest_path,
        preprocessing_version=PREPROCESSING_VERSION,
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
        ds_res=args.denoise_res,
        vox_native=particles.pixel_size,
        metadata_pixel_size=particles.metadata_pixel_size,
        native_pixel_size_overridden=args.native_pixel_size is not None,
        vox_denoise=voxel_size,
        vox_ds=voxel_size,
        denoised_stack=str(stack_path),
        stack_size=stack_stat.st_size,
        stack_mtime_ns=stack_stat.st_mtime_ns,
        axes="row=y, column=x; Fortran flatten",
        ctf_corrected=True,
        ctf_model="radial_quantized_mean_defocus_no_astigmatism",
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
        noise_denoised=True,
        whiten=True,
        whitened_output=False,
        single_whiten_filter=True,
        noise_psd=noise_psd,
        noise_psd_sampling_points=basis.pts,
        noise_psd_source="selected_image_backgrounds_pooled",
        flip_sign=args.flip_sign,
        background_mean_removed=True,
        subtract_background=True,
        fastpca_subtract_background=False,
        background_radius=args.background_radius,
        background_means=background_means,
        background_stds=background_stds,
        background_std_normalized=False,
        contrast_removed=False,
        correct_contrast=True,
        contrast_corrected_covariance=True,
        contrast_est=contrast_est,
        relion_poses_used=False,
        relion_orientation_angles_used=False,
        relion_origins_used=bool(args.recenter_from_star),
        recenter_from_star=bool(args.recenter_from_star),
        image_centering_enabled=image_centering,
        centering_method=centering_method,
        centering_method_version=CENTERING_METHOD_VERSION if image_centering else 0,
        relion_origin_source=json.dumps(origin_source, sort_keys=True),
        relion_origins_xy_native_pixels=selected_origins,
        applied_shifts_xy_native_pixels=(
            applied_shifts_denoise * (particles.res / args.denoise_res)
            if image_centering else selected_origins),
        applied_shifts_xy_denoise_pixels=applied_shifts_denoise,
        recentering=json.dumps(recentering, sort_keys=True),
        reference_used=False,
        reference_map_used=False,
        single_pass=True,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        fastpca_backend=fastpca_backend,
        fle_precision=basis.precision,
        eps=args.eps,
        python_version=sys.version,
        numpy_version=np.__version__,
        aspire_version=aspire.__version__,
        fast_pca_commit=_commit(fast_pca_root),
        fle_2d_commit=_commit(fle_root),
        kam_cryoem_commit=_commit(ROOT),
    )

    import matplotlib.pyplot as plt
    from .utils import show_particle_image

    fig, axes = plt.subplots(2, n_preview, figsize=(2.1 * n_preview, 4.2), squeeze=False)
    rows = (
        (0, preview_input, "Input"),
        (1, denoised[:n_preview], "Denoised"),
    )
    for row, images, label in rows:
        for column, image in enumerate(images):
            show_particle_image(axes[row, column], image)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(label)
    fig.tight_layout(pad=0.2)
    fig.savefig(output / "denoising_preview.pdf", bbox_inches="tight")
    plt.close(fig)
    write_seconds = time.monotonic() - write_started

    driver_path = args.driver_path or Path(__file__)
    report = {
        "dataset": args.dataset_label,
        "preprocessing_version": PREPROCESSING_VERSION,
        "device": args.device,
        "fastpca_backend": fastpca_backend,
        "fle_precision": basis.precision,
        "fle_max_order": None if args.denoise_max_order is None else int(args.denoise_max_order),
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
        "ctf_model": "radial quantized mean-defocus, no astigmatism",
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
        "noise": "one pooled radial PSD estimated from selected-image backgrounds",
        "flip_sign": args.flip_sign,
        "caveats": [
            "The radial CTF ignores measured astigmatism and its orientation.",
            "Mean defocus is rounded to the configured bin before CTF correction.",
            "Mean-only background subtraction does not normalize per-image noise or contrast.",
            "CWF outputs latent unwhitened amplitudes; do not CTF-correct or debias again.",
            "Residual denoising bias remains; CWF is not an unbiased-moment guarantee.",
        ],
        "relion_poses_used": False,
        "relion_orientation_angles_used": False,
        "relion_origins_used": bool(args.recenter_from_star),
        "recenter_from_star": bool(args.recenter_from_star),
        "image_centering_enabled": image_centering,
        "centering_method": centering_method,
        "centering_method_version": CENTERING_METHOD_VERSION if image_centering else 0,
        "centering_diagnostic": centering_diagnostic,
        "recentering": recentering,
        "reference_used": False,
        "reference_map_used": False,
        "whitened_output": False,
        "background_std_normalized": False,
        "contrast_removed": False,
        "output_stack": str(stack_path),
        "manifest": str(manifest_path),
        "elapsed_seconds": time.monotonic() - started,
        "timings_seconds": {
            "basis_setup": basis_seconds,
            "centering_setup": centering_setup_seconds,
            "image_centering_within_load_downsample_and_noise_psd": centering_seconds,
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
        "implementation": {
            "driver": _code_identity(driver_path),
            "denoising": _code_identity(Path(__file__)),
            "fast_pca": _code_identity(fast_pca_root / "fast_cryo_pca.py"),
            "fast_pca_utils": _code_identity(
                fast_pca_root / "utils_cwf_fast_batch.py"
            ),
            "fle": _code_identity(fle_root / "src/fle_2d/fle_2d.py"),
            "preprocessing": _code_identity(PACKAGE_DIR / "preprocessing.py"),
        },
    }
    if image_centering:
        report["implementation"]["centering"] = _code_identity(PACKAGE_DIR / "centering.py")
        report["caveats"].append(
            "The common shift is estimated from noisy observations; per-image shifts stay in the images and are modeled in the moments."
        )
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
# CUDA FLE transforms for FastPCA (was gpu_fle.py)
# ===========================================================================

def evaluate_fle_radial(basis, images):
    """Analyze only the zero-angular-order Fourier--Bessel coefficients.

    The output equals ``basis.evaluate_t(images)[..., basis.idx_list[0]]``
    up to the FLE approximation tolerance.  A polar NUFFT and angular average
    replace the full angular transform; only the order-zero radial interpolation
    is evaluated.  The input and output follow the NumPy FLE interface.
    """
    if hasattr(basis, "evaluate_t_radial"):
        return basis.evaluate_t_radial(images)
    from scipy.fft import dct, idct

    array = np.asarray(images)
    if (array.ndim not in (2, 3) or array.shape[-2:] != (basis.L1, basis.L1)
            or array.size == 0):
        raise ValueError(
            f"images must have shape ({basis.L1}, {basis.L1}) or "
            f"(n, {basis.L1}, {basis.L1}) with n > 0"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("images must be numeric")
    if np.iscomplexobj(array) and np.any(array.imag != 0):
        raise ValueError("FLE analysis expects real images")
    if not np.all(np.isfinite(array)):
        raise ValueError("images contains nonfinite values")
    single = array.size == basis.L1 ** 2
    values = np.array(array.real, dtype=basis.dt, copy=True).reshape(
        -1, basis.L1, basis.L1
    )
    if basis.L > basis.L1:
        values = np.pad(values, ((0, 0), (1, 0), (1, 0)))
    values[:, basis.idx] = 0
    polar = basis.step1(values).reshape(
        len(values), basis.n_radial, basis.n_angular
    )
    radial = np.mean(polar.real, axis=2)
    if basis.n_interp > basis.n_radial:
        radial = dct(radial, axis=1, type=2) / (2 * basis.n_radial)
        radial = np.concatenate((radial, np.zeros_like(radial)), axis=1)
        radial = idct(radial, axis=1, type=2) * 2 * radial.shape[1]
    coefficients = np.asarray(
        (basis.A3[0] @ radial.T).T, dtype=basis.dt
    )
    coefficients = coefficients * basis.cs[basis.idx_list[0]] / basis.h
    return np.squeeze(coefficients) if single else coefficients


def _cuda_modules():
    """Import and validate the CUDA dependencies lazily."""
    try:
        import cupy as cp
        import cupyx.scipy.fft as cufft
        import cupyx.scipy.sparse as cusparse
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "GPU FLE requires CuPy with cupyx.scipy.fft and sparse support"
        ) from error
    try:
        import cufinufft
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("GPU FLE requires the cufinufft package") from error
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
    try:
        from fle_2d import FLEBasis2D
    except ImportError as error:
        raise ImportError("GPUFLEBasis2D requires fle_2d on PYTHONPATH") from error

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
            if isinstance(device_id, (bool, np.bool_)) or int(device_id) != device_id:
                raise ValueError("device_id must be a nonnegative integer")
            if int(device_id) < 0:
                raise ValueError("device_id must be a nonnegative integer")
            if (isinstance(max_cached_plans, (bool, np.bool_))
                    or int(max_cached_plans) != max_cached_plans
                    or int(max_cached_plans) < 1):
                raise ValueError("max_cached_plans must be a positive integer")

            self.device_id = int(device_id)
            self.max_cached_plans = int(max_cached_plans)
            cp, cufft, cusparse, cufinufft = _cuda_modules()
            self._cp = cp
            self._cufft = cufft
            self._cusparse = cusparse
            self._cufinufft = cufinufft
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
            except Exception as error:  # pragma: no cover - requires CUDA failure
                raise RuntimeError("CUDA runtime initialization failed for GPU FLE") from error
            if self.device_id >= device_count:
                raise RuntimeError(
                    f"CUDA device {self.device_id} requested, but only "
                    f"{device_count} device(s) are visible"
                )

            # FLE precomputation is CPU-only and establishes the canonical basis.
            with cp.cuda.Device(self.device_id):
                super().__init__(*args, **kwargs)
                if self.precision not in ("single", "double"):
                    raise ValueError(
                        "GPU FLE supports precision='single' or precision='double'"
                    )
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

        def clear_plan_cache(self):
            """Release references to all cached cuFINUFFT plans."""
            with self._plan_lock:
                self._plan_cache.clear()

        def _plan(self, nufft_type, n_trans):
            if nufft_type not in (1, 2):
                raise ValueError("nufft_type must be 1 or 2")
            if isinstance(n_trans, (bool, np.bool_)) or int(n_trans) != n_trans:
                raise ValueError("n_trans must be a positive integer")
            n_trans = int(n_trans)
            if n_trans < 1:
                raise ValueError("n_trans must be a positive integer")
            key = (nufft_type, n_trans)
            plan = self._plan_cache.get(key)
            if plan is not None:
                self._plan_cache.move_to_end(key)
                return plan
            isign = 1 if nufft_type == 1 else -1
            try:
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
            except Exception as error:  # pragma: no cover - requires a GPU
                raise RuntimeError(
                    f"failed to create cuFINUFFT type-{nufft_type} plan for "
                    f"{n_trans} transform(s)"
                ) from error
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
                try:
                    output = plan.execute(cp.ascontiguousarray(data))
                    # cuFINUFFT execution may be asynchronous.  A cached plan must
                    # not be reused by another Python thread until its work has
                    # completed on the current stream.
                    cp.cuda.get_current_stream().synchronize()
                    return output
                except Exception as error:  # pragma: no cover - requires a GPU
                    raise RuntimeError(
                        f"cuFINUFFT type-{nufft_type} execution failed for "
                        f"batch size {n_trans}"
                    ) from error

        @staticmethod
        def _finite_host_array(value, name):
            try:
                array = np.asarray(value)
            except Exception as error:
                raise TypeError(f"{name} must be convertible to a NumPy array") from error
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"{name} must be numeric")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains nonfinite values")
            return array

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

        def _forward_radial_gpu(self, images):
            """Order-zero FLE analysis of padded, disk-masked device images."""
            cp = self._cp
            polar_half = self._execute_plan(
                2, cp.asarray(images, dtype=self._complex_dtype)
            ) * self.h ** 2
            polar_half = polar_half.reshape(
                len(images), self.n_radial, self.n_angular // 2
            )
            # The omitted angular half is the complex conjugate for real input.
            radial = cp.mean(polar_half.real, axis=2)
            if self.n_interp > self.n_radial:
                radial = self._cufft.dct(radial, axis=1, type=2)
                radial /= 2 * self.n_radial
                radial = cp.concatenate((radial, cp.zeros_like(radial)), axis=1)
                radial = (
                    self._cufft.idct(radial, axis=1, type=2)
                    * 2 * radial.shape[1]
                )
            coefficients = cp.asarray(
                (self._a3_gpu[0] @ radial.T).T, dtype=self._real_dtype
            )
            return coefficients * self._cs_gpu[self._idx_list_gpu[0]] / self.h

        def evaluate_t_radial_gpu(self, images):
            """Analyze only angular order zero and return coefficients on CUDA.

            Accepts NumPy or CuPy images of shape ``(L1, L1)`` or
            ``(n, L1, L1)``.  Padding, disk masking, coefficient normalization,
            and one-image squeezing match :meth:`evaluate_t`; inputs are never
            modified.  Even small images use the polar FLE algorithm here.
            """
            cp = self._cp
            with cp.cuda.Device(self.device_id):
                values = cp.asarray(images)
                if (values.ndim not in (2, 3)
                        or values.shape[-2:] != (self.L1, self.L1)
                        or values.size == 0):
                    raise ValueError(
                        f"images must have shape ({self.L1}, {self.L1}) or "
                        f"(n, {self.L1}, {self.L1}) with n > 0"
                    )
                if not cp.issubdtype(values.dtype, cp.number):
                    raise TypeError("images must be numeric")
                if cp.iscomplexobj(values) and bool(cp.any(values.imag != 0).item()):
                    raise ValueError("FLE analysis expects real images")
                if not bool(cp.all(cp.isfinite(values)).item()):
                    raise ValueError("images contains nonfinite values")
                single = values.size == self.L1 ** 2
                values = cp.array(
                    values.real, dtype=self._real_dtype, copy=True
                ).reshape(-1, self.L1, self.L1)
                if self.L > self.L1:
                    values = cp.pad(values, ((0, 0), (1, 0), (1, 0)))
                values[:, self._disk_outside_gpu] = 0
                coefficients = self._forward_radial_gpu(values)
                return cp.squeeze(coefficients) if single else coefficients

        def evaluate_t_radial(self, images):
            """Return NumPy order-zero coefficients, matching the FLE API."""
            with self._cp.cuda.Device(self.device_id):
                return self._cp.asnumpy(self.evaluate_t_radial_gpu(images))

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
            array = self._finite_host_array(f, "images")
            if np.iscomplexobj(array) and np.any(array.imag != 0):
                raise ValueError("FLE analysis expects real images")
            single = array.size == self.L1**2
            if array.size % self.L1**2 or array.ndim not in (2, 3):
                raise ValueError(
                    f"images must have shape ({self.L1}, {self.L1}) or "
                    f"(n, {self.L1}, {self.L1})"
                )
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
            array = self._finite_host_array(a, "coefficients")
            single = array.size == self.ne
            if array.size % self.ne or array.ndim not in (1, 2):
                raise ValueError(
                    f"coefficients must have shape ({self.ne},) or (n, {self.ne})"
                )
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
            if images.ndim != 3 or images.shape[1:] != (self.L, self.L):
                raise ValueError(
                    f"GPU NUFFT input must have shape (n, {self.L}, {self.L})"
                )
            if images.shape[0] < 1:
                raise ValueError("GPU NUFFT input batch is empty")
            with cp.cuda.Device(self.device_id):
                values = cp.asarray(images, dtype=self._complex_dtype)
                if not bool(cp.all(cp.isfinite(values)).item()):
                    raise ValueError("GPU NUFFT input contains nonfinite values")
                return self._execute_plan(2, values)


    def estimate_radial_psd_gpu(imgs, basis, bgRadius=0.9, dtype=None):
        """Estimate radial background PSD with the basis' cached cuFINUFFT plan.

        The formula and output layout match
        ``fast-cryoEM-PCA.utils_cwf_fast_batch.estimate_radial_psd``.  ``basis``
        must be a :class:`GPUFLEBasis2D`; this requirement prevents an accidental
        CPU fallback in a requested GPU run.
        """
        if not isinstance(basis, GPUFLEBasis2D):
            raise TypeError("basis must be a GPUFLEBasis2D")
        if (not np.isfinite(bgRadius) or not 0 < bgRadius < np.sqrt(2)):
            raise ValueError("bgRadius must be finite and between 0 and sqrt(2)")
        images = GPUFLEBasis2D._finite_host_array(imgs, "images")
        if np.iscomplexobj(images) and np.any(images.imag != 0):
            raise ValueError("radial PSD estimation expects real images")
        if (images.ndim != 3 or images.shape[0] < 1
                or images.shape[1:] != (basis.L, basis.L)):
            raise ValueError(
                f"images must have shape (n, {basis.L}, {basis.L})"
            )
        requested_dtype = np.dtype(np.float64 if dtype is None else dtype)
        if requested_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("dtype must be float32 or float64")
        if requested_dtype != basis.gpu_real_dtype:
            raise ValueError(
                f"PSD dtype {requested_dtype.name} does not match GPU basis "
                f"precision {basis.gpu_real_dtype.name}"
            )

        side = basis.L
        axis = np.arange(-side // 2, side // 2, dtype=requested_dtype) / (side // 2)
        radius = np.hypot(axis[:, None], axis[None, :])
        mask = radius >= bgRadius
        denominator = int(np.count_nonzero(mask))
        if denominator == 0:
            raise ValueError("background mask is empty")

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
        if result.shape != (len(images), basis.n_radial):
            raise RuntimeError("unexpected radial PSD output shape")
        if not np.all(np.isfinite(result)) or np.any(result < 0):
            raise FloatingPointError("GPU radial PSD is nonfinite or negative")
        return result



    _GPU_FLE["basis"] = GPUFLEBasis2D
    _GPU_FLE["psd"] = estimate_radial_psd_gpu
    return GPUFLEBasis2D, estimate_radial_psd_gpu
