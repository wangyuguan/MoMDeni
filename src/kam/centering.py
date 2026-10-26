"""Image-only common-offset and noise-calibrated particle centering.

The dataset estimator first removes a guarded common Fourier-phase offset,
then calibrates shrinkage of individual radial corrections using paired noise
replicas. No reference volume, viewing angles or particle-origin metadata is
used. Rejected corrections remain exact no-ops on the original observations.

Implements the radial cumulative-mass criterion of Heimowitz, Sharon and
Singer, *Centering noisy images with application to cryo-EM*, arXiv:2009.04810.
Local angular averaging uses the zero-angular-order Fourier--Bessel modes
and FLE2D's fast polar NUFFT and radial interpolation.

Unlike the public MATLAB implementation, radial mass is cumulatively summed
before scoring, and annuli contain only pixels inside the support disk.
Estimator-only phase flipping, sign correction and intensity normalization are
never applied to the returned observation: estimate a displacement, then use
``apply_image_shifts`` on the original images. Pixel size and CTF metadata do
not change. Translation uses periodic boundaries; it cannot recover signal
that was already cut off by particle extraction.

Centers are zero-based (row, column). ``shifts_xy`` is the displacement to
APPLY, in input pixels: positive x moves right and positive y moves down.
The target center is (height // 2, width // 2), including even-sized images.
Estimated residual shifts need not be independent, Gaussian or exactly zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import j0


METHOD = "common_phase_guarded_replica_fb"
METHOD_VERSION = 3


__all__ = [
    "METHOD", "METHOD_VERSION", "CenteringConfig", "estimate_dataset_shifts",
    "estimate_background_noise_psd",
    "CenteringResult", "NoisyImageCenterer", "estimate_image_centers",
    "apply_image_shifts", "phase_flip_images", "wiener_filter_images",
    "radial_cumulative_scores", "centering_distance_mask",
    "FourierMean", "weighted_fourier_mean",
    "estimate_mean_phase", "estimate_global_center", "calibrate_replica_shrinkage",
    "generate_colored_noise",
]


@dataclass(frozen=True)
class CenteringResult:
    """Per-image results, always with a leading image axis, on the CPU.

    ``valid`` excludes flat images and flat/undefined score landscapes; it is
    not a confidence probability. ``boundary_hit`` marks a minimum at the
    edge of the allowed search. Invalid images receive zero displacement.
    Optional score maps contain NaN outside the search and for invalid images.
    """

    centers_yx: np.ndarray
    shifts_xy: np.ndarray
    scores: np.ndarray
    valid: np.ndarray
    boundary_hit: np.ndarray
    score_maps: np.ndarray | None
    parameters: dict


def centering_distance_mask(centers_yx, valid, image_shape, max_distance_fraction):
    """Keep valid centers within a Euclidean distance from the box center.

    The threshold is a fraction of the shorter image side. Coordinates are
    measured before translation, with the same integer origin as the centerer.
    """
    centers = np.asarray(centers_yx, dtype=np.float64)
    valid = np.asarray(valid)
    shape = np.asarray(image_shape)
    if (centers.ndim != 2 or centers.shape[1] != 2
            or valid.shape != (len(centers),) or valid.dtype.kind != "b"):
        raise ValueError("expected (n,2) centers and one Boolean valid flag per image")
    if (shape.shape != (2,) or shape.dtype.kind not in "iu"
            or np.any(shape < 1)):
        raise ValueError("image_shape must contain two positive integers")
    if (isinstance(max_distance_fraction, (bool, np.bool_))
            or not np.isscalar(max_distance_fraction)
            or not np.isfinite(max_distance_fraction)
            or not 0 < max_distance_fraction <= 0.5):
        raise ValueError("max_distance_fraction must be finite and in (0, 0.5]")
    distances = np.linalg.norm(centers - shape // 2, axis=1)
    inside = np.all((centers >= 0) & (centers < shape), axis=1)
    keep = (valid & inside & np.isfinite(distances)
            & (distances <= float(max_distance_fraction) * shape.min()))
    return keep, distances


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _image_stack(images):
    data = np.asarray(images)
    if (data.ndim not in (2, 3) or min(data.shape) < 1
            or np.iscomplexobj(data) or data.dtype.kind not in "fiu"
            or not np.all(np.isfinite(data))):
        raise ValueError("images must be finite real (height,width) or (n,height,width) arrays")
    return np.asarray(data if data.ndim == 3 else data[None], dtype=np.float64)


def _fft2(images):
    return np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(images, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))


def _ifft2(fourier):
    return np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(fourier, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))


def _real_ifft(fourier):
    if not np.all(np.isfinite(fourier)):
        raise ValueError("Fourier operation produced nonfinite values; check image scaling")
    result = _ifft2(fourier)
    if not np.all(np.isfinite(result)):
        raise ValueError("Inverse Fourier transform produced nonfinite values; check image scaling")
    scale = np.maximum(np.max(np.abs(result.real), axis=(-2, -1)), np.finfo(np.float64).tiny)
    if np.any(np.max(np.abs(result.imag), axis=(-2, -1)) > 1e-9 * scale):
        raise ValueError("Fourier operation breaks real-image symmetry; check CTF/Nyquist samples")
    return result.real


def wiener_filter_images(images, ctf, balance=0.05):
    """Wiener-deconvolve copies for center estimation: multiply by H/(H^2+b).

    Phase flipping only corrects the transfer-function sign; the amplitude
    |H| still suppresses the low frequencies that carry an image's mass, so
    a mass-concentration criterion sees an almost mean-free ring instead of
    a blob. The Wiener weight restores those amplitudes in a controlled way
    and flips the phase at the same time. It is a pointwise Fourier filter,
    hence translation-equivariant. It can still change the shape-dependent
    center selected by a mass-concentration criterion.

    ``balance`` is the noise-to-signal power ratio, a positive scalar or a
    centered-grid array broadcastable against ``ctf``. The filtered copy is
    for estimation only; observations keep their CTF.
    """
    stack = _image_stack(images)
    transfer = np.asarray(ctf)
    if (transfer.shape not in (stack.shape, stack.shape[1:])
            or np.iscomplexobj(transfer) or transfer.dtype.kind not in "fiu"
            or not np.all(np.isfinite(transfer))):
        raise ValueError("ctf must be a finite real centered (H,W) or (N,H,W) array")
    weight = np.asarray(balance, dtype=np.float64)
    if (np.iscomplexobj(weight) or not np.all(np.isfinite(weight))
            or not np.all(weight > 0)):
        raise ValueError("balance must be positive and finite")
    result = _real_ifft(_fft2(stack) * (transfer / (transfer ** 2 + weight)))
    return result[0] if np.ndim(images) == 2 else result


def phase_flip_images(images, ctf):
    """Phase flip a copy for center estimation, not full CTF inversion.

    ``ctf`` is a finite real centered Fourier grid, shared (H,W) or per-image
    (N,H,W). A zero CTF keeps phase +1; no division or noise amplification is
    performed. After this operation signal still contains the CTF amplitude.
    The filter must preserve discrete Hermitian symmetry on nonzero samples.
    """
    stack = _image_stack(images)
    transfer = np.asarray(ctf)
    if (transfer.shape not in (stack.shape, stack.shape[1:])
            or np.iscomplexobj(transfer) or transfer.dtype.kind not in "fiu"
            or not np.all(np.isfinite(transfer))):
        raise ValueError("ctf must be a finite real centered (H,W) or (N,H,W) array")
    result = _real_ifft(_fft2(stack) * np.where(transfer < 0, -1., 1.))
    return result[0] if np.ndim(images) == 2 else result


def apply_image_shifts(images, shifts_xy):
    """Apply periodic XY corrections once, without intensity normalization.

    Zero corrections preserve pixels exactly, and integer corrections use
    exact array rolls. Fractional corrections discard even-grid Nyquist lines
    to retain real-image symmetry. Floating-point input precision is preserved;
    integer observations are promoted only when fractional shifts are needed.
    Supports one image or a stack, including rectangular images.
    """
    data = np.asarray(images)
    if (data.ndim not in (2, 3) or min(data.shape) < 1
            or np.iscomplexobj(data) or data.dtype.kind not in "fiu"):
        raise ValueError("images must be finite real (height,width) or (n,height,width) arrays")
    stack = data if data.ndim == 3 else data[None]
    _check_finite_batches(stack)
    shifts = np.asarray(shifts_xy)
    if (shifts.shape not in ((2,), (len(stack), 2))
            or np.iscomplexobj(shifts) or shifts.dtype.kind not in "fiu"
            or not np.all(np.isfinite(shifts))):
        raise ValueError("shifts_xy must be finite applied (x,y) or (N,2) displacements")
    shifts = np.broadcast_to(shifts, (len(stack), 2)).astype(np.float64)
    integer = np.all(shifts == np.rint(shifts), axis=1)
    result = stack.astype(np.float64) if data.dtype.kind in "iu" and not np.all(integer) else stack.copy()
    moved = np.flatnonzero(np.any(shifts != 0, axis=1))
    for index in moved[integer[moved]]:
        dx, dy = shifts[index].astype(int)
        result[index] = np.roll(stack[index], (dy, dx), axis=(-2, -1))
    fractional = moved[~integer[moved]]
    height, width = stack.shape[-2:]
    fy, fx = np.fft.fftfreq(height), np.fft.fftfreq(width)
    for first in range(0, len(fractional), 128):
        indices = fractional[first:first+128]
        work = np.asarray(stack[indices], dtype=np.float64)
        fourier = np.fft.fft2(np.fft.ifftshift(work, axes=(-2, -1)))
        if height % 2 == 0:
            fourier[:, height//2, :] = 0
        if width % 2 == 0:
            fourier[:, :, width//2] = 0
        phase = (shifts[indices, 0, None, None]*fx[None, None, :]
                 + shifts[indices, 1, None, None]*fy[None, :, None])
        translated = np.fft.fftshift(np.fft.ifft2(fourier*np.exp(-2j*np.pi*phase)), axes=(-2, -1))
        if np.max(abs(translated.imag)) > 1e-10*max(1., float(np.max(abs(translated.real)))):
            raise FloatingPointError("Translation broke real-image symmetry")
        result[indices] = translated.real
    return result[0] if data.ndim == 2 else result


def _check_finite_batches(images, batch_size=128):
    for first in range(0, len(images), batch_size):
        if not np.isfinite(images[first:first+batch_size]).all():
            raise ValueError("images must contain only finite values")


def _scores_from_cumulative(cumulative, xp):
    # One normalization per image, over all candidate centers, never per batch.
    maximum_mass = xp.max(cumulative[..., -1], axis=-1, keepdims=True)
    denominator = xp.where(maximum_mass > 0, maximum_mass, 1.)
    scores = xp.sum(xp.abs(1. - cumulative / denominator[..., None]), axis=-1)
    return xp.where(maximum_mass > 0, scores, xp.nan)


def radial_cumulative_scores(ring_masses):
    """Score (..., candidate, radius) annular mass arrays by paper Eq. (26).

    The last axis begins at radius zero. Annuli are m-1 < distance <= m.
    Emax is the largest total disk mass over candidates of the SAME image.
    Absolute values are retained because a truncated Fourier--Bessel expansion can
    have negative lobes. Nonpositive Emax produces NaN scores.
    """
    mass = np.asarray(ring_masses)
    if (mass.ndim < 2 or min(mass.shape) < 1 or np.iscomplexobj(mass)
            or mass.dtype.kind not in "fiu" or not np.all(np.isfinite(mass))):
        raise ValueError("ring_masses must be finite real (...,candidate,radius) arrays")
    return _scores_from_cumulative(np.cumsum(mass, axis=-1, dtype=np.float64), np)


def _fb_cumulative_plan(basis, radius):
    """Integrate the same discrete radial basis used by FLE analysis.

    FLE uses an even patch [-R,R-1]^2; the omitted +R disk boundary samples
    vanish at Bessel roots. Coefficients are discrete inner products, so the
    synthesis samples include FLE's h factor, not only continuous normalization.
    """
    grid_y, grid_x = np.meshgrid(np.arange(-radius, radius),
                                np.arange(-radius, radius), indexing="ij")
    squared_distance = grid_y**2 + grid_x**2
    disk = squared_distance <= radius**2
    offsets = np.column_stack((grid_y.ravel(), grid_x.ravel()))
    distance = np.sqrt(squared_distance[disk])
    indices = np.asarray(basis.idx_list[0])
    samples = basis.h * basis.cs[indices][None, :] * j0(
        distance[:, None] / radius * basis.lmds[indices][None, :])
    ring = np.ceil(distance).astype(np.int64)
    ring_samples = np.zeros((radius + 1, len(indices)), dtype=np.float64)
    np.add.at(ring_samples, ring, samples)
    cumulative_samples = np.cumsum(ring_samples, axis=0).T.copy()
    ring_counts = np.bincount(ring, minlength=radius + 1)
    for array in (offsets, cumulative_samples, ring_counts):
        array.setflags(write=False)
    return offsets, cumulative_samples, ring_counts


class NoisyImageCenterer:
    """Reusable local Fourier--Bessel plan for a particle support radius.

    ``radius`` is an integer upper bound on particle radius in input pixels,
    not an expansion degree or shift standard deviation. A candidate disk
    must fit entirely inside the image: large radii restrict the search.
    Only radius and optional max_shift affect the search geometry. Defaults
    for FLE accuracy and chunk sizes bound CPU/GPU working memory. Each
    candidate uses its own local origin; a global angular average is not a
    substitute for translating the origin before radial analysis.

    GPU execution uses CuPy/cuFINUFFT for radial analysis and scoring. Basis
    setup, optional phase flipping, and displacement application use CPU
    NumPy/SciPy. No RELION poses or reference map is required.
    """

    def __init__(self, radius, *, device="cpu", fle_eps=1e-6, fle_root=None,
                 candidate_batch_size=128, image_batch_size=16):
        self.radius = _positive_integer(radius, "radius")
        self.candidate_batch_size = _positive_integer(candidate_batch_size, "candidate_batch_size")
        self.image_batch_size = _positive_integer(image_batch_size, "image_batch_size")
        if (isinstance(fle_eps, (bool, np.bool_))
                or not np.isfinite(fle_eps) or not 0 < fle_eps < 1):
            raise ValueError("fle_eps must be in (0,1)")
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        self.fle_eps = float(fle_eps)
        self.device = device
        self.xp = np
        if device == "gpu":
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("no CUDA devices")
            except Exception as exc:
                raise RuntimeError("GPU centering requires a working CuPy/CUDA installation") from exc
            self.xp = cp
        from .denoise import DEFAULT_FLE_ROOT, evaluate_fle_radial, _gpu_fle_components
        root = Path(DEFAULT_FLE_ROOT if fle_root is None else fle_root).expanduser()
        if (root / "src/fle_2d").is_dir() and str(root / "src") not in sys.path:
            sys.path.insert(0, str(root / "src"))
        from fle_2d import FLEBasis2D

        basis_class = FLEBasis2D if device == "cpu" else _gpu_fle_components()[0]
        self.basis = basis_class(2 * self.radius, bandlimit=2 * self.radius,
                                 eps=self.fle_eps, mode="real", precision="double")
        self.radial_indices = np.asarray(self.basis.idx_list[0])
        self._analyze = (lambda patches: evaluate_fle_radial(self.basis, patches)) if device == "cpu" else self.basis.evaluate_t_radial_gpu
        self.offsets_yx, cumulative, self.ring_counts = _fb_cumulative_plan(self.basis, self.radius)
        self._cumulative_samples = self.xp.asarray(cumulative)
        self._offsets = self.xp.asarray(self.offsets_yx)

    def estimate(self, images, *, ctf=None, ctf_mode="flip", wiener_balance=0.05,
                 contrast_sign=1., max_shift=None, return_score_maps=False):
        """Estimate centers from image copies, leaving observations unchanged.

        Optional CTF handling and contrast_sign (either +1 or -1) affect
        only the estimator. With a ``ctf``, ``ctf_mode="flip"`` phase flips
        the estimator copies and ``"wiener"`` applies the translation-
        equivariant Wiener weight ``H/(H^2+wiener_balance)``, which also
        restores the CTF-suppressed low-frequency mass that the radial
        criterion scores. ``max_shift`` bounds EACH axis from the image
        origin in pixels; None searches all complete disks. Pixel-grid
        precision is intentional. Score ties prefer the image origin.
        """
        stack = _image_stack(images)
        n_images, height, width = stack.shape
        if 2 * self.radius + 1 > min(height, width):
            raise ValueError("the full particle support disk must fit inside the image")
        if (isinstance(contrast_sign, (bool, np.bool_))
                or contrast_sign not in (-1., 1.)):
            raise ValueError("contrast_sign must be +1 or -1")
        if max_shift is not None and (
                isinstance(max_shift, (bool, np.bool_))
                or not np.isfinite(max_shift) or max_shift < 0):
            raise ValueError("max_shift must be nonnegative and finite, or None")
        if max_shift is not None:
            max_shift = float(max_shift)
        if not isinstance(return_score_maps, (bool, np.bool_)):
            raise ValueError("return_score_maps must be Boolean")
        if ctf_mode not in ("flip", "wiener"):
            raise ValueError("ctf_mode must be 'flip' or 'wiener'")
        transfer = None if ctf is None else np.asarray(ctf)
        if transfer is not None and transfer.shape not in (stack.shape, stack.shape[1:]):
            raise ValueError("ctf must have image shape (H,W) or stack shape (N,H,W)")
        origin = np.array((height // 2, width // 2), dtype=np.int64)
        yy, xx = np.meshgrid(np.arange(self.radius, height - self.radius),
                             np.arange(self.radius, width - self.radius), indexing="ij")
        candidates = np.column_stack((yy.ravel(), xx.ravel()))
        if max_shift is not None:
            candidates = candidates[np.all(np.abs(candidates - origin) <= max_shift, axis=1)]
        if not len(candidates):
            raise ValueError("no candidate center satisfies radius and max_shift")
        # Deterministic tie preference, independent of chunking and row-major order.
        distance_order = np.argsort(np.sum((candidates - origin)**2, axis=1), kind="stable")
        candidates = candidates[distance_order]
        centers = np.repeat(origin[None], n_images, axis=0)
        scores = np.full(n_images, np.nan)
        valid = np.zeros(n_images, dtype=bool)
        boundary = np.zeros(n_images, dtype=bool)
        score_maps = np.full(stack.shape, np.nan) if return_score_maps else None
        lower, upper = candidates.min(axis=0), candidates.max(axis=0)
        xp = self.xp
        gpu_candidates = xp.asarray(candidates)
        for start in range(0, n_images, self.image_batch_size):
            end = min(start + self.image_batch_size, n_images)
            observed = stack[start:end]
            # FFT roundoff must not turn a constant image into unit-contrast noise.
            raw_nonflat = np.ptp(observed, axis=(1, 2)) > 0
            if transfer is not None:
                batch_transfer = transfer if transfer.ndim == 2 else transfer[start:end]
                observed = (phase_flip_images(observed, batch_transfer)
                            if ctf_mode == "flip" else
                            wiener_filter_images(observed, batch_transfer,
                                                 wiener_balance))
            work = contrast_sign * observed
            work = work - np.min(work, axis=(1, 2), keepdims=True)
            ranges = np.max(work, axis=(1, 2))
            nonflat = raw_nonflat & (ranges > 0)
            work /= np.where(nonflat, ranges, 1.)[:, None, None]
            device_images = xp.asarray(work)
            cumulative = xp.empty((end - start, len(candidates), self.radius + 1), dtype=xp.float64)
            for first in range(0, len(candidates), self.candidate_batch_size):
                last = min(first + self.candidate_batch_size, len(candidates))
                positions = gpu_candidates[first:last]
                patches = device_images[
                    :, positions[:, 0, None] + self._offsets[None, :, 0],
                    positions[:, 1, None] + self._offsets[None, :, 1]]
                # Only local m=0 modes are analyzed, not a full angular expansion.
                coefficients = self._analyze(patches.reshape(
                    -1, 2 * self.radius, 2 * self.radius)).reshape(
                    end - start, last - first, len(self.radial_indices))
                cumulative[:, first:last] = coefficients @ self._cumulative_samples
            landscapes = _scores_from_cumulative(cumulative, xp)
            if self.device == "gpu":
                landscapes = xp.asnumpy(landscapes)
            for local in range(end - start):
                index, landscape = start + local, landscapes[local]
                if not nonflat[local] or not np.all(np.isfinite(landscape)):
                    continue
                tie_tolerance = 1e-12 * max(1., float(np.max(np.abs(landscape))))
                if np.ptp(landscape) <= tie_tolerance:
                    continue
                minimum = float(np.min(landscape))
                best = int(np.flatnonzero(landscape <= minimum + tie_tolerance)[0])
                centers[index] = candidates[best]
                scores[index] = landscape[best]
                valid[index] = True
                boundary[index] = bool(np.any(centers[index] == lower) or np.any(centers[index] == upper))
                if score_maps is not None:
                    score_maps[index, candidates[:, 0], candidates[:, 1]] = landscape
        applied_yx = origin[None] - centers
        return CenteringResult(
            centers_yx=centers, shifts_xy=applied_yx[:, ::-1].copy(), scores=scores,
            valid=valid, boundary_hit=boundary, score_maps=score_maps,
            parameters={
                "method": "fb_cumulative_radial_mass", "version": 2,
                "radius_pixels": self.radius, "fle_eps": self.fle_eps,
                "basis": "fourier_bessel", "radial_modes": len(self.radial_indices),
                "patch_side": 2 * self.radius,
                "transform": "fle_radial_polar_nufft",
                "radial_eigenvalues": self.basis.lmds[self.radial_indices].tolist(),
                "device": self.device, "n_candidates": len(candidates),
                "max_shift_pixels_per_axis": max_shift, "image_shape": [height, width],
                "target_center_yx": origin.tolist(),
                "ctf_mode_for_estimation": "none" if transfer is None else ctf_mode,
                "wiener_balance": (float(np.mean(wiener_balance))
                                   if transfer is not None and ctf_mode == "wiener"
                                   else None),
                "phase_flipped_for_estimation": (
                    transfer is not None and ctf_mode == "flip"),
                "contrast_sign_for_estimation": float(contrast_sign),
                "relion_poses_used": False, "reference_map_used": False,
                "applied_shift_convention": "xy; positive right/down; exp(-2*pi*i*f.dot(shift))",
                "normalization": "estimator copy only: subtract image minimum, divide by range",
                "criterion": "sum_r abs(1 - cumulative_mass_r / max_candidate_total_mass)",
                "annuli": "ceil(distance), restricted to distance <= radius",
                "candidate_batch_size": self.candidate_batch_size,
                "image_batch_size": self.image_batch_size,
            })


def estimate_image_centers(images, radius, *, device="cpu", ctf=None,
                           ctf_mode="flip", wiener_balance=0.05,
                           contrast_sign=1., max_shift=None,
                           return_score_maps=False):
    """Low-level, unshrunk FB criterion; use estimate_dataset_shifts for a dataset."""
    return NoisyImageCenterer(radius, device=device).estimate(
        images, ctf=ctf, ctf_mode=ctf_mode, wiener_balance=wiener_balance,
        contrast_sign=contrast_sign, max_shift=max_shift,
        return_score_maps=return_score_maps)


@dataclass
class FourierMean:
    coefficients: np.ndarray
    variance: np.ndarray
    count: int


def weighted_fourier_mean(images, ctf=None, noise_psd=None, *, filter_indices=None,
                          contrast_sign=1.0, batch_size=128):
    """CTF/noise WLS estimate of the unmasked Cartesian image Fourier mean.

    Arrays and PSD/CTF grids are centered (DC at side//2). ``noise_psd`` has
    convention E|fft2(noise)|^2/side^2: constant PSD equals pixel variance.
    ``ctf`` can be one grid, a per-image bank, or a bank with filter_indices.
    PSD may likewise be shared or a bank selected by filter_indices.
    Returned coefficients use fft2(image)/side^2; variance is the expected
    squared modulus of its error, not the real-component variance.

    Independence of latent images and CTF/noise group is required to interpret
    this weighted mean as a common population mean.
    """
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[-2] != images.shape[-1] or not len(images):
        raise ValueError("images must be nonempty (N, side, side)")
    n, side, _ = images.shape
    if not np.isfinite(images).all() or contrast_sign not in (-1, 1):
        raise ValueError("finite images and contrast_sign +/-1 required")
    ids = np.arange(n) if filter_indices is None else np.asarray(filter_indices)
    if ids.shape != (n,) or ids.dtype.kind not in "iu":
        raise ValueError("filter_indices must be integer shape (N,)")

    def batch_field(value, start, stop, default):
        if value is None:
            return default
        value = np.asarray(value)
        if value.shape == (side, side):
            return value
        if value.ndim == 3 and value.shape[1:] == (side, side):
            return value[ids[start:stop]]
        raise ValueError("CTF/PSD must be a grid or an indexed grid bank")

    numerator = np.zeros((side, side), dtype=np.complex128)
    normal = np.zeros((side, side), dtype=np.float64)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        transfer = batch_field(ctf, start, stop, 1.0)
        psd = batch_field(noise_psd, start, stop, 1.0)
        transfer, psd = np.asarray(transfer), np.asarray(psd)
        if not np.isfinite(transfer).all() or not np.isfinite(psd).all() or np.any(psd <= 0):
            raise ValueError("finite CTF and finite strictly positive PSD required")
        observed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(images[start:stop], axes=(-2, -1)), axes=(-2, -1)),
                                   axes=(-2, -1)) * (contrast_sign / side**2)
        precision = side**2 / psd
        numerator += np.sum(observed * transfer * precision, axis=0)
        contribution = transfer**2 * precision
        normal += ((stop-start) * contribution if contribution.ndim <= 2
                   else np.sum(contribution, axis=0))
    mean = np.divide(numerator, normal, out=np.zeros_like(numerator), where=normal > 0)
    variance = np.divide(1.0, normal, out=np.full_like(normal, np.inf), where=normal > 0)
    return FourierMean(mean, variance, n)


def _samples(mean, min_shell, max_shell):
    side = mean.coefficients.shape[0]
    q = np.arange(side) - side//2
    yy, xx = np.meshgrid(q, q, indexing="ij")
    radius = np.hypot(xx, yy)
    # One sample per Hermitian pair; DC and Nyquist are never included.
    mask = ((radius >= min_shell) & (radius <= max_shell)
            & ((xx > 0) | ((xx == 0) & (yy > 0)))
            & (np.abs(xx) < side/2) & (np.abs(yy) < side/2)
            & np.isfinite(mean.variance) & (mean.variance > 0))
    if mask.sum() < 8:
        raise ValueError("too few observable nonzero low-frequency samples")
    coords = np.column_stack((xx[mask]/side, yy[mask]/side))
    return coords, mean.coefficients[mask], 2.0/mean.variance[mask]


def _profile(coords, observed, precision, model):
    """Return a translation-profiled residual and auxiliary diagnostics."""
    weight = np.sqrt(precision)
    # Scaling the Taylor coordinates changes conditioning, not their span.
    q = coords / max(np.max(np.linalg.norm(coords, axis=1)), 1e-12)
    x, y = q.T
    even = np.column_stack((np.ones(len(q)), x*x, x*y, y*y))
    odd = np.column_stack((x**3, x*x*y, x*y*y, y**3))
    if model == "taylor_com":
        a = even * weight[:, None]
        b = odd * weight[:, None]
        # QR residual projection is stable and leaves no free linear odd term.
        qa = np.linalg.qr(a, mode="reduced")[0]
        qb = np.linalg.qr(b, mode="reduced")[0]

    def objective(correction):
        aligned = observed * np.exp(-2j*np.pi*(coords @ correction))
        if model == "even_phase":
            residual = aligned.imag * weight
            return float(residual @ residual)
        if model != "taylor_com":
            raise ValueError("unknown model")
        re, im = aligned.real * weight, aligned.imag * weight
        re = re - qa @ (qa.T @ re)
        im = im - qb @ (qb.T @ im)
        return float(re @ re + im @ im)

    return objective


def estimate_mean_phase(mean, *, min_shell=1, max_shell=4, max_shift=None,
                        model="even_phase", seed_correction=None):
    """Profile a common correction; correction is the array shift to APPLY."""
    side = mean.coefficients.shape[0]
    limit = float(side/4-1 if max_shift is None else max_shift)
    if not 0 <= limit < side/2 or min_shell < 1:
        raise ValueError("require 0 <= max_shift < side/2 and exclude DC")
    coords, observed, precision = _samples(mean, min_shell, max_shell)
    objective = _profile(coords, observed, precision, model)
    grid = np.arange(-int(limit), int(limit)+1, dtype=float)
    candidates = np.array(np.meshgrid(grid, grid, indexing="xy")).reshape(2, -1).T
    scores = np.array([objective(c) for c in candidates])
    starts = candidates[np.argsort(scores)[:8]]
    starts = np.vstack((starts, np.zeros(2)))
    if seed_correction is not None:
        starts = np.vstack((starts, np.clip(seed_correction, -limit, limit)))
    fits = [minimize(objective, s, method="L-BFGS-B", bounds=[(-limit, limit)]*2,
                     options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 200})
            for s in starts]
    fit = min(fits, key=lambda r: (r.fun, np.linalg.norm(r.x)))
    result = np.asarray(fit.x)
    return dict(model=model, correction_xy_pixels=result.tolist(),
                displacement_xy_pixels=(-result).tolist(), min_shell=float(min_shell),
                max_shell=float(max_shell), max_shift_pixels=limit,
                n_independent_frequencies=len(coords), objective=float(fit.fun),
                zero_objective=objective(np.zeros(2)),
                gain_vs_no_motion=objective(np.zeros(2))-float(fit.fun),
                real_signal_chi2=float(np.sum(precision*observed.real**2)),
                boundary_hit=bool(np.any(np.abs(result) > limit-.05)),
                optimizer_converged=bool(fit.success))


def estimate_global_center(images, ctf=None, noise_psd=None, *, filter_indices=None,
                           contrast_sign=1, min_shell=1, max_shell=4,
                           max_shift=None, n_splits=4, seed=0):
    """Common-center estimates with disjoint-image and frequency-band checks.

    ``guarded_correction_xy_pixels`` is conservatively zero unless split and
    band evidence agree. Its thresholds are diagnostic defaults, not optimized
    on RELION labels. The unguarded estimates are always returned for audit.
    A consistent non-even ensemble can still fool this guard; inspect Taylor
    fit and downstream paired reconstruction rather than treating it as proof.
    """
    images = np.asarray(images)
    n = len(images)
    if n_splits < 2 or n < 16*n_splits:
        raise ValueError("need >=2 splits with >=16 images per split")
    indices = np.arange(n) if filter_indices is None else np.asarray(filter_indices)
    full = weighted_fourier_mean(images, ctf, noise_psd, filter_indices=indices,
                                 contrast_sign=contrast_sign)
    main = estimate_mean_phase(full, min_shell=min_shell, max_shell=max_shell,
                               max_shift=max_shift)
    cubic = estimate_mean_phase(full, min_shell=min_shell, max_shell=max_shell,
                                max_shift=max_shift, model="taylor_com",
                                seed_correction=main["correction_xy_pixels"])
    groups = np.array_split(np.random.default_rng(seed).permutation(n), n_splits)
    splits, means = [], []
    for group in groups:
        m = weighted_fourier_mean(images[group], ctf, noise_psd,
                                  filter_indices=indices[group], contrast_sign=contrast_sign)
        means.append(m)
        splits.append(estimate_mean_phase(m, min_shell=min_shell, max_shell=max_shell,
                                          max_shift=max_shift))
    correction = np.array(main["correction_xy_pixels"])
    split_xy = np.array([s["correction_xy_pixels"] for s in splits])
    split_rms = float(np.sqrt(np.mean(np.sum((split_xy-correction)**2, axis=1))))
    heldout = []
    for k, m in enumerate(means):
        others = np.mean(np.delete(split_xy, k, axis=0), axis=0)
        coords, obs, pre = _samples(m, min_shell, max_shell)
        cost = _profile(coords, obs, pre, "even_phase")
        heldout.append(float(cost(np.zeros(2))-cost(others)))
    band_specs = [(min_shell, max(2, max_shell-1)),
                  (min(max_shell-1, min_shell+1), max_shell)]
    bands = [estimate_mean_phase(full, min_shell=lo, max_shell=hi,
                                 max_shift=max_shift) for lo, hi in band_specs]
    band_difference = float(np.linalg.norm(np.subtract(
        bands[0]["correction_xy_pixels"], bands[1]["correction_xy_pixels"])))
    reasons = []
    if main["boundary_hit"]:
        reasons.append("search_boundary")
    if split_rms > 1.5:
        reasons.append("image_split_disagreement")
    if band_difference > 2:
        reasons.append("frequency_band_disagreement")
    if min(heldout) < 0 or sum(heldout) < 12:
        reasons.append("insufficient_heldout_phase_gain")
    return dict(n_images=n, side=int(images.shape[-1]),
                correction_convention="Array translation to apply: (dx,dy), positive dx moves right",
                mean_assumptions="Clean population independent of CTF/noise group; stationary centered Gaussian noise",
                even_phase_assumption="Centered ensemble Fourier mean real/even, e.g. uniform in-plane orientation",
                taylor_assumption="Common zero-COM gauge; cubic low-frequency Taylor approximation; DC excluded",
                dc_used=False, relion_poses_used=False, reference_map_used=False,
                image_positive_clipping=False, mean_radialized=False,
                main=main, taylor_com=cubic, splits=splits,
                split_vector_rms_pixels=split_rms,
                heldout_phase_gains=heldout, bands=bands,
                band_difference_pixels=band_difference,
                taylor_vs_even_difference_pixels=float(np.linalg.norm(
                    correction-np.array(cubic["correction_xy_pixels"]))),
                guard_passed=not reasons, guard_failure_reasons=reasons,
                guarded_correction_xy_pixels=(correction if not reasons else np.zeros(2)).tolist())


def _replicas(plus, minus):
    plus, minus = np.asarray(plus), np.asarray(minus)
    if (plus.shape != minus.shape or plus.ndim != 3 or plus.shape[-1] != 2
            or plus.shape[0] < 1 or plus.shape[1] < 2
            or np.iscomplexobj(plus) or np.iscomplexobj(minus)
            or plus.dtype.kind not in "fiu" or minus.dtype.kind not in "fiu"
            or not np.isfinite(plus).all() or not np.isfinite(minus).all()):
        raise ValueError("Replicas must be finite real matching (noise_seeds, images>=2, 2) arrays")
    return plus.astype(np.float64), minus.astype(np.float64)


def _moments(plus, minus, counts=None):
    """One covariance per replica seed, sharing image resampling weights."""
    seeds, n, _ = plus.shape
    weights = np.ones(n) if counts is None else np.asarray(counts, dtype=np.float64)
    mean_plus = np.einsum("n,sni->si", weights, plus)/n
    mean_minus = np.einsum("n,sni->si", weights, minus)/n
    cross = (np.einsum("n,sni,snj->sij", weights, plus, minus)
             - n*mean_plus[:, :, None]*mean_minus[:, None, :])/(n-1)
    c_seeds = .5*(cross+cross.transpose(0, 2, 1))
    differences = plus-minus
    d_seeds = np.einsum("n,sni,snj->sij", weights, differences, differences)/(4*n)
    return c_seeds.mean(axis=0), d_seeds.mean(axis=0), c_seeds, d_seeds, mean_plus, mean_minus


def _point_lambda(c, d):
    persistent = max(float(np.trace(c)), 0.)
    error = max(float(np.trace(d)), 0.)
    return persistent/(persistent+error) if persistent+error > 0 else 0.


def calibrate_replica_shrinkage(plus, minus, *, seed=271, n_bootstrap=1000, confidence=.95):
    """Calibrate one scalar shrinkage from paired (seed, image, XY) shifts.

    For independent Gaussian replicas with noise covariance 2*Sigma, the
    local model r+/- = u + e+/- has Cov(e+/-)=2*V. Therefore the symmetric
    cross-covariance C estimates Cov(u), and E[(r+-r-)(r+-r-)^T]/4 estimates
    the estimator error covariance V at original noise level Sigma.

    A zero-mean displacement prior is imposed; reported replica means are
    never added to the correction. The raw trace of C is clipped at zero
    for the scalar rule, not replaced by the trace of a PSD projection.
    Images, not artificial noise seeds, are bootstrapped: every seed uses
    the SAME resampled image indices. Micrograph dependence is not modeled.
    """
    plus, minus = _replicas(plus, minus)
    if (isinstance(n_bootstrap, (bool, np.bool_))
            or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1
            or not np.isscalar(confidence) or not np.isfinite(confidence)
            or not 0 < confidence < 1):
        raise ValueError("Need a positive integer bootstrap count and confidence in (0,1)")
    c, d, cs, ds, mean_plus, mean_minus = _moments(plus, minus)
    n = plus.shape[1]
    rng = np.random.default_rng(seed)
    bootstrap_trace = np.empty(n_bootstrap)
    bootstrap_lambda = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        cb, db, *_ = _moments(plus, minus, counts)
        bootstrap_trace[b] = np.trace(cb)
        bootstrap_lambda[b] = _point_lambda(cb, db)
    tail = (1-confidence)/2
    trace_ci = np.quantile(bootstrap_trace, [tail, 1-tail])
    lambda_ci = np.quantile(bootstrap_lambda, [tail, 1-tail])
    point = _point_lambda(c, d)
    guard = bool(trace_ci[0] > 0 and point > 0)
    eigs, vectors = np.linalg.eigh(c)
    c_psd = (vectors*np.maximum(eigs, 0)[None, :])@vectors.T
    return dict(
        C=c.tolist(), D=d.tolist(), C_psd_diagnostic=c_psd.tolist(),
        C_eigenvalues_raw=eigs.tolist(), D_eigenvalues_raw=np.linalg.eigvalsh(d).tolist(),
        C_per_noise_seed=cs.tolist(), D_per_noise_seed=ds.tolist(),
        trace_C=float(np.trace(c)), trace_D=float(np.trace(d)), trace_C_ci=trace_ci.tolist(),
        lambda_point=float(point), lambda_selected=float(point if guard else 0.),
        lambda_bootstrap_ci=lambda_ci.tolist(), guard_passed=guard,
        guard_rule="Select zero unless the two-sided bootstrap interval for raw trace(C) is strictly positive",
        confidence=float(confidence), n_bootstrap=int(n_bootstrap), bootstrap_seed=int(seed),
        n_images=int(n), n_noise_seeds=int(plus.shape[0]), bootstrap_unit="image",
        bootstrap_shared_across_noise_seeds=True,
        replica_means=dict(plus=mean_plus.tolist(), minus=mean_minus.tolist()),
        residual_prior_mean_xy=[0., 0.], means_added_back=False,
        warnings=[
            "Noise replicas are independent only for Gaussian observation noise and correct added-noise covariance.",
            "Double-noise to original-noise extrapolation assumes locally additive estimator errors.",
            "Integer quantization and search-boundary saturation can invalidate this extrapolation.",
            "Persistent shape-dependent center bias contributes to C and is not removed by this guard.",
            "Image bootstrap does not account for dependence between particles from one micrograph.",
        ])


def shrink_corrections(corrections_xy, calibration):
    """Apply the selected scalar to original-observation corrections, no mean."""
    corrections = np.asarray(corrections_xy)
    coefficient = calibration["lambda_selected"]
    if (corrections.ndim != 2 or corrections.shape[1] != 2
            or np.iscomplexobj(corrections) or not np.isfinite(corrections).all()
            or not np.isfinite(coefficient) or not 0 <= coefficient <= 1):
        raise ValueError("Need finite (images,2) corrections and a selected lambda in [0,1]")
    return np.asarray(corrections, dtype=np.float64)*coefficient


def generate_colored_noise(shape, noise_psd, *, seed=271, nyquist_zero=True):
    """Draw real stationary Gaussian noise from one centered shared PSD.

    Convention: E|fft2(noise)|^2/side^2 = noise_psd. For even grids the default
    removes both Nyquist lines; the effective PSD is zero on those lines.
    Apply that same mask to observations when relying on replica independence.
    """
    shape = tuple(shape)
    psd = np.asarray(noise_psd)
    if (len(shape) < 2 or any(not isinstance(s, (int, np.integer)) or s < 1 for s in shape)
            or shape[-2] != shape[-1] or psd.shape != shape[-2:]
            or np.iscomplexobj(psd) or psd.dtype.kind not in "fiu"
            or not np.isfinite(psd).all() or np.any(psd < 0)):
        raise ValueError("Need a square image shape and finite real nonnegative shared PSD")
    side = shape[-1]
    unshifted = np.fft.ifftshift(np.asarray(psd, dtype=np.float64))
    reverse = (-np.arange(side)) % side
    if not np.allclose(unshifted, unshifted[reverse][:, reverse], rtol=1e-10, atol=1e-12):
        raise ValueError("Real stationary noise requires a Hermitian-symmetric PSD")
    spectrum = np.fft.fft2(np.random.default_rng(seed).normal(size=shape), axes=(-2, -1))
    spectrum *= np.sqrt(unshifted)
    if nyquist_zero and side % 2 == 0:
        spectrum[..., side//2, :] = 0
        spectrum[..., :, side//2] = 0
    noise = np.fft.ifft2(spectrum, axes=(-2, -1))
    if np.max(np.abs(noise.imag)) > 1e-10*max(1., float(np.max(np.abs(noise.real)))):
        raise FloatingPointError("Noise generation broke real-image symmetry")
    return noise.real


@dataclass(frozen=True)
class CenteringConfig:
    """Dataset centering settings in fractions of the input image side.

    ``ctf_mode`` controls the individual radial criterion; the common-phase
    estimator always uses supplied CTFs. ``phase_flip`` is an alias for ``flip``.
    The default individual correction is calibrated, not a hard radial argmin.
    """

    calibration_images: int = 8192
    calibration_seed: int = 271
    common_max_shell: int = 4
    common_max_shift_fraction: float = 15/64
    individual_centering: bool = True
    individual_shrinkage: bool = True
    individual_calibration_images: int = 4096
    individual_calibration_seed: int = 1983
    support_radius_fraction: float = 16/64
    residual_max_shift_fraction: float = 4/64
    wiener_balance: float = .05
    ctf_mode: str = "wiener"
    contrast_sign: int = -1
    reject_boundary_residuals: bool = False
    batch_size: int = 128
    device: str = "cpu"
    fle_root: str | None = None

    def __post_init__(self):
        for name in ("calibration_images", "individual_calibration_images"):
            value = _positive_integer(getattr(self, name), name)
            if value < 64:
                raise ValueError(f"{name} must be at least 64")
            object.__setattr__(self, name, value)
        for name in ("common_max_shell", "batch_size"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        for name in ("calibration_seed", "individual_calibration_seed"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        for name in ("individual_centering", "individual_shrinkage", "reject_boundary_residuals"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be Boolean")
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in ("common_max_shift_fraction", "support_radius_fraction", "residual_max_shift_fraction"):
            value = getattr(self, name)
            lower = 0 if name == "residual_max_shift_fraction" else np.nextafter(0., 1.)
            if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                    or np.iscomplexobj(value) or not np.isfinite(value) or not lower <= value < .5):
                raise ValueError(f"{name} must be finite in {'[0' if lower == 0 else '(0'}, 0.5)")
            object.__setattr__(self, name, float(value))
        if (isinstance(self.wiener_balance, (bool, np.bool_)) or not np.isscalar(self.wiener_balance)
                or np.iscomplexobj(self.wiener_balance) or not np.isfinite(self.wiener_balance)
                or self.wiener_balance <= 0):
            raise ValueError("wiener_balance must be finite and positive")
        object.__setattr__(self, "wiener_balance", float(self.wiener_balance))
        if self.ctf_mode not in ("wiener", "flip", "phase_flip", "none"):
            raise ValueError("ctf_mode must be 'wiener', 'flip' (or 'phase_flip'), or 'none'")
        if self.ctf_mode == "phase_flip":
            object.__setattr__(self, "ctf_mode", "flip")
        if isinstance(self.contrast_sign, (bool, np.bool_)) or self.contrast_sign not in (-1, 1):
            raise ValueError("contrast_sign must be +1 or -1")
        object.__setattr__(self, "contrast_sign", int(self.contrast_sign))
        if self.device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        if self.fle_root is not None:
            object.__setattr__(self, "fle_root", str(Path(self.fle_root).expanduser()))


def estimate_background_noise_psd(images, *, background_radius=.9, batch_size=128):
    """Estimate a centered radial PSD from pooled image-background periodograms.

    Returns ``(psd_grid, shell_profile)``. Constant PSD v denotes pixel noise
    variance v, i.e. E|fft2(noise)|²/side². The hard annular window smooths the
    estimate; the masked-mean DC is replaced by shell 1. This PSD calibrates
    centering, not the separate FLE-sampled PSD used by CWF denoising.
    """
    images = np.asarray(images)
    batch_size = _positive_integer(batch_size, "batch_size")
    if (images.ndim != 3 or len(images) < 1 or images.shape[-2] != images.shape[-1]
            or images.shape[-1] < 2 or images.dtype.kind not in "fiu" or np.iscomplexobj(images)):
        raise ValueError("Provide nonempty real square image stacks")
    if (isinstance(background_radius, (bool, np.bool_)) or not np.isscalar(background_radius)
            or np.iscomplexobj(background_radius) or not np.isfinite(background_radius)
            or not 0 < background_radius < np.sqrt(2)):
        raise ValueError("background_radius must be finite in (0, sqrt(2))")
    _check_finite_batches(images, batch_size)
    side = images.shape[-1]
    axis = (np.arange(side)-side//2)/(side/2)
    distance = np.hypot(axis[:, None], axis[None, :])
    mask = distance >= background_radius
    mean_mask = distance > background_radius
    if not mask.any() or not mean_mask.any():
        raise ValueError("background_radius leaves no background samples")
    shell = np.rint(np.hypot(*np.meshgrid(np.arange(side)-side//2,
                                         np.arange(side)-side//2))).astype(int)
    counts = np.bincount(shell.ravel())
    total = np.zeros_like(counts, dtype=np.float64)
    for first in range(0, len(images), batch_size):
        raw = images[first:first+batch_size]
        means = np.mean(raw[:, mean_mask].astype(np.float64), axis=1)
        batch = (raw-means[:, None, None]).astype(np.float32)
        power = abs(np.fft.fftshift(np.fft.fft2(batch*mask), axes=(-2, -1)))**2
        power = np.sum(power, axis=0, dtype=np.float64)/mask.sum()
        total += np.bincount(shell.ravel(), weights=power.ravel(), minlength=len(total))
    radial = np.divide(total, counts*len(images), out=np.zeros_like(total), where=counts > 0)
    if len(radial) > 1:
        radial[0] = radial[1]
    positive = radial[radial > 0]
    floor = np.median(positive)*1e-8 if len(positive) else np.finfo(float).tiny
    radial = np.maximum(radial, floor)
    return radial[shell], radial


def _estimator_images(images, corrections_xy):
    """Mask the estimator copy and replica noise identically, even for zero shifts."""
    shifted = apply_image_shifts(images, corrections_xy)
    side = shifted.shape[-1]
    fourier = np.fft.fft2(np.fft.ifftshift(shifted, axes=(-2, -1)))
    if side % 2 == 0:
        fourier[:, side//2, :] = 0
        fourier[:, :, side//2] = 0
    return np.asarray(np.fft.fftshift(np.fft.ifft2(fourier), axes=(-2, -1)).real,
                      dtype=shifted.dtype)


def estimate_dataset_shifts(images, ctf=None, noise_psd=None, *, ctf_indices=None, config=None):
    """Estimate applied XY shifts from raw observations and CTF/noise metadata.

    The common correction is guarded using image splits and frequency bands.
    Individual radial shifts are scaled by a dataset-calibrated noise-replica
    factor. A zero factor skips target-image radial estimation. Apply returned
    ``shifts_xy`` once to the ORIGINAL observations with apply_image_shifts.
    No images are discarded here, and input order is never changed.
    """
    config = CenteringConfig() if config is None else config
    if not isinstance(config, CenteringConfig):
        raise ValueError("config must be a CenteringConfig")
    images = np.asarray(images)
    if (images.ndim != 3 or images.shape[-2] != images.shape[-1]
            or images.shape[-1] < 1 or images.dtype.kind != "f"):
        raise ValueError("Provide a finite floating-point (images,side,side) stack")
    _check_finite_batches(images, config.batch_size)
    n, side, _ = images.shape
    bank = None if ctf is None else np.asarray(ctf)
    ids = np.arange(n) if ctf_indices is None else np.asarray(ctf_indices)
    if ids.shape != (n,) or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("ctf_indices must be a matching nonnegative integer vector")
    if bank is not None:
        if (bank.ndim not in (2, 3) or bank.shape[-2:] != (side, side)
                or bank.dtype.kind not in "fiu" or np.iscomplexobj(bank)):
            raise ValueError("CTFs must be finite real centered Fourier grids")
        _check_finite_batches(bank if bank.ndim == 3 else bank[None], config.batch_size)
        if bank.ndim == 3 and (len(bank) < 1 or np.any(ids >= len(bank))):
            raise ValueError("CTF bank indices are out of range")
    if noise_psd is not None:
        noise_psd = np.asarray(noise_psd)
        if (noise_psd.shape != (side, side) or noise_psd.dtype.kind not in "fiu"
                or np.iscomplexobj(noise_psd) or not np.isfinite(noise_psd).all()
                or np.any(noise_psd <= 0)):
            raise ValueError("Provide a finite strictly positive shared centered noise-PSD grid")
        reverse = (-np.arange(side)) % side
        unshifted = np.fft.ifftshift(noise_psd)
        if not np.allclose(unshifted, unshifted[reverse][:, reverse], rtol=1e-10, atol=1e-12):
            raise ValueError("Real stationary noise requires a Hermitian-symmetric PSD")
    started = time.monotonic()
    calibration = np.sort(np.random.default_rng(config.calibration_seed).choice(
        n, min(n, config.calibration_images), replace=False))
    residual = np.zeros((n, 2))
    accepted, valid, boundary = (np.zeros(n, dtype=bool) for _ in range(3))
    scores = np.full(n, np.nan)
    calibration_arrays = {}
    individual_calibration = None
    radius = max(1, int(round(config.support_radius_fraction*side)))
    residual_limit = max(0, int(round(config.residual_max_shift_fraction*side)))
    status = "complete" if n >= 64 else "insufficient_images"
    flat = n > 0 and all(not np.ptp(images[first:first+config.batch_size], axis=(-2, -1)).any()
                        for first in range(0, n, config.batch_size))
    if n >= 64 and flat:
        status = "flat_images"
    if status != "complete":
        common_report = dict(n_images=n, side=side, guard_passed=False,
            guard_failure_reasons=[status], guarded_correction_xy_pixels=[0., 0.],
            main=None, splits=[], bands=[], reference_map_used=False, relion_poses_used=False,
            diagnostic="No centering fit performed; too few observations or all images spatially flat")
        common = np.zeros(2)
        shrink = 0.
    else:
        if config.individual_centering and config.individual_shrinkage and noise_psd is None:
            raise ValueError("Calibrated individual centering requires an estimated noise PSD")
        if noise_psd is not None and np.min(noise_psd) < side**2/np.finfo(float).max:
            raise ValueError("noise_psd is too small for stable Fourier-mean precision")
        common_report = estimate_global_center(images[calibration], bank, noise_psd,
            filter_indices=ids[calibration], contrast_sign=config.contrast_sign,
            max_shell=config.common_max_shell, max_shift=config.common_max_shift_fraction*side, seed=0)
        common = np.asarray(common_report["guarded_correction_xy_pixels"], dtype=float)
        shrink = 1.
        if config.individual_centering:
            if 2*radius+1 > side:
                raise ValueError("The particle support disk must fit inside the image")
            kwargs = dict(device=config.device)
            if config.fle_root is not None:
                kwargs["fle_root"] = config.fle_root
            centerer = NoisyImageCenterer(radius, **kwargs)

            def transfer(positions):
                if config.ctf_mode == "none" or bank is None:
                    return None
                return bank if bank.ndim == 2 else bank[ids[positions]]

            mode = "wiener" if config.ctf_mode == "none" else config.ctf_mode
            if config.individual_shrinkage:
                positions = np.sort(np.random.default_rng(config.individual_calibration_seed).choice(
                    n, min(n, config.individual_calibration_images), replace=False))
                observed = _estimator_images(images[positions], np.broadcast_to(common, (len(positions), 2)))
                plus = np.empty((2, len(positions), 2))
                minus = np.empty_like(plus)
                flags = np.empty((2, 2, len(positions)), dtype=bool)
                replica_valid = np.empty_like(flags)
                for seed_index, noise_seed in enumerate((113, 229)):
                    noise = generate_colored_noise(observed.shape, noise_psd, seed=noise_seed)
                    for sign_index, sign in enumerate((1, -1)):
                        replica = plus if sign > 0 else minus
                        for first in range(0, len(positions), config.batch_size):
                            stop = min(first+config.batch_size, len(positions))
                            estimate = centerer.estimate(observed[first:stop]+sign*noise[first:stop],
                                ctf=transfer(positions[first:stop]), ctf_mode=mode,
                                wiener_balance=config.wiener_balance, contrast_sign=config.contrast_sign,
                                max_shift=residual_limit)
                            replica[seed_index, first:stop] = estimate.shifts_xy
                            flags[seed_index, sign_index, first:stop] = estimate.boundary_hit
                            replica_valid[seed_index, sign_index, first:stop] = estimate.valid
                    print(f"Centering uncertainty calibration noise seed {noise_seed} complete", flush=True)
                individual_calibration = calibrate_replica_shrinkage(plus, minus)
                shrink = float(individual_calibration["lambda_selected"])
                calibration_arrays.update(individual_calibration_positions=positions,
                    replica_plus_shifts_xy=plus, replica_minus_shifts_xy=minus,
                    replica_boundary=flags, replica_valid=replica_valid)
                del observed, noise
            for first in range(0, n if shrink > 0 else 0, config.batch_size):
                stop = min(first+config.batch_size, n)
                positions = np.arange(first, stop)
                transformer = _estimator_images if config.individual_shrinkage else apply_image_shifts
                corrected = transformer(images[first:stop], np.broadcast_to(common, (stop-first, 2)))
                estimate = centerer.estimate(corrected, ctf=transfer(positions), ctf_mode=mode,
                    wiener_balance=config.wiener_balance, contrast_sign=config.contrast_sign,
                    max_shift=residual_limit)
                residual[first:stop] = estimate.shifts_xy
                valid[first:stop], boundary[first:stop] = estimate.valid, estimate.boundary_hit
                scores[first:stop] = estimate.scores
                print(f"Individual radial centering {stop}/{n}", flush=True)
            accepted = valid & (~boundary if config.reject_boundary_residuals else True)
    shifts = np.broadcast_to(common, (n, 2)).copy()
    shifts[accepted] += shrink*residual[accepted]
    fraction = lambda values: float(values.mean()) if len(values) else 0.
    report = dict(method=METHOD, method_version=METHOD_VERSION, status=status,
        configuration=asdict(config), n_images=n, input_side=side, common=common_report,
        radius_pixels=radius, residual_max_shift_per_axis=residual_limit,
        residual_accepted_fraction=fraction(accepted), residual_boundary_fraction=fraction(boundary),
        individual_shrinkage_factor=shrink if config.individual_centering else None,
        individual_calibration=individual_calibration,
        individual_target_estimation_skipped=bool(config.individual_centering and shrink == 0),
        replica_noise_seeds=[113, 229], replica_bootstrap_replicates=1000, replica_bootstrap_confidence=.95,
        ctf_used=bank is not None,
        noise_weights="provided shared PSD" if noise_psd is not None else "white weights assumed",
        source="raw images and CTF/noise metadata only", MoM_used=False,
        reference_map_used=False, relion_poses_or_origins_used=False, input_order_preserved=True,
        shift_convention="XY input pixels, positive right/down; apply once to original input",
        individual_step_validated_for_MoM=False,
        warnings=["Common correction removes a shared bias, not every residual displacement.",
                  "Radial validity and boundary flags are not calibrated confidence probabilities.",
                  "Persistent shape-dependent centering bias can change higher moments.",
                  "Replica calibration uses estimated PSD and approximate noise-level transfer; rejection does not prove physical zero shift.",
                  "Periodic translation cannot recover signal truncated at extraction."],
        elapsed_seconds=time.monotonic()-started)
    arrays = dict(shifts_xy=shifts, common_shift_xy=common, residual_estimates_xy=residual,
        residual_accepted=accepted, residual_valid=valid, residual_boundary=boundary,
        residual_scores=scores, calibration_positions=calibration)
    arrays.update(calibration_arrays)
    return arrays, report
