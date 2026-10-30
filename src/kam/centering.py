"""Image-only common-offset (global) centering.

The dataset estimator removes a guarded common Fourier-phase offset: the mean of the CTF/noise-weighted images has a real (even) Fourier
transform under uniform in-plane rotations, so the entire phase of the mean comes from a shift common to the dataset. The estimate is kept
only when disjoint image splits and frequency bands agree, and is zero otherwise. No reference volume, viewing angles or particle-origin
metadata is used. The remaining per-image shifts are not estimated: they are modeled in the moments (see ``kam.shift_distribution``).

Estimator-only sign correction and intensity normalization are never applied to the returned observation: estimate a displacement, then use
``apply_image_shifts`` on the original images. Pixel size and CTF metadata do not change. Translation uses periodic boundaries; it cannot
recover signal that was already cut off by particle extraction.

``shifts_xy`` is the displacement to APPLY, in input pixels: positive x moves right and positive y moves down.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.optimize import minimize


METHOD = "common_phase_guarded_fb"
METHOD_VERSION = 4


__all__ = [
    "METHOD", "METHOD_VERSION", "CenteringConfig", "estimate_dataset_shifts",
    "estimate_background_noise_psd", "apply_image_shifts",
    "FourierMean", "weighted_fourier_mean",
    "estimate_mean_phase", "estimate_global_center",
]


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


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


@dataclass(frozen=True)
class CenteringConfig:
    """Common-shift estimator settings.

    ``common_max_shift_fraction`` bounds the common shift in fractions of the input image side.
    """

    calibration_images: int = 8192
    calibration_seed: int = 271
    common_max_shell: int = 4
    common_max_shift_fraction: float = 15/64
    contrast_sign: int = -1
    batch_size: int = 128

    def __post_init__(self):
        value = _positive_integer(self.calibration_images, "calibration_images")
        if value < 64:
            raise ValueError("calibration_images must be at least 64")
        object.__setattr__(self, "calibration_images", value)
        for name in ("common_max_shell", "batch_size"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        value = self.calibration_seed
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("calibration_seed must be a nonnegative integer")
        object.__setattr__(self, "calibration_seed", int(value))
        value = self.common_max_shift_fraction
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value)
                or not np.isfinite(value) or not 0 < value < .5):
            raise ValueError("common_max_shift_fraction must be finite in (0, 0.5)")
        object.__setattr__(self, "common_max_shift_fraction", float(value))
        if isinstance(self.contrast_sign, (bool, np.bool_)) or self.contrast_sign not in (-1, 1):
            raise ValueError("contrast_sign must be +1 or -1")
        object.__setattr__(self, "contrast_sign", int(self.contrast_sign))


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


def estimate_dataset_shifts(images, ctf=None, noise_psd=None, *, ctf_indices=None, config=None):
    """Estimate the common XY shift to apply from raw observations and CTF/noise metadata.

    The correction is guarded using image splits and frequency bands, and is zero when they disagree.
    Apply the returned ``shifts_xy`` once to the ORIGINAL observations with apply_image_shifts.
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
    else:
        if noise_psd is not None and np.min(noise_psd) < side**2/np.finfo(float).max:
            raise ValueError("noise_psd is too small for stable Fourier-mean precision")
        common_report = estimate_global_center(images[calibration], bank, noise_psd,
            filter_indices=ids[calibration], contrast_sign=config.contrast_sign,
            max_shell=config.common_max_shell, max_shift=config.common_max_shift_fraction*side, seed=0)
        common = np.asarray(common_report["guarded_correction_xy_pixels"], dtype=float)
    shifts = np.broadcast_to(common, (n, 2)).copy()
    report = dict(method=METHOD, method_version=METHOD_VERSION, status=status,
        configuration=asdict(config), n_images=n, input_side=side, common=common_report,
        ctf_used=bank is not None,
        noise_weights="provided shared PSD" if noise_psd is not None else "white weights assumed",
        source="raw images and CTF/noise metadata only", MoM_used=False,
        reference_map_used=False, relion_poses_or_origins_used=False, input_order_preserved=True,
        shift_convention="XY input pixels, positive right/down; apply once to original input",
        warnings=["The common correction removes a shared bias, not every per-image displacement.",
                  "Persistent shape-dependent centering bias can change higher moments.",
                  "Periodic translation cannot recover signal truncated at extraction."],
        elapsed_seconds=time.monotonic()-started)
    arrays = dict(shifts_xy=shifts, common_shift_xy=common, calibration_positions=calibration)
    return arrays, report
