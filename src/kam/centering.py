"""Image-only common-offset (global) centering.

Under uniform in-plane rotations the clean mean image of a particle dataset is point symmetric about a single center, and the only thing that can
move that center away from the middle of the box is a shift common to the dataset. The estimator therefore averages the CTF-corrected images,
finds the point about which that mean is symmetric and returns the displacement that moves it to the middle of the box. No reference volume,
viewing angles or particle-origin metadata is used. The remaining per-image shifts are not estimated: they are modeled in the moments (see
``kam.shift_distribution``).

The estimate is applied only if it is at least ``min_shift_pixels`` (smaller values cannot be told from zero) and lies inside the range where the
symmetry center is unambiguous (it is defined modulo the box size); otherwise the common shift is zero.

Intensity normalization is never applied to the returned observation: estimate a displacement, then use ``apply_image_shifts`` on the original
images. Pixel size and CTF metadata do not change. Translation uses periodic boundaries; it cannot recover signal that was already cut off by
particle extraction.

``shifts_xy`` is the displacement to APPLY, in input pixels: positive x moves right and positive y moves down.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np


METHOD = "mean_symmetry_center"
METHOD_VERSION = 1


__all__ = [
    "METHOD", "METHOD_VERSION", "CenteringConfig", "estimate_dataset_shifts",
    "apply_image_shifts", "ctf_corrected_mean", "mean_center",
]


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


def ctf_corrected_mean(images, ctf=None, ctf_indices=None, batch_size=256):
    """Least-squares CTF-corrected Fourier mean ``sum(H y) / sum(H^2)`` of the images, and its information ``sum(H^2)``.

    Arrays are centered (DC at side//2) and coefficients use fft2(image)/side^2. ``ctf`` is a bank of centered grids selected per image by
    ``ctf_indices``; without it the plain mean (information = number of images) is returned. The information is the inverse variance of the mean
    and is small where the CTF crosses zero.
    """
    n, side, _ = images.shape
    ids = np.arange(n) if ctf_indices is None else np.asarray(ctf_indices)
    numerator = np.zeros((side, side), dtype=np.complex128)
    denominator = np.zeros((side, side), dtype=np.float64)
    for first in range(0, n, batch_size):
        stop = min(first + batch_size, n)
        batch = np.asarray(images[first:stop], dtype=np.float64)
        fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(batch, axes=(-2, -1))), axes=(-2, -1)) / side**2
        if ctf is None:
            numerator += fourier.sum(axis=0)
            denominator += stop - first
        else:
            transfer = np.asarray(ctf)[ids[first:stop]]
            numerator += (transfer * fourier).sum(axis=0)
            denominator += (transfer * transfer).sum(axis=0)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0), denominator


def mean_center(mean, information, max_shell=4):
    """Displacement (dx, dy) in pixels, to APPLY, that moves the point-symmetry center of a Fourier mean to the middle of the box.

    If the mean image is point symmetric about c, its transform is A(k) exp(-2 pi i k.c/side) with real A. Squaring removes the sign of A, and
    f(t) = Re sum_k information(k) mean(k)^2 exp(2 pi i k.t/side), a smooth trigonometric polynomial in t, is the autocorrelation of the symmetric
    part and so peaks at t = 2c; weighting by the information (inverse variance of the mean) is the maximum-likelihood weighting. The peak is found
    on a coarse grid over one period and refined locally. Only shells 1..max_shell are used: DC is the arbitrary background level of raw images.
    """
    side = mean.shape[0]
    q = np.arange(side) - side // 2
    ky, kx = np.meshgrid(q, q, indexing="ij")
    keep = (np.hypot(kx, ky) >= 1) & (np.hypot(kx, ky) <= max_shell)
    k = np.column_stack((kx[keep], ky[keep])) / side
    power = information[keep] * mean[keep] ** 2

    def peak(tx, ty):
        tx, ty = np.meshgrid(tx, ty)
        value = (np.exp(2j * np.pi * (tx.reshape(-1, 1) * k[:, 0] + ty.reshape(-1, 1) * k[:, 1])) @ power).real
        best = int(np.argmax(value))
        return tx.ravel()[best], ty.ravel()[best]

    tx, ty = peak(np.arange(-side / 2, side / 2, 0.5), np.arange(-side / 2, side / 2, 0.5))     # t = 2c is defined modulo the box size
    for step, span in ((0.05, 0.75), (0.005, 0.06)):
        offsets = np.arange(-span, span + step / 2, step)
        tx, ty = peak(tx + offsets, ty + offsets)
    return -np.array([tx, ty]) / 2


@dataclass(frozen=True)
class CenteringConfig:
    """Common-shift estimator settings.

    ``max_shift_fraction`` bounds the common shift in fractions of the input image side (the symmetry center is ambiguous beyond a quarter).
    """

    calibration_images: int = 8192
    calibration_seed: int = 271
    max_shell: int = 4
    min_shift_pixels: float = 0.5
    max_shift_fraction: float = 15/64

    def __post_init__(self):
        for name in ("calibration_images", "max_shell"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.calibration_images < 64:
            raise ValueError("calibration_images must be at least 64")
        if (isinstance(self.calibration_seed, (bool, np.bool_)) or not isinstance(self.calibration_seed, (int, np.integer))
                or self.calibration_seed < 0):
            raise ValueError("calibration_seed must be a nonnegative integer")
        if not np.isfinite(self.min_shift_pixels) or self.min_shift_pixels < 0:
            raise ValueError("min_shift_pixels must be finite and nonnegative")
        if not np.isfinite(self.max_shift_fraction) or not 0 < self.max_shift_fraction < .25:
            raise ValueError("max_shift_fraction must be finite in (0, 0.25)")


def estimate_dataset_shifts(images, ctf=None, ctf_indices=None, config=None):
    """Estimate the common XY shift to apply from raw observations and CTF metadata.

    Returns ``(arrays, report)``; ``arrays["shifts_xy"]`` is the shift to apply to every image once, with apply_image_shifts, to the ORIGINAL
    observations. No images are discarded here, and input order is never changed.
    """
    config = CenteringConfig() if config is None else config
    if not isinstance(config, CenteringConfig):
        raise ValueError("config must be a CenteringConfig")
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[-2] != images.shape[-1] or images.dtype.kind != "f":
        raise ValueError("Provide a finite floating-point (images,side,side) stack")
    n, side, _ = images.shape
    ids = None if ctf_indices is None else np.asarray(ctf_indices)
    if ids is not None and (ids.shape != (n,) or ids.dtype.kind not in "iu"):
        raise ValueError("ctf_indices must be an integer vector with one entry per image")
    started = time.monotonic()
    calibration = np.sort(np.random.default_rng(config.calibration_seed).choice(
        n, min(n, config.calibration_images), replace=False))
    sample = images[calibration]
    if not np.isfinite(sample).all():
        raise ValueError("images must contain only finite values")
    estimate = np.zeros(2)
    if n < 64:
        reason = "too_few_images"
    elif not np.ptp(sample, axis=(-2, -1)).any():
        reason = "flat_images"
    else:
        estimate = mean_center(*ctf_corrected_mean(sample, ctf, None if ids is None else ids[calibration]), config.max_shell)
        if np.any(np.abs(estimate) > config.max_shift_fraction * side):
            reason = "beyond_unambiguous_range"
        elif np.linalg.norm(estimate) < config.min_shift_pixels:
            reason = "below_min_shift"
        else:
            reason = None
    applied = estimate if reason is None else np.zeros(2)
    report = dict(method=METHOD, method_version=METHOD_VERSION, configuration=asdict(config), n_images=n, input_side=side,
        n_calibration_images=len(calibration), estimate_xy_pixels=estimate.tolist(), applied_xy_pixels=applied.tolist(),
        applied=reason is None, not_applied_reason=reason, ctf_used=ctf is not None,
        source="raw images and CTF metadata only", MoM_used=False, reference_map_used=False,
        relion_poses_or_origins_used=False, input_order_preserved=True,
        shift_convention="XY input pixels, positive right/down; apply once to original input",
        warnings=["The common correction removes a shared bias, not every per-image displacement.",
                  "Periodic translation cannot recover signal truncated at extraction."],
        elapsed_seconds=time.monotonic() - started)
    arrays = dict(shifts_xy=np.broadcast_to(applied, (n, 2)).copy(), common_shift_xy=applied, calibration_positions=calibration)
    return arrays, report
