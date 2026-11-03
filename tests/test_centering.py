"""Common-shift (global) centering recovers a known shift of synthetic particle images."""

import numpy as np
import pytest

from kam.centering import CenteringConfig, apply_image_shifts, estimate_dataset_shifts

SIDE = 64
BLOBS = np.array([(1.0, 9.0, 2.0), (0.8, -6.0, 7.0), (0.6, 3.0, -11.0), (0.9, -10.0, -4.0), (0.5, 12.0, 8.0)])     # amplitude, x, y


def particles(shift, n=1500, noise=0.4, ctf=True, seed=0):
    """Images of one asymmetric particle in uniformly random in-plane orientations, a common ``shift`` and per-image jitter."""
    rng = np.random.default_rng(seed)
    psi = rng.uniform(0, 2 * np.pi, n)
    jitter = rng.normal(0, 1.0, (n, 2))
    axis = np.arange(SIDE) - SIDE // 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    images = np.zeros((n, SIDE, SIDE))
    for amplitude, x, y in BLOBS:
        cx = x * np.cos(psi) - y * np.sin(psi) + shift[0] + jitter[:, 0]
        cy = x * np.sin(psi) + y * np.cos(psi) + shift[1] + jitter[:, 1]
        images += amplitude * np.exp(-((xx - cx[:, None, None]) ** 2 + (yy - cy[:, None, None]) ** 2) / (2 * 3.0**2))
    bank = ids = None
    if ctf:                                                           # two radial CTFs, the second crosses zero inside the estimator's band
        k = np.hypot(*np.meshgrid(axis, axis, indexing="ij")) / SIDE
        bank = np.stack([-0.1 + 0 * k, np.sin(2 * np.pi * (k - 0.03) * 16) ** 3 - 0.1])
        ids = rng.integers(0, 2, n)
        fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(images, axes=(-2, -1))), axes=(-2, -1)) * bank[ids]
        images = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fourier, axes=(-2, -1))), axes=(-2, -1)).real
    return (images + rng.normal(0, noise, images.shape)).astype(np.float32), bank, ids


@pytest.mark.parametrize("shift", [(3.3, -2.1), (-7.0, 6.0), (0.0, 9.5)])
def test_recovers_common_shift(shift):
    images, bank, ids = particles(shift)
    arrays, report = estimate_dataset_shifts(images, bank, ids)
    assert report["applied"] and report["not_applied_reason"] is None
    np.testing.assert_allclose(arrays["common_shift_xy"], -np.array(shift), atol=0.15)       # displacement to APPLY undoes the shift
    np.testing.assert_array_equal(arrays["shifts_xy"], np.broadcast_to(arrays["common_shift_xy"], (len(images), 2)))
    corrected = apply_image_shifts(images, arrays["common_shift_xy"]).astype(np.float32)
    assert np.linalg.norm(estimate_dataset_shifts(corrected, bank, ids)[0]["common_shift_xy"]) < 0.2


def test_plain_mean_without_ctf():
    images, _, _ = particles((4.0, -3.0), ctf=False)
    arrays, _ = estimate_dataset_shifts(images)
    np.testing.assert_allclose(arrays["common_shift_xy"], (-4.0, 3.0), atol=0.15)


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.3, -0.2)])
def test_small_shift_is_not_applied(shift):
    images, bank, ids = particles(shift)
    arrays, report = estimate_dataset_shifts(images, bank, ids)
    assert not report["applied"] and report["not_applied_reason"] == "below_min_shift"
    np.testing.assert_array_equal(arrays["shifts_xy"], 0)
    assert np.linalg.norm(report["estimate_xy_pixels"]) < 0.5


def test_estimate_beyond_range_is_not_applied():
    images, bank, ids = particles((7.0, 0.0))
    arrays, report = estimate_dataset_shifts(images, bank, ids, config=CenteringConfig(max_shift_fraction=0.05))
    assert report["not_applied_reason"] == "beyond_unambiguous_range"
    np.testing.assert_array_equal(arrays["shifts_xy"], 0)


def test_flat_and_tiny_inputs_give_zero():
    assert estimate_dataset_shifts(np.ones((200, SIDE, SIDE), dtype=np.float32))[1]["not_applied_reason"] == "flat_images"
    assert estimate_dataset_shifts(particles((5.0, 5.0), n=32, ctf=False)[0])[1]["not_applied_reason"] == "too_few_images"
