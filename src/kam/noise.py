"""Stationary detector-noise simulation and estimation.

A radial PSD can be supplied either as a RECOVAR-style shell table or as a
continuous function of radial frequency. Simulation is implemented in
:mod:`simulation`.
"""

import numpy as np

from .moments import slice_images, real_images
from .optimization import apply_fourier_image_shifts


def _symmetrize_centered_psd(psd):
    """Enforce the Hermitian-pair symmetry of a centered square PSD."""
    psd = np.asarray(psd, dtype=np.float64)
    if psd.ndim != 2 or psd.shape[0] != psd.shape[1]:
        raise ValueError("psd must be a square two-dimensional array")
    res = psd.shape[0]
    neg = (2 * (res // 2) - np.arange(res)) % res
    return 0.5 * (psd + psd[np.ix_(neg, neg)])


def estimate_noise_psd_legacy(images, pixel_size, output_res=None,
                              batch_size=32, bg_radius=None, max_d=None):
    """Estimate a radial PSD from observed images with ASPIRE's legacy method.

    ``images`` are complete real-space particle images, not Fourier samples
    restricted to the reconstruction disk. The estimator uses the region
    outside a centered disk as noise-only background. Its filter values have
    real-space covariance units; multiplication by ``L**2`` converts them to
    the raw, unnormalized-FFT power ``E[|N_hat(k)|**2]`` used by the moment
    debiasing formulas in this repository.

    If ``output_res`` is smaller than the input side length, the estimate is
    Fourier-cropped with the same normalization as the simulated images.
    """
    if np.iscomplexobj(images):
        raise ValueError("LegacyNoiseEstimator requires real-space images")
    images = np.asarray(images)
    if (images.ndim != 3 or images.shape[0] < 1
            or images.shape[1] != images.shape[2]):
        raise ValueError("images must have shape (n_images, L, L)")
    if not np.all(np.isfinite(images)):
        raise ValueError("images contain non-finite values")
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be finite and positive")
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    input_res = int(images.shape[-1])
    if output_res is None:
        output_res = input_res
    if int(output_res) != output_res or not 1 <= output_res <= input_res:
        raise ValueError("output_res must be an integer between 1 and image size")
    output_res = int(output_res)

    # Imported lazily so simulation without empirical noise estimation does not
    # require ASPIRE at module-import time.
    from aspire.noise import LegacyNoiseEstimator
    from aspire.source import ArrayImageSource

    source = ArrayImageSource(
        np.asarray(images, dtype=np.float32), pixel_size=float(pixel_size))
    estimator = LegacyNoiseEstimator(
        source, bg_radius=bg_radius, max_d=max_d,
        batch_size=int(batch_size), normalize_psd=True)
    aspire_psd = np.asarray(
        estimator.filter.evaluate_grid(input_res, dtype=np.float64))
    psd = _symmetrize_centered_psd(input_res ** 2 * aspire_psd)
    psd = np.maximum(psd, 0.0)

    if output_res != input_res:
        start = (input_res // 2) - (output_res // 2)
        psd = psd[start:start + output_res, start:start + output_res].copy()
        psd *= (output_res / input_res) ** 4
        psd = np.maximum(_symmetrize_centered_psd(psd), 0.0)

    if not np.all(np.isfinite(psd)) or not np.any(psd > 0):
        raise RuntimeError("LegacyNoiseEstimator returned an invalid PSD")
    return psd


def background_subtracted_noise_covariance(native_psd, background_weights,
                                           ds_res, support):
    """Propagate stationary noise through background-mean subtraction.

    ``native_psd`` is the centered, unnormalized native Fourier power
    ``E[|N_hat(k)|**2]``. ``background_weights`` is a real, nonnegative native
    image summing to one; each image is transformed as ``image - sum(w*image)``
    before Fourier cropping. ``support`` contains unique Fortran-order indices
    on the ``ds_res`` grid, excluding its unpaired even-grid Nyquist edges.

    Returns the complex covariance ``E[N_hat[i] * conj(N_hat[j])]`` on that
    support. Subtraction changes only DC, but correlates it with other Fourier
    coefficients. Its variance includes *all* native frequencies, including
    those outside the cropped image. With ``W = FFT(w)`` and
    ``a = (ds_res / native_res)**2``, the changed entries are
    ``C[0,k] = -a**2 * W[-k] * native_psd[k]`` and
    ``C[0,0] = a**2 * sum_{k != 0} |W[k]|**2 * native_psd[k]``.
    """
    if np.iscomplexobj(native_psd):
        raise ValueError("native_psd must be real")
    psd = np.asarray(native_psd, dtype=np.float64)
    if (psd.ndim != 2 or psd.shape[0] != psd.shape[1]
            or psd.shape[0] < 1):
        raise ValueError("native_psd must be a nonempty square array")
    if not np.all(np.isfinite(psd)) or np.any(psd < 0):
        raise ValueError("native_psd must be finite and nonnegative")
    native_res = psd.shape[0]
    neg = (2 * (native_res // 2) - np.arange(native_res)) % native_res
    if not np.allclose(psd, psd[np.ix_(neg, neg)], rtol=1e-10, atol=1e-12):
        raise ValueError("native_psd must have real-image Hermitian-pair symmetry")
    if np.iscomplexobj(background_weights):
        raise ValueError("background_weights must be real")
    weights = np.asarray(background_weights, dtype=np.float64)
    if weights.shape != psd.shape:
        raise ValueError("background_weights must have the native PSD shape")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("background_weights must be finite and nonnegative")
    if not np.isclose(np.sum(weights), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("background_weights must sum to one")
    if int(ds_res) != ds_res or not 1 <= ds_res <= native_res:
        raise ValueError("ds_res must be an integer between 1 and native size")
    ds_res = int(ds_res)
    support = np.asarray(support)
    if (support.ndim != 1 or support.size < 1
            or not np.issubdtype(support.dtype, np.integer)):
        raise ValueError("support must be a nonempty one-dimensional integer array")
    if (np.any(support < 0) or np.any(support >= ds_res ** 2)
            or np.unique(support).size != support.size):
        raise ValueError("support must contain unique indices inside the output grid")
    rows, cols = np.unravel_index(support, (ds_res, ds_res), order="F")
    if ds_res % 2 == 0 and (np.any(rows == 0) or np.any(cols == 0)):
        raise ValueError("support must exclude the unpaired even-grid Nyquist edges")

    start = native_res // 2 - ds_res // 2
    psd_selected = psd[rows + start, cols + start]
    power_scale = (ds_res / native_res) ** 4
    covariance = np.diag(power_scale * psd_selected).astype(np.complex128)
    dc_positions = np.flatnonzero(
        (rows == ds_res // 2) & (cols == ds_res // 2))
    if dc_positions.size:
        weight_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(weights)))
        dc = int(dc_positions[0])
        # W(-k) = conj(W(k)) because the averaging weights are real.
        cross = (-power_scale * psd_selected
                 * np.conj(weight_fft[rows + start, cols + start]))
        covariance[dc, :] = cross
        covariance[:, dc] = np.conj(cross)
        weighted_power = np.abs(weight_fft) ** 2 * psd
        weighted_power[native_res // 2, native_res // 2] = 0.0
        covariance[dc, dc] = power_scale * np.sum(weighted_power)
    return covariance


def radial_noise_psd(radial_profile, res):
    """Evaluate a radial Fourier-noise model on a centered square grid.

    ``radial_profile`` is either a one-dimensional table or a NumPy-vectorized
    callable. A table follows the RECOVAR convention: its index is the nearest
    integer Fourier shell, and frequencies beyond the table use its final
    value. A callable is evaluated at the continuous shell radius
    ``sqrt(kx**2 + ky**2)`` without rounding.

    The result may be an absolute PSD or a relative shape to be normalized by
    :func:`estimate_colored_noise_psd`.
    """
    if int(res) != res or res < 1:
        raise ValueError("res must be a positive integer")
    res = int(res)
    coords = np.arange(res, dtype=np.float64) - res // 2
    kx, ky = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(kx * kx + ky * ky)
    if callable(radial_profile):
        try:
            values = radial_profile(radius)
        except Exception as exc:
            raise ValueError(
                "radial_profile must accept a NumPy array of shell radii") from exc
        if np.iscomplexobj(values):
            raise ValueError("radial_profile must return real values")
        values = np.asarray(values, dtype=np.float64)
        try:
            psd = np.array(np.broadcast_to(values, (res, res)), copy=True)
        except ValueError as exc:
            raise ValueError(
                "radial_profile must return a scalar or an array broadcastable "
                f"to ({res}, {res})") from exc
    else:
        if np.iscomplexobj(radial_profile):
            raise ValueError("radial_profile table must be real")
        table = np.asarray(radial_profile, dtype=np.float64)
        if table.ndim != 1 or table.size == 0:
            raise ValueError("radial_profile table must be a nonempty 1-D array")
        shell = np.rint(radius).astype(np.int64)
        psd = table[np.minimum(shell, table.size - 1)]
    if not np.all(np.isfinite(psd)) or np.any(psd < 0):
        raise ValueError("the radial noise PSD must be finite and nonnegative")
    return psd


def _probe_signal_energy(vol_base, rots, ds_res, base_res,
                         ctf_bank_base=None, group_of=None,
                         image_shifts=None, n_probe=2000, eps=1e-8):
    """Return mean native-grid energy of clean unit-contrast probe images."""
    n_probe = min(int(n_probe), len(rots))
    if n_probe < 1:
        raise ValueError("at least one rotation is required")
    probe = slice_images(vol_base, rots[:n_probe], eps=eps)
    if ctf_bank_base is not None:
        if group_of is None:
            raise ValueError("group_of is required with ctf_bank_base")
        probe = probe * ctf_bank_base[np.asarray(group_of[:n_probe])]
    if image_shifts is not None:
        apply_fourier_image_shifts(
            probe, image_shifts[:n_probe] * (base_res / ds_res), base_res)
    return float(np.mean(np.sum(real_images(probe) ** 2, axis=(1, 2))))


def estimate_colored_noise_psd(vol_base, rots, ds_res, base_res, target_snr,
                               radial_profile, ctf_bank_base=None,
                               group_of=None, image_shifts=None,
                               n_probe=2000, eps=1e-8):
    """Scale a radial PSD model to a target native-image SNR.

    ``target_snr`` is the mean clean real-space image energy divided by the
    expected real-space noise energy. Returns ``(psd_base, psd_ds)`` as
    centered 2-D arrays containing ``E[|N_hat(k)|^2]`` before and after
    Fourier cropping. With the unnormalized FFT convention, Parseval gives
    ``E[||N||^2] = sum(psd_base) / base_res**2``. Fourier downsampling scales
    amplitudes by ``ds_res**2/base_res**2``, hence powers by its square.
    """
    if (not np.isfinite(target_snr)) or target_snr <= 0:
        raise ValueError("target_snr must be finite and positive")
    if not (1 <= int(ds_res) <= int(base_res)):
        raise ValueError("require 1 <= ds_res <= base_res")
    ds_res, base_res = int(ds_res), int(base_res)
    relative_psd = radial_noise_psd(radial_profile, base_res)
    relative_power = float(np.sum(relative_psd))
    if relative_power <= 0:
        raise ValueError("the radial noise model must contain positive total power")
    signal_energy = _probe_signal_energy(
        vol_base, rots, ds_res, base_res,
        ctf_bank_base=ctf_bank_base, group_of=group_of,
        image_shifts=image_shifts, n_probe=n_probe, eps=eps)
    scale = signal_energy * base_res ** 2 / (
        float(target_snr) * relative_power)
    psd_base = scale * relative_psd
    # Apply the Fourier-crop convention, including mixed-parity grids.
    start = (base_res // 2) - (ds_res // 2)
    psd_ds = psd_base[start:start + ds_res, start:start + ds_res].copy()
    psd_ds *= (ds_res / base_res) ** 4
    return psd_base, psd_ds
