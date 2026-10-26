"""Synthetic cryo-EM data generation for moment-based reconstruction.

The module provides MRC I/O, von Mises--Fisher viewing distributions, rotation
sampling, and native-grid image formation with CTF, shift, contrast, noise, and
Fourier downsampling.
"""

import numpy as np

from .moments import (
    slice_images, real_images, centered_fft2, centered_ifft2, fourier_crop_images,
    embed_support, _progress, _n_batches,
)
from .optimization import apply_fourier_image_shifts


# --------------------------------------------------------------------------- #
#  Volume I/O
# --------------------------------------------------------------------------- #
def load_volume(path):
    """Read a 3-D MRC map and its isotropic voxel size in Angstrom."""
    import mrcfile
    with mrcfile.open(path, permissive=True) as m:
        vol = np.array(m.data, dtype=np.float64)
        axis_mapping = (
            int(m.header.mapc), int(m.header.mapr), int(m.header.maps))
        voxel_xyz = np.array([
            float(m.voxel_size.x),
            float(m.voxel_size.y),
            float(m.voxel_size.z),
        ])
    if vol.ndim != 3:
        raise ValueError(f"MRC volume must be three-dimensional, got {vol.shape}")
    if sorted(axis_mapping) != [1, 2, 3]:
        raise ValueError(
            f"MRC header for {path} has invalid mapc/mapr/maps axis mapping: "
            f"{axis_mapping}")
    # Convert MRC (section, row, column) storage to canonical (z, y, x) order.
    stored_physical_axes = (axis_mapping[2], axis_mapping[1], axis_mapping[0])
    permutation = tuple(stored_physical_axes.index(axis) for axis in (3, 2, 1))
    if permutation != (0, 1, 2):
        vol = np.ascontiguousarray(np.transpose(vol, permutation))
    if not np.all(np.isfinite(vol)):
        raise ValueError(f"MRC volume {path} contains non-finite density values")
    if not np.all(np.isfinite(voxel_xyz)) or np.any(voxel_xyz <= 0):
        raise ValueError(
            f"MRC header for {path} has no valid positive voxel size: "
            f"{voxel_xyz.tolist()}")
    if not np.allclose(voxel_xyz, voxel_xyz[0], rtol=1e-5, atol=1e-7):
        raise ValueError(
            "the simulator requires isotropic MRC voxels, but the header for "
            f"{path} gives {voxel_xyz.tolist()} Angstrom")
    return vol, float(np.mean(voxel_xyz))


def save_mrc(vol, path, voxel_size):
    """Write a real volume to an MRC file (float32, isotropic voxel size)."""
    import mrcfile
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(vol, dtype=np.float32))
        m.voxel_size = voxel_size


# --------------------------------------------------------------------------- #
#  von Mises-Fisher on S^2
# --------------------------------------------------------------------------- #
def _tangent_basis(mu):
    """Orthonormal pair spanning the tangent plane at ``mu``."""
    a = np.array([1.0, 0.0, 0.0]) if abs(mu[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = a - mu * (a @ mu)
    b1 /= np.linalg.norm(b1)
    return b1, np.cross(mu, b1)


def sample_vmf3(mu, kappa, n, rng):
    """Sample ``n`` unit vectors from a von Mises--Fisher law on ``S^2``.

    ``kappa=0`` is uniform; increasing ``kappa`` concentrates samples near
    ``mu``. The sampler uses the exact Ulrich--Wood construction.
    """
    mu = np.asarray(mu, float)
    mu = mu / np.linalg.norm(mu)
    if kappa < 1e-8:
        v = rng.standard_normal((n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    u = rng.uniform(size=n)
    w = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa   # cos angle to mu
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    b1, b2 = _tangent_basis(mu)
    return (w[:, None] * mu[None, :]
            + s[:, None] * (np.cos(phi)[:, None] * b1[None, :]
                            + np.sin(phi)[:, None] * b2[None, :]))


def sample_vmf_mixture(n, mus, kappas, weights, rng):
    """Sample ``n`` directions from a mixture of vMF components (shuffled)."""
    counts = rng.multinomial(n, weights)
    dirs = [sample_vmf3(mus[k], kappas[k], c, rng)
            for k, c in enumerate(counts) if c > 0]
    dirs = np.concatenate(dirs, axis=0)
    rng.shuffle(dirs)
    return dirs


def random_vmf_mixture(K, kappa, rng):
    """Return a ``K``-component vMF mixture with random means and equal weights."""
    mus = rng.standard_normal((K, 3))
    mus /= np.linalg.norm(mus, axis=1, keepdims=True)
    return mus, np.full(K, float(kappa)), np.full(K, 1.0 / K)


# --------------------------------------------------------------------------- #
#  Directions -> rotations
# --------------------------------------------------------------------------- #
def _batch_Rz(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1.0
    return R


def _batch_Ry(b):
    c, s = np.cos(b), np.sin(b)
    R = np.zeros((len(b), 3, 3))
    R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 1, 1] = 1.0; R[:, 2, 0] = -s; R[:, 2, 2] = c
    return R


def directions_to_rotations(dirs, rng):
    """Convert viewing directions to ZYZ rotations with uniform in-plane angle."""
    beta = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    alpha = np.arctan2(dirs[:, 1], dirs[:, 0])
    gamma = rng.uniform(0.0, 2.0 * np.pi, dirs.shape[0])
    return _batch_Rz(alpha) @ _batch_Ry(beta) @ _batch_Rz(gamma)


# --------------------------------------------------------------------------- #
#  Native-detector image formation
# --------------------------------------------------------------------------- #
def simulate_noisy_downsampled_images(vol_base, rots, support, ds_res, base_res,
                                      sigma2_real, rng, ctf_bank_base=None,
                                      group_of=None, image_shifts=None,
                                      image_contrasts=None,
                                      chunk=4000, eps=1e-8, n_show=0,
                                      return_components=False,
                                      noise_psd_base=None,
                                      n_native_images=0):
    """Simulate native-grid images and Fourier-downsample them to ``ds_res``.

    CTF, shifts, positive image contrasts, and detector noise are applied at
    ``base_res``. Shifts are specified in ``ds_res`` pixels. Noise is either
    real white noise with variance ``sigma2_real`` or colored noise with
    centered Fourier power ``noise_psd_base``; the two specifications are
    mutually exclusive.

    Returns support-restricted Fourier images and optional preview images.
    With ``return_components=True``, also returns clean projections and
    noiseless corrupted projections on the working support. If
    ``n_native_images`` is positive, the final return value is that many
    complete observed native-grid real images for empirical noise estimation.
    """
    n = len(rots)
    d = len(support)
    if int(n_native_images) != n_native_images or not 0 <= n_native_images <= n:
        raise ValueError("n_native_images must be an integer between zero and n")
    n_native_images = int(n_native_images)
    if not np.isfinite(sigma2_real) or sigma2_real < 0:
        raise ValueError("sigma2_real must be finite and nonnegative")
    colored_sqrt_psd = None
    if noise_psd_base is not None:
        noise_psd_base = np.asarray(noise_psd_base, dtype=np.float64)
        if (noise_psd_base.shape != (base_res, base_res)
                or not np.all(np.isfinite(noise_psd_base))
                or np.any(noise_psd_base < 0)):
            raise ValueError(
                "noise_psd_base must be a finite nonnegative base_res square")
        if sigma2_real > 0:
            raise ValueError(
                "sigma2_real and noise_psd_base cannot both specify noise")
        neg = (2 * (base_res // 2) - np.arange(base_res)) % base_res
        if not np.allclose(noise_psd_base,
                           noise_psd_base[np.ix_(neg, neg)],
                           rtol=1e-12, atol=1e-14):
            raise ValueError(
                "noise_psd_base must be symmetric under frequency negation")
        colored_sqrt_psd = np.sqrt(noise_psd_base)
    has_colored_noise = (colored_sqrt_psd is not None
                         and np.any(colored_sqrt_psd > 0))
    has_noise = sigma2_real > 0 or has_colored_noise
    if image_contrasts is not None:
        image_contrasts = np.asarray(image_contrasts, dtype=float)
        if (image_contrasts.shape != (n,)
                or not np.all(np.isfinite(image_contrasts))
                or np.any(image_contrasts <= 0)):
            raise ValueError(
                "image_contrasts must contain one finite positive value per image")
    Iflat = np.zeros((n, d), dtype=np.complex128)
    components = None
    if return_components:
        components = {"clean": np.empty((n, d), dtype=np.complex128),
                      "ctf_noiseless": np.empty((n, d), dtype=np.complex128)}
    native_images = (np.empty((n_native_images, base_res, base_res), dtype=np.float32)
                     if n_native_images else None)
    show = None
    starts = range(0, n, chunk)
    desc = "simulating noisy images" if has_noise else "simulating images"
    for start in _progress(starts, _n_batches(n, chunk), desc):
        sl = slice(start, min(start + chunk, n))
        imgs = slice_images(vol_base, rots[sl], eps=eps)
        if return_components:
            components["clean"][sl] = fourier_crop_images(
                imgs, ds_res).reshape(sl.stop - sl.start, -1, order="F")[:, support]
        if ctf_bank_base is not None:
            imgs = imgs * ctf_bank_base[group_of[sl]]
        if image_shifts is not None:
            apply_fourier_image_shifts(
                imgs, image_shifts[sl] * (base_res / ds_res), base_res)
        if image_contrasts is not None:
            imgs *= image_contrasts[sl, None, None]
        if return_components:
            components["ctf_noiseless"][sl] = fourier_crop_images(
                imgs, ds_res).reshape(sl.stop - sl.start, -1, order="F")[:, support]
        real_imgs = real_images(imgs)
        if has_colored_noise:
            # This normalization gives unit expected power per Fourier sample.
            white = rng.normal(
                0.0, 1.0 / base_res, size=real_imgs.shape)
            noise_fourier = centered_fft2(white.astype(np.complex128))
            noise_fourier *= colored_sqrt_psd[None, :, :]
            noisy_fourier = centered_fft2(real_imgs.astype(np.complex128))
            noisy_fourier += noise_fourier
        else:
            if sigma2_real > 0:
                real_imgs = real_imgs + rng.normal(
                    0.0, np.sqrt(sigma2_real), size=real_imgs.shape)
            noisy_fourier = centered_fft2(real_imgs.astype(np.complex128))
        if start < n_native_images:
            stop = min(sl.stop, n_native_images)
            count = stop - start
            native_images[start:stop] = np.real(
                centered_ifft2(noisy_fourier[:count])).astype(np.float32)
        noisy_ds = fourier_crop_images(noisy_fourier, ds_res)
        Iflat[sl] = noisy_ds.reshape(sl.stop - sl.start, -1, order="F")[:, support]
        if start == 0 and n_show:
            show = {"raw": noisy_fourier[:n_show].copy(),
                    "ds": embed_support(Iflat[:n_show], ds_res, support)}
    result = (Iflat, show)
    if return_components:
        result += (components,)
    if n_native_images:
        result += (native_images,)
    return result
