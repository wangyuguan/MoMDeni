"""Alignment, reference matching, and moment preparation/reconstruction.

Merged from alignment.py (rigid volume alignment and FSC),
reference_matching.py (deterministic reference selection and fixed-weight
Fourier-image fitting), and mom_reconstruction.py (dataset-independent
preparation of complete disk-restricted M1--M3 from a denoised stack and
independent SphFB M2/M3 matching). Semantics of each part are unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import finufft
import mrcfile
import numpy as np
from scipy import ndimage, signal
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation
from skimage.registration import phase_cross_correlation

from . import utils
from .moments import (
    FBMomentBasis,
    UPSAMPFAC,
    _on_gpu,
    _progress,
    _slice_freq_grid,
    centered_coords_1d,
    centered_fftn,
    centered_ifftn,
    disk_support,
    fourier_crop,
    so3_sphere_grid,
    use_device,
    empirical_moments as form_empirical_moments,
)
from .optimization import SphFBParam, joint_objective_and_grad
from .simulation import load_volume, save_mrc
from .viewing import LowDegreeViewingDensity, fit_single_moment_low_degree


# ===========================================================================
# Volume alignment and FSC (was alignment.py)
# ===========================================================================

_FSC_EMPTY_SHELL = 1e-12  # shell energy, relative to the total, treated as empty
_ALIGNMENT_SEARCH_MIN_SIDE = 32
_ALIGNMENT_SEARCH_OVERSAMPLING = 2.5
_ALIGNMENT_SIMILARITY_RESOLUTION = 20.0
_ALIGNMENT_SCALE_BOUNDS = (0.95, 1.05)
_ALIGNMENT_SCALE_GRID_SIZE = 21
_ALIGNMENT_TRANSLATION_UPSAMPLE = 100
_ALIGNMENT_BOX_SIZE_RTOL = 0.05


# Rigid transforms
def reflect_volume(vol, axis=0):
    """Reflect ``vol`` about the grid center along ``axis``.

    The index map ``a -> 2*(L//2) - a (mod L)`` preserves the origin. A direct
    array reversal shifts the origin by one voxel when ``L`` is even.
    """
    L = vol.shape[axis]
    idx = (2 * (L // 2) - np.arange(L)) % L
    return np.take(vol, idx, axis=axis)


def rotate_volume(vol, R, eps=1e-8):
    """Return ``V(R^T x)`` sampled on the input grid.

    The Fourier relation is ``Vhat_rot(k) = Vhat(R^T k)``. A type-2 NUFFT
    evaluates the original transform at these rotated grid points.
    """
    L = vol.shape[0]
    c = centered_coords_1d(L)
    kx, ky, kz = np.meshgrid(c, c, c, indexing="ij")
    K = np.stack([kx.ravel(), ky.ravel(), kz.ravel()])          # (3, L^3)
    # The NUFFT evaluates Vhat(-xi); negate coordinates to sample at R^T k.
    P = (-(np.asarray(R).T @ K)) * (2.0 * np.pi / L)
    volc = np.ascontiguousarray(vol, dtype=np.complex128)
    # Thread initialization dominates small transforms.
    nth = 1 if L <= 64 else 0
    f = finufft.nufft3d2(np.ascontiguousarray(P[0]), np.ascontiguousarray(P[1]),
                         np.ascontiguousarray(P[2]), volc, isign=1, eps=eps,
                         upsampfac=UPSAMPFAC, nthreads=nth)
    return np.real(centered_ifftn(f.reshape(L, L, L)))


def rotate_volume_spline(vol, R, order=3):
    """Rotate ``vol`` by spline resampling.

    This approximation is used during alignment search; the selected rotation is
    subsequently applied with :func:`rotate_volume`.
    """
    L = vol.shape[0]
    ctr = (L // 2) * np.ones(3)
    A = np.asarray(R, dtype=float).T
    return ndimage.affine_transform(vol, A, offset=ctr - A @ ctr, order=order,
                                    mode="constant", cval=0.0)


def similarity_transform_volume(
        vol, R, magnification=1.0, translation=None, order=3):
    """Apply rotation, isotropic magnification, and translation in one pass.

    ``translation`` is expressed in output-array voxels.  A positive component
    moves the map contents toward increasing indices on that array axis.
    ``magnification > 1`` makes features larger.  The returned array samples

    ``vol[c + R.T @ (y - c - translation) / magnification]``.

    Reflection, when requested by the caller, is an exact index permutation
    performed before this interpolating transform.
    """
    volume = np.asarray(vol)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError("similarity transform requires a cubic 3-D volume")
    magnification = float(magnification)
    if not np.isfinite(magnification) or magnification <= 0.0:
        raise ValueError("magnification must be finite and positive")
    shift = (np.zeros(3, dtype=float) if translation is None
             else np.asarray(translation, dtype=float))
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError("translation must contain three finite values")
    rotation = np.asarray(R, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("R must be a finite 3-by-3 matrix")
    center = (volume.shape[0] // 2) * np.ones(3)
    matrix = rotation.T / magnification
    offset = center - matrix @ (center + shift)
    return ndimage.affine_transform(
        volume, matrix, offset=offset, order=order,
        mode="constant", cval=0.0, prefilter=(order > 1),
    )


# Alignment search
def _ncc(a, b):
    """Return normalized cross-correlation, invariant to overall scale.

    Ufunc reductions avoid BLAS thread overhead for the small arrays used in the
    orientation scan.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.sum(a * b) / (na * nb))


def _orientation_grid(n):
    """Construct approximately ``n^3`` rotations covering ``SO(3)``.

    Sampling ``cos(beta)`` avoids excessive concentration near the poles.
    """
    a = 2.0 * np.pi * np.arange(n) / n
    b = np.arccos(np.linspace(1.0, -1.0, n))
    g = 2.0 * np.pi * np.arange(n) / n
    A, B, G = np.meshgrid(a, b, g, indexing="ij")
    return Rotation.from_euler("ZYZ", np.stack([A.ravel(), B.ravel(), G.ravel()],
                                               axis=1)).as_matrix()


_REFINE_MAXITER = 400


def _refine(ref, vol, R0):
    """Locally maximize correlation over ``SO(3)`` from ``R0``.

    The progress bar follows Nelder--Mead callbacks and may stop before its maximum
    when the simplex converges.
    """
    r0 = Rotation.from_matrix(R0).as_rotvec()
    try:
        from tqdm import tqdm
        bar = tqdm(total=_REFINE_MAXITER, desc="align refine", unit="it",
                   leave=False, file=sys.stdout)
    except ImportError:
        bar = None

    def neg(v):
        return -_ncc(ref, rotate_volume_spline(vol, Rotation.from_rotvec(v).as_matrix()))

    res = minimize(neg, r0, method="Nelder-Mead",
                   callback=(lambda xk: bar.update(1)) if bar else None,
                   options=dict(xatol=1e-4, fatol=1e-8, maxiter=_REFINE_MAXITER))
    if bar is not None:
        bar.close()
    return Rotation.from_rotvec(res.x).as_matrix(), -res.fun


def align_volumes(ref, vol, n_grid=12, eps=1e-6, refine=True, verbose=True):
    """Best ``O(3)`` alignment of ``vol`` onto ``ref``.

    Each handedness is searched on a coarse orientation grid and optionally refined.
    Candidates are scored by normalized cross-correlation. Spline interpolation is
    used for the search, and the selected transform is applied once in Fourier space.

    Returns ``(aligned, R, reflected, corr)``.
    """
    grid = _orientation_grid(n_grid)
    best = None
    for reflected in (False, True):
        v = reflect_volume(vol) if reflected else vol
        tag = "reflected" if reflected else "direct"
        scores = [_ncc(ref, rotate_volume_spline(v, R))
                  for R in _progress(grid, len(grid), f"align scan ({tag})",
                                     unit="rot")]
        k = int(np.argmax(scores))
        R, c = grid[k], scores[k]
        if refine:
            R, c = _refine(ref, v, R)
        if verbose:
            print(f"    {'reflected' if reflected else 'direct   '}: "
                  f"corr = {c:.4f}", flush=True)
        if best is None or c > best[3]:
            best = (v, R, reflected, c)

    v, R, reflected, _ = best
    aligned = rotate_volume(v, R, eps=eps)
    return aligned, R, reflected, _ncc(ref, aligned)


# Fourier shell correlation
def fsc(v1, v2):
    """Fourier shell correlation between two real volumes on the same grid.

    Returns ``(k, curve)``, where ``k`` is the integer shell radius in cycles per
    box and ``k=0`` is the DC shell. Resolution is ``L * voxel_size / k``. A
    shell holding only roundoff in either volume, such as the shells outside a
    band-limited reference, has no defined correlation and is NaN.
    """
    L = v1.shape[0]
    F1, F2 = centered_fftn(v1), centered_fftn(v2)
    c = centered_coords_1d(L)
    cx, cy, cz = np.meshgrid(c, c, c, indexing="ij")
    rad = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
    shell = np.round(rad).astype(int)
    kmax = L // 2
    num = np.zeros(kmax + 1)
    d1 = np.zeros(kmax + 1)
    d2 = np.zeros(kmax + 1)
    keep = shell <= kmax
    np.add.at(num, shell[keep], np.real(F1[keep] * np.conj(F2[keep])))
    np.add.at(d1, shell[keep], np.abs(F1[keep]) ** 2)
    np.add.at(d2, shell[keep], np.abs(F2[keep]) ** 2)
    denom = np.sqrt(d1 * d2)
    defined = ((d1 > _FSC_EMPTY_SHELL * d1.sum())
               & (d2 > _FSC_EMPTY_SHELL * d2.sum()))
    curve = np.divide(num, denom, out=np.full_like(num, np.nan), where=defined)
    return np.arange(kmax + 1), curve


def fsc_resolution(k, curve, voxel_size, L, cutoff=0.5):
    """Return the resolution in Angstrom at the first FSC cutoff crossing.

    Return an exact cutoff sample at positive frequency, or interpolate only
    between adjacent finite samples strictly above and below the cutoff.
    Shells after the first non-finite shell beyond DC are ignored. Returns
    ``None`` when no such crossing exists, including an entirely below-cutoff
    curve; below-cutoff samples must not produce an extrapolated crossing.
    """
    curve = np.asarray(curve, dtype=float)
    undefined = np.flatnonzero(~np.isfinite(curve[1:])) + 1
    n_defined = int(undefined[0]) if undefined.size else len(curve)
    for i in range(1, n_defined):
        k0, k1 = k[i - 1], k[i]
        c0, c1 = curve[i - 1], curve[i]
        if c1 == cutoff and np.isfinite(k1) and k1 > 0:
            return float(L * voxel_size / k1)
        if (np.isfinite(c0) and c0 > cutoff > c1
                and np.isfinite(k0) and np.isfinite(k1)):
            kc = k0 + (c0 - cutoff) / (c0 - c1) * (k1 - k0)
            return float(L * voxel_size / kc) if kc > 0 else None
    return None


# ===========================================================================
# Reference selection and matching (was reference_matching.py)
# ===========================================================================

def _frequencies(ds_res, support):
    """Array-axis frequencies in cycles/pixel, in supported Fortran order."""
    if int(ds_res) != ds_res or ds_res < 2:
        raise ValueError("ds_res must be an integer of at least two")
    ds_res = int(ds_res)
    support = np.asarray(support)
    if (support.ndim != 1 or support.size == 0
            or not np.issubdtype(support.dtype, np.integer)
            or support.min() < 0 or support.max() >= ds_res ** 2
            or len(np.unique(support)) != len(support)):
        raise ValueError("support must contain distinct integer image indices")
    c = (np.arange(ds_res) - ds_res // 2) / ds_res
    down, right = np.meshgrid(c, c, indexing="ij")
    return np.column_stack((down.ravel(order="F")[support],
                            right.ravel(order="F")[support]))


def _selection_radii(n_pixels, ds_res, support):
    if support is not None and ds_res is None:
        raise ValueError("ds_res is required with an explicit support")
    if ds_res is None:
        # The experiment uses radius L/2-1. Its support width determines the
        # sampled radii even when adjacent side lengths share a disk count.
        limit = int(np.ceil(2 * np.sqrt(n_pixels / np.pi))) + 5
        for side in range(2, limit):
            candidate = disk_support(side, side / 2 - 1)
            if len(candidate) == n_pixels:
                ds_res, support = side, candidate
                break
        else:
            side = int(np.sqrt(n_pixels))
            if side ** 2 != n_pixels:
                raise ValueError("cannot infer image support; supply ds_res and support")
            ds_res, support = side, np.arange(n_pixels)
    elif support is None:
        support = (np.arange(n_pixels) if n_pixels == ds_res ** 2
                   else disk_support(ds_res, ds_res / 2 - 1))
    if len(support) != n_pixels:
        raise ValueError("image width does not match ds_res/support")
    return np.linalg.norm(_frequencies(ds_res, support), axis=1) * ds_res


def select_reference_images(Iflat, filter_indices, n_refs=16, seed=0, *,
                            ds_res=None, support=None):
    """Select representative, diverse rows without using poses or truth.

    Exclude norms outside the 10th--90th percentiles, then use a seeded pool
    of at most 4096 rows. Start nearest the median radial-power profile and
    farthest-sample normalized profiles, preferring unused CTF groups while
    any remain. Radial shell powers are square-rooted after normalization
    (Hellinger features), reducing domination by bright images or DC power.

    The default infers the experiment's radius ``ds_res/2-1`` disk from the
    column count, with a full-square fallback. Pass ``ds_res`` and ``support``
    explicitly for other layouts or to remove this inference's ambiguity.
    Returns distinct integer row indices, in selection order.
    """
    images = np.asarray(Iflat)
    groups = np.asarray(filter_indices)
    if images.ndim != 2 or min(images.shape) == 0:
        raise ValueError("Iflat must be a nonempty (n_images, n_pixels) array")
    if groups.shape != (len(images),):
        raise ValueError("filter_indices must contain one group per image")
    if int(n_refs) != n_refs or not 1 <= n_refs <= min(len(images), 4096):
        raise ValueError("n_refs must be between one and min(n_images, 4096)")
    n_refs = int(n_refs)
    radii = _selection_radii(images.shape[1], ds_res, support)
    norms = np.empty(len(images))
    for start in range(0, len(images), 1024):
        chunk = images[start:start + 1024]
        norms[start:start + len(chunk)] = np.linalg.norm(chunk, axis=1)
    valid = np.isfinite(norms) & (norms > 0)
    if not np.any(valid):
        raise ValueError("no finite nonzero images are available")
    lo, hi = np.quantile(norms[valid], [.1, .9])
    eligible = np.flatnonzero(valid & (norms >= lo) & (norms <= hi))
    if len(eligible) < n_refs:
        raise ValueError("too few images remain after excluding norm extremes")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(eligible)
    # Reserve one random candidate per group before filling the bounded pool.
    _, first = np.unique(groups[shuffled], return_index=True)
    first = np.sort(first)
    take = min(4096, len(eligible))
    reserved = np.zeros(len(shuffled), dtype=bool)
    reserved[first] = True
    pool = np.concatenate((shuffled[first], shuffled[~reserved]))[:take]
    rng.shuffle(pool)
    shells = np.floor(radii + 1e-12).astype(int)
    power = np.abs(images[pool]) ** 2
    profiles = np.column_stack([
        power[:, shells == shell].sum(axis=1)
        for shell in np.unique(shells)
    ])
    profiles = np.sqrt(profiles / profiles.sum(axis=1, keepdims=True))
    center = np.median(profiles, axis=0)
    current = int(np.argmin(np.sum((profiles - center) ** 2, axis=1)))
    chosen, used_groups = [], set()
    distances = np.full(len(pool), np.inf)
    available = np.ones(len(pool), dtype=bool)
    for _ in range(n_refs):
        chosen.append(current)
        available[current] = False
        used_groups.add(groups[pool[current]].item())
        distances = np.minimum(distances,
                               np.sum((profiles - profiles[current]) ** 2, axis=1))
        fresh = available & ~np.isin(groups[pool], list(used_groups))
        candidates = fresh if np.any(fresh) else available
        current = int(np.argmax(np.where(candidates, distances, -np.inf)))
    return pool[np.asarray(chosen)]


def _targets(targets, n_pixels):
    targets = np.asarray(targets, dtype=np.complex128)
    if (targets.ndim != 2 or targets.shape[0] == 0
            or targets.shape[1] != n_pixels or not np.all(np.isfinite(targets))):
        raise ValueError("targets must be a finite nonempty (K, len(support)) array")
    norm_sq = np.sum(np.abs(targets) ** 2, axis=1)
    if not np.all(np.isfinite(norm_sq) & (norm_sq > np.finfo(float).tiny)):
        raise ValueError("every target must have a finite, nonzero norm")
    return targets, norm_sq


def _phases(shifts, frequencies):
    return np.exp(-2j * np.pi * (np.asarray(shifts) @ frequencies.T))


def reference_objective_and_grad(theta, ref_param, targets, shifts, amplitudes,
                                 ds_res, support):
    """Return normalized mean squared residual, real theta gradient, predictions.

    ``ref_param`` is precomputed at the K reference rotations and the same
    image support. Poses and nonnegative amplitudes are held fixed here:
    ``mean_k ||a_k phase(s_k) F_k(theta)-target_k||^2 / ||target_k||^2``.
    The K references have equal fixed weights, independent of the sphere PMF.
    """
    frequencies = _frequencies(ds_res, support)
    targets, norm_sq = _targets(targets, len(frequencies))
    shifts, amplitudes = np.asarray(shifts, float), np.asarray(amplitudes, float)
    if shifts.shape != (len(targets), 2) or not np.all(np.isfinite(shifts)):
        raise ValueError("shifts must be a finite (K, 2) array")
    if (amplitudes.shape != (len(targets),)
            or not np.all(np.isfinite(amplitudes) & (amplitudes >= 0))):
        raise ValueError("amplitudes must be a finite nonnegative (K,) array")
    images = np.asarray(ref_param.forward_images(theta, support=support))
    if images.shape != targets.shape:
        raise ValueError("ref_param must be precomputed at the K reference poses")
    factor = amplitudes[:, None] * _phases(shifts, frequencies)
    predicted = factor * images
    residual = predicted - targets
    weights = 1.0 / (len(targets) * norm_sq)
    loss = float(np.sum(weights * np.sum(np.abs(residual) ** 2, axis=1)))
    image_grad = 2 * weights[:, None] * np.conj(factor) * residual
    gradient = ref_param.adjoint_images(image_grad, support=support)
    return loss, np.asarray(gradient, dtype=float), predicted


def _fit_amplitude(image, target, norm_sq):
    power = float(np.vdot(image, image).real)
    amplitude = (max(0.0, float(np.vdot(image, target).real) / power)
                 if power > np.finfo(float).tiny else 0.0)
    # Strictly positive boundary approximation to the nonnegative LS solution.
    amplitude = max(np.finfo(float).tiny, amplitude)
    residual = amplitude * image - target
    return float(np.vdot(residual, residual).real / norm_sq), amplitude


def match_reference_images(theta, param, grid, targets, ds_res, support,
                           shift_bound=2.0, previous=None, *, refine=True,
                           refine_maxiter=80, rotation_batch=256, use_gpu=None):
    """Match each reference over all grid rotations/shifts, then refine on CPU.

    The coarse shift lattice is the integers from ``-ceil(shift_bound)`` to
    ``ceil(shift_bound)``, clipped to the bound (including fractional endpoints).
    Amplitudes are positive least-squares fits, with a near-zero boundary.
    Matrix products are batched, and use CUDA when the active project device
    is GPU; ``use_gpu`` optionally overrides that choice.

    Continuous refinement uses CPU FINUFFT with ``eps=1e-8`` and the exact
    ``slice_images`` convention, plus finite-difference L-BFGS-B with
    ``R = Rotation.from_rotvec(delta) @ R_start``. Only supported frequencies
    are evaluated, on one CPU thread to avoid tiny-transform thread overhead.
    Each rotation-vector component is bounded to +/- pi/4 around its start.
    Previous poses, if supplied, are rescored against the current volume,
    with amplitudes re-fitted. A refinement is accepted only if it improves
    its starting candidate. ``grid_indices`` retain the coarse seed index
    (or the previous seed, -1 if unknown), not an exact refined-grid lookup.

    ``param`` must use the working ``ds_res`` volume and image support.
    Returns rots, shifts, amplitudes, relative_errors (relative L2 norms),
    grid_indices, and coarse_relative_errors before previous/local candidates.
    """
    frequencies = _frequencies(ds_res, support)
    targets, norm_sq = _targets(targets, len(frequencies))
    if not np.isfinite(shift_bound) or shift_bound < 0:
        raise ValueError("shift_bound must be finite and nonnegative")
    if (int(rotation_batch) != rotation_batch or rotation_batch < 1
            or int(refine_maxiter) != refine_maxiter or refine_maxiter < 1):
        raise ValueError("rotation_batch and refine_maxiter must be positive integers")
    if getattr(param, "L", ds_res) != ds_res:
        raise ValueError("param.L must equal ds_res")
    rots = np.asarray(grid.rots, dtype=float)
    if (rots.ndim != 3 or rots.shape[1:] != (3, 3) or len(rots) == 0
            or not np.all(np.isfinite(rots))):
        raise ValueError("grid.rots must be a finite nonempty (M, 3, 3) array")
    images = np.asarray(param.forward_images(theta, rots=rots, support=support),
                        dtype=np.complex128)
    if images.shape != (len(rots), len(frequencies)) or not np.all(np.isfinite(images)):
        raise ValueError("param produced invalid grid images")
    extent = int(np.ceil(shift_bound))
    axis = np.unique(np.clip(np.arange(-extent, extent + 1), -shift_bound, shift_bound))
    shifts = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    shifts = shifts[np.argsort(np.sum(shifts ** 2, axis=1), kind="stable")]
    phases = _phases(shifts, frequencies)
    bank = (targets[None, :, :] * phases[:, None, :].conj()).reshape(-1, images.shape[1])
    enabled = _on_gpu() if use_gpu is None else bool(use_gpu)
    if enabled and not utils.gpu_available():
        raise RuntimeError("GPU reference matching requested but CUDA is unavailable")
    if enabled:
        torch, dev = utils._gpu_torch(), utils._gpu_dev()
        bank_gpu = torch.as_tensor(np.ascontiguousarray(bank.T), device=dev)

    count = len(targets)
    best_scores = np.full(count, -np.inf)
    best_grid, best_shift = np.zeros(count, int), np.zeros(count, int)
    for start in range(0, len(rots), int(rotation_batch)):
        block = images[start:start + int(rotation_batch)]
        if enabled:
            block_gpu = torch.as_tensor(np.ascontiguousarray(block), device=dev)
            dots = (block_gpu.conj() @ bank_gpu).real.cpu().numpy()
        else:
            dots = (block.conj() @ bank.T).real
        dots = np.maximum(dots.reshape(len(block), len(shifts), count), 0.0)
        power = np.sum(np.abs(block) ** 2, axis=1)
        scores = dots ** 2 / np.maximum(power[:, None, None], np.finfo(float).tiny)
        scores = scores.reshape(-1, count)
        winners = np.argmax(scores, axis=0)
        values = scores[winners, np.arange(count)]
        update = values > best_scores
        best_scores[update] = values[update]
        best_grid[update] = start + winners[update] // len(shifts)
        best_shift[update] = winners[update] % len(shifts)

    result_rots, result_shifts = rots[best_grid].copy(), shifts[best_shift].copy()
    errors, amplitudes = np.empty(count), np.empty(count)
    for k in range(count):
        errors[k], amplitudes[k] = _fit_amplitude(
            images[best_grid[k]] * phases[best_shift[k]], targets[k], norm_sq[k])
    coarse_errors = np.sqrt(errors.copy())
    if refine or previous is not None:
        volume = np.ascontiguousarray(param.volume(theta), dtype=np.complex128)
        kx, ky = _slice_freq_grid(ds_res, ds_res)
        slice_frequencies = np.vstack((kx[support], ky[support]))

        def evaluate(rotation, shift, k):
            points = 2 * np.pi * (rotation[:, :2] @ slice_frequencies)
            image = finufft.nufft3d2(
                *[np.ascontiguousarray(axis) for axis in points], volume,
                isign=1, eps=1e-8, upsampfac=UPSAMPFAC, nthreads=1)
            image = image * _phases(np.asarray(shift)[None], frequencies)[0]
            return _fit_amplitude(image, targets[k], norm_sq[k])

        if previous is not None:
            prev_rots = np.asarray(previous["rots"], float)
            prev_shifts = np.asarray(previous["shifts"], float)
            prev_indices = np.asarray(previous.get("grid_indices", np.full(count, -1)))
            if (prev_rots.shape != result_rots.shape or prev_shifts.shape != (count, 2)
                    or prev_indices.shape != (count,)
                    or not np.all(np.isfinite(prev_rots))
                    or not np.all(np.isfinite(prev_shifts))):
                raise ValueError("previous poses must match the reference count and be finite")
        for k in range(count):
            # Score continuous candidates with the same accurate CPU operator.
            errors[k], amplitudes[k] = evaluate(result_rots[k], result_shifts[k], k)
            coarse_errors[k] = np.sqrt(errors[k])
            if previous is not None and np.all(np.abs(prev_shifts[k]) <= shift_bound):
                error, amplitude = evaluate(prev_rots[k], prev_shifts[k], k)
                if error < errors[k]:
                    errors[k], amplitudes[k] = error, amplitude
                    result_rots[k], result_shifts[k] = prev_rots[k], prev_shifts[k]
                    best_grid[k] = prev_indices[k]
            if not refine:
                continue
            base = result_rots[k].copy()

            def objective(x):
                rotation = Rotation.from_rotvec(x[:3]).as_matrix() @ base
                return evaluate(rotation, x[3:], k)[0]

            x0 = np.r_[np.zeros(3), result_shifts[k]]
            fit = minimize(objective, x0, method="L-BFGS-B",
                           bounds=[(-np.pi / 4, np.pi / 4)] * 3
                                  + [(-shift_bound, shift_bound)] * 2,
                           options={"maxiter": int(refine_maxiter), "eps": 1e-4,
                                    "ftol": 1e-12, "gtol": 1e-7, "maxls": 30})
            candidate = Rotation.from_rotvec(fit.x[:3]).as_matrix() @ base
            error, amplitude = evaluate(candidate, fit.x[3:], k)
            if np.isfinite(error) and error < errors[k]:
                errors[k], amplitudes[k] = error, amplitude
                result_rots[k], result_shifts[k] = candidate, fit.x[3:]
    return {"rots": result_rots, "shifts": result_shifts,
            "amplitudes": amplitudes, "relative_errors": np.sqrt(errors),
            "grid_indices": best_grid, "coarse_relative_errors": coarse_errors}


# ===========================================================================
# Moment preparation and reconstruction (was mom_reconstruction.py)
# ===========================================================================

DEFAULT_DENOISE_RES = 64
DEFAULT_MOMENT_RES = 16
DEFAULT_N_IMAGES = 10000
DEFAULT_ELL_MAX = 5
DEFAULT_VIEWING_DEGREE = 2
DEFAULT_SPHERE_NODES = 200
DEFAULT_N_INIT = 1
DEFAULT_MAXITER = 10000
DEFAULT_MAX_M3_ENTRIES = 10_000_000
DEFAULT_ALIGNMENT_LOWPASS = 80.0
FB_PREPARATION_VERSION = "steerable_fb_denoised_moments_v1"

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
RECONSTRUCTION_IMPLEMENTATION_FILES = {
    "moment_reconstruction": Path(__file__),
    "viewing": PACKAGE_DIR / "viewing.py",
    "moments": PACKAGE_DIR / "moments.py",
    "optimization": PACKAGE_DIR / "optimization.py",
    "simulation": PACKAGE_DIR / "simulation.py",
    "utils": PACKAGE_DIR / "utils.py",
}


@dataclass(frozen=True)
class MomentPreparationConfig:
    """Configuration for :func:`prepare_moments`."""

    denoising_dir: Path
    selection_file: Path
    output_dir: Path
    expected_dataset: str | None = None
    n_images: int = DEFAULT_N_IMAGES
    denoise_res: int = DEFAULT_DENOISE_RES
    moment_res: int = DEFAULT_MOMENT_RES
    batch_size: int = 256
    device: str = "gpu"
    max_m3_entries: int = DEFAULT_MAX_M3_ENTRIES
    fb_ell_max: int = DEFAULT_ELL_MAX
    augment_reflections: bool = False

    def __post_init__(self):
        if not isinstance(self.augment_reflections, (bool, np.bool_)):
            raise ValueError("augment_reflections must be Boolean")
        for name in ("denoising_dir", "selection_file", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if (isinstance(self.fb_ell_max, (bool, np.bool_))
                or int(self.fb_ell_max) != self.fb_ell_max
                or self.fb_ell_max < 0):
            raise ValueError("fb_ell_max must be a nonnegative integer")
        _positive_integer(self.n_images, "n_images")
        _positive_integer(self.batch_size, "batch_size")
        _positive_integer(self.max_m3_entries, "max_m3_entries")
        _even_side(self.denoise_res, "denoise_res")
        _even_side(self.moment_res, "moment_res")
        if self.moment_res > self.denoise_res:
            raise ValueError("moment_res cannot exceed denoise_res")
        _device_name(self.device)
        if (self.expected_dataset is not None
                and (not isinstance(self.expected_dataset, str)
                     or not self.expected_dataset.strip())):
            raise ValueError("expected_dataset must be nonempty or None")


@dataclass(frozen=True)
class PreparedMomentsResult:
    """Paths and dimensions produced by :func:`prepare_moments`."""

    output_dir: Path
    moments_path: Path
    report_path: Path
    dataset: str
    n_images: int
    denoise_res: int
    moment_res: int
    pixel_size: float
    n_frequencies: int
    n_m2_entries: int
    n_m3_entries: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration for one independent M2-only or M3-only fit."""

    prepared_dir: Path
    output_dir: Path
    order: int
    ell_max: int = DEFAULT_ELL_MAX
    viewing_degree: int = DEFAULT_VIEWING_DEGREE
    sphere_nodes: int = DEFAULT_SPHERE_NODES
    sphere_degree: int | None = None
    n_init: int = DEFAULT_N_INIT
    maxiter: int = DEFAULT_MAXITER
    device: str = "gpu"
    print_every: int = 100
    seed: int = 0
    initial_fit: Path | None = None
    resume: bool = False

    def __post_init__(self):
        for name in ("prepared_dir", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.initial_fit is not None:
            object.__setattr__(self, "initial_fit", Path(self.initial_fit))
        if self.order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        if (isinstance(self.ell_max, (bool, np.bool_))
                or int(self.ell_max) != self.ell_max or self.ell_max < 0):
            raise ValueError("ell_max must be a nonnegative integer")
        if self.viewing_degree not in (0, 2, 4):
            raise ValueError("viewing_degree must be 0, 2, or 4")
        for name in ("sphere_nodes", "n_init", "maxiter"):
            _positive_integer(getattr(self, name), name)
        if self.sphere_degree is not None:
            # An M3 entry is a product of three Wigner factors of degree
            # <= ell_max times a degree-(viewing_degree) density, so the
            # quadrature is exact once its degree reaches 3*ell_max +
            # viewing_degree; the node count follows from the degree.
            _positive_integer(self.sphere_degree, "sphere_degree")
            minimum = 3 * self.ell_max + self.viewing_degree
            if self.sphere_degree < minimum:
                raise ValueError(
                    f"sphere_degree {self.sphere_degree} cannot integrate the "
                    f"degree-{minimum} M3 model of ell_max={self.ell_max}, "
                    f"viewing_degree={self.viewing_degree} exactly")
        if (isinstance(self.print_every, (bool, np.bool_))
                or int(self.print_every) != self.print_every
                or self.print_every < 0):
            raise ValueError("print_every must be a nonnegative integer")
        if (isinstance(self.seed, (bool, np.bool_))
                or int(self.seed) != self.seed):
            raise ValueError("seed must be an integer")
        if not isinstance(self.resume, (bool, np.bool_)):
            raise ValueError("resume must be Boolean")
        if self.order == 1 and self.initial_fit is not None:
            raise ValueError("M1 does not use initial_fit")
        if self.order == 3 and self.initial_fit is None:
            raise ValueError(
                "M3 requires initial_fit pointing to a compatible M2 fit "
                "directory")
        _device_name(self.device)


@dataclass(frozen=True)
class ReconstructionResult:
    """Summary of a completed independent moment fit."""

    output_dir: Path
    fit_path: Path
    reconstruction_path: Path
    summary_path: Path
    order: int
    best_init: int
    objective: float
    relative_moment_error: float
    n_volume_parameters: int
    n_view_parameters: int
    actual_sphere_nodes: int
    so3_atoms: int
    elapsed_seconds: float


@dataclass(frozen=True)
class PreparedMomentData:
    """Validated arrays and geometry loaded from ``moments.npz``."""

    directory: Path
    report: dict
    values: dict
    target: np.ndarray
    order: int
    side: int
    pixel_size: float
    n_images: int
    moment_basis: str = "fb"
    fb_basis: FBMomentBasis | None = None
    shapes: tuple = ()
    augment_reflections: bool = False


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _even_side(value, name):
    value = _positive_integer(value, name)
    if value < 4 or value % 2:
        raise ValueError(f"{name} must be even and at least 4")
    return value


def _device_name(value):
    value = str(value).lower()
    if value not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _elapsed_seconds(start):
    """Return a validated elapsed time from a monotonic clock."""

    elapsed = float(time.monotonic() - start)
    if not np.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("monotonic elapsed time is invalid")
    return elapsed


def _scalar(saved, key):
    value = np.asarray(saved[key])
    if value.size != 1:
        raise ValueError(f"manifest field {key} must be scalar")
    return value.reshape(()).item()


def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_fingerprint():
    """Return stable identities for every reconstruction implementation file."""

    return {
        name: {
            "path": str(RECONSTRUCTION_IMPLEMENTATION_FILES[name].resolve()),
            "sha256": _sha256(RECONSTRUCTION_IMPLEMENTATION_FILES[name]),
        }
        for name in sorted(RECONSTRUCTION_IMPLEMENTATION_FILES)
    }


def _sha256_array(value):
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes()).hexdigest()


def _git_commit():
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest_path(directory, value):
    path = Path(str(value))
    if not path.is_absolute():
        path = directory / path
    return path.resolve(strict=True)


def validate_denoising_output(
        directory, expected_dataset=None, *, min_images=1,
        expected_denoise_res=None):
    """Validate a :mod:`denoising` output and return its source geometry."""

    directory = Path(directory).resolve(strict=True)
    manifest_path = (directory / "manifest.npz").resolve(strict=True)
    report_path = (directory / "report.json").resolve(strict=True)
    with np.load(manifest_path, allow_pickle=False) as saved:
        required = {
            "preprocessing_version", "dataset", "star_path", "data_folder",
            "stack_prefix", "source_fingerprint", "source_rows",
            "image_names", "n_total", "n_images", "seed", "native_res",
            "denoise_res", "vox_native", "vox_denoise", "denoised_stack",
            "ctf_corrected", "noise_denoised", "whitened_output",
            "background_std_normalized", "contrast_removed",
            "relion_poses_used", "relion_origins_used", "reference_used",
            "contrast_est",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(
                f"denoising manifest is missing fields: {sorted(missing)}")
        dataset = str(_scalar(saved, "dataset"))
        n_total = int(_scalar(saved, "n_total"))
        n_images = int(_scalar(saved, "n_images"))
        native_res = int(_scalar(saved, "native_res"))
        denoise_res = int(_scalar(saved, "denoise_res"))
        native_pixel = float(_scalar(saved, "vox_native"))
        denoise_pixel = float(_scalar(saved, "vox_denoise"))
        source_rows = np.asarray(saved["source_rows"], dtype=np.int64)
        image_names = np.asarray(saved["image_names"])
        contrast_est = np.asarray(saved["contrast_est"], dtype=np.float64)
        stack_path = _manifest_path(
            directory, _scalar(saved, "denoised_stack"))
        expected_flags = {
            "ctf_corrected": True,
            "noise_denoised": True,
            "whitened_output": False,
            "background_std_normalized": False,
            "contrast_removed": False,
            "relion_poses_used": False,
            "relion_origins_used": False,
            "reference_used": False,
        }
        for key, expected in expected_flags.items():
            if bool(_scalar(saved, key)) is not expected:
                raise ValueError(
                    f"unexpected denoising manifest field {key}="
                    f"{_scalar(saved, key)!r}; expected {expected!r}")
        provenance = {
            key: _scalar(saved, key)
            for key in ("preprocessing_version", "star_path", "data_folder",
                        "stack_prefix", "source_fingerprint", "seed")
        }
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError(
            f"denoising dataset {dataset!r} differs from requested "
            f"{expected_dataset!r}")
    if (n_total < n_images or n_images < min_images
            or native_res < denoise_res or native_pixel <= 0
            or not np.isfinite(native_pixel)):
        raise ValueError("invalid denoising counts or native geometry")
    if (expected_denoise_res is not None
            and denoise_res != expected_denoise_res):
        raise ValueError(
            f"denoised images have side {denoise_res}, expected "
            f"{expected_denoise_res}")
    expected_pixel = native_res * native_pixel / denoise_res
    if (not np.isfinite(denoise_pixel)
            or not math.isclose(denoise_pixel, expected_pixel,
                                rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError(
            "denoised pixel size does not preserve the native particle box")
    if (source_rows.shape != (n_images,)
            or image_names.shape != (n_images,)
            or np.any(source_rows < 0)
            or len(np.unique(source_rows)) != n_images):
        raise ValueError("manifest particle identities are invalid")
    if (contrast_est.shape != (n_images,)
            or not np.all(np.isfinite(contrast_est))):
        raise ValueError("manifest contrast estimates are invalid")
    with mrcfile.mmap(stack_path, mode="r", permissive=False) as stack:
        pixel_xy = np.array([
            float(stack.voxel_size.x), float(stack.voxel_size.y)])
        if (stack.data.shape != (n_images, denoise_res, denoise_res)
                or stack.data.dtype != np.dtype(np.float32)
                or not np.allclose(pixel_xy, denoise_pixel,
                                   rtol=2e-6, atol=0.0)):
            raise ValueError(
                "denoised MRC stack and manifest geometry differ")
    provenance.update(
        dataset=dataset, n_total=n_total, n_images=n_images,
        native_res=native_res, native_pixel=native_pixel,
        denoise_res=denoise_res, denoise_pixel=denoise_pixel,
        source_rows=source_rows, image_names=image_names,
        contrast_est=contrast_est,
    )
    return directory, stack_path, manifest_path, report_path, provenance


def load_selection(selection_file, manifest, expected_count):
    """Validate selected particle identities against a denoising manifest."""

    selection_file = Path(selection_file).resolve(strict=True)
    expected_count = _positive_integer(expected_count, "expected_count")
    with np.load(selection_file, allow_pickle=False) as saved:
        required = {
            "selected_positions", "selected_source_rows",
            "selected_image_names",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"selection is missing fields: {sorted(missing)}")
        positions = np.asarray(saved["selected_positions"], dtype=np.int64)
        rows = np.asarray(saved["selected_source_rows"], dtype=np.int64)
        names = np.asarray(saved["selected_image_names"])
    if (positions.shape != (expected_count,) or rows.shape != positions.shape
            or names.shape != positions.shape):
        raise ValueError(
            f"selection must contain {expected_count:,} identities")
    if (np.any(positions < 0) or np.any(positions >= manifest["n_images"])
            or np.any(np.diff(positions) <= 0)):
        raise ValueError(
            "selected_positions must be sorted, unique, and in range")
    np.testing.assert_array_equal(rows, manifest["source_rows"][positions])
    np.testing.assert_array_equal(names, manifest["image_names"][positions])
    return selection_file, positions, rows, names


def empirical_moments_from_denoised_images(
        images, *, moment_res=DEFAULT_MOMENT_RES, ell_max=DEFAULT_ELL_MAX,
        batch_size=256):
    """Form the in-plane-invariant steerable FB M1--M3 blocks.

    Images are expanded in :class:`moments.FBMomentBasis` with angular
    frequencies ``|q| <= ell_max``. Only blocks whose angular frequencies sum
    to zero are formed; this equals averaging each image over all in-plane
    rotations. No additional correction is applied.
    """

    side = _even_side(moment_res, "moment_res")
    batch_size = _positive_integer(batch_size, "batch_size")
    basis, (m1, m2, m3), diagnostics = form_empirical_moments(
        images, side, ell_max, batch_size=batch_size)
    if diagnostics["q0_reality_relative_error"] > 1e-10:
        raise ValueError("q=0 FB coefficients violate Fourier reality")
    if not all(np.all(np.isfinite(value)) for value in (m1, m2, m3)):
        raise FloatingPointError("empirical moments contain non-finite values")
    metadata = dict(
        preparation_version=FB_PREPARATION_VERSION,
        moment_basis="fb",
        n_images=len(images), denoise_res=diagnostics["image_side"],
        moment_res=side, fb=basis.summary(), batch_size=batch_size,
        q0_reality_relative_error=diagnostics["q0_reality_relative_error"],
        fft_convention=(
            "image DTFT matching the centered unnormalized FFT crop; angular "
            "Fourier integrals evaluated exactly with Bessel functions"),
        moment_normalization="uniform average over selected images",
        moment_compression=(
            "in-plane-invariant blocks only (angular frequencies sum to "
            "zero), |q| <= ell_max"),
        moment_weighting=(
            "each flat block scaled by sqrt(number of complete-tensor blocks "
            "it represents)"),
        m1_definition="mean(c_0)",
        m2_definition="mean(c_q c_q^H), q = 0..ell_max",
        m3_definition=(
            "mean(c_a (x) c_b (x) conjugate(c_{a+b})), 0 <= a <= b, "
            "a + b <= ell_max"),
        ctf_correction_applied=False, noise_debias_applied=False,
        whitening_applied=False, shrinkage_applied=False,
        mean_subtraction_applied=False,
        per_image_contrast_division_applied=False,
        shifts_removed=False, poses_used=False,
    )
    return dict(
        mbar1=m1, mbar2=m2, mbar3=m3, metadata=metadata, basis=basis,
    )


def _fb_archive_fields(basis):
    nodes, weights, radial, counts = basis.radial_arrays()
    return dict(
        moment_basis="fb", fb_ell_max=basis.ell_max,
        fb_radial_nodes=nodes, fb_radial_weights=weights,
        fb_radial_basis=radial, fb_radial_counts=counts,
        fb_basis_sha256=basis.fingerprint(),
    )


def prepare_moments(config: MomentPreparationConfig):
    """Prepare complete M1--M3 from a selected denoised image subset."""

    if not isinstance(config, MomentPreparationConfig):
        raise TypeError("config must be MomentPreparationConfig")
    output = config.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new or empty moment preparation directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timings = {}

    phase_started = time.monotonic()
    use_device(config.device)
    expected_m3 = FBMomentBasis(
        config.moment_res, config.fb_ell_max).m3_size
    if expected_m3 > config.max_m3_entries:
        raise ValueError(
            f"side-{config.moment_res} Fourier-Bessel M3 has "
            f"{expected_m3:,} entries, exceeding "
            f"max_m3_entries={config.max_m3_entries:,}")
    timings["setup"] = _elapsed_seconds(phase_started)

    phase_started = time.monotonic()
    (denoising_dir, stack_path, manifest_path, denoising_report,
     manifest) = validate_denoising_output(
         config.denoising_dir, config.expected_dataset,
         min_images=config.n_images,
         expected_denoise_res=config.denoise_res,
    )
    selection_path, positions, rows, names = load_selection(
        config.selection_file, manifest, config.n_images)
    selection_report_path = selection_path.with_name("selection_report.json")
    if not selection_report_path.is_file():
        selection_report_path = selection_path.with_name("report.json")
    selection_report = None
    if selection_report_path.is_file():
        selection_report = json.loads(selection_report_path.read_text())
        if int(selection_report.get("n_selected", -1)) != config.n_images:
            raise ValueError("selection report and archive disagree")
        if (selection_report.get("uses_reference_map", False)
                or selection_report.get(
                    "uses_relion_poses_for_selection", False)):
            raise ValueError(
                "selection must not use a reference map or RELION poses")
    timings["input_validation"] = _elapsed_seconds(phase_started)

    phase_started = time.monotonic()
    manifest_hash = _sha256(manifest_path)
    stack_stat = stack_path.stat()
    with mrcfile.mmap(stack_path, mode="r", permissive=False) as handle:
        selected = np.asarray(handle.data[positions], dtype=np.float32)
    timings["selected_image_loading"] = _elapsed_seconds(phase_started)

    phase_started = time.monotonic()
    empirical = empirical_moments_from_denoised_images(
        selected, moment_res=config.moment_res,
        ell_max=config.fb_ell_max, batch_size=config.batch_size,
    )
    del selected
    timings["moment_formation"] = _elapsed_seconds(phase_started)

    phase_started = time.monotonic()
    validate_denoising_output(
        denoising_dir, manifest["dataset"], min_images=config.n_images,
        expected_denoise_res=config.denoise_res,
    )
    new_stat = stack_path.stat()
    if (_sha256(manifest_path) != manifest_hash
            or new_stat.st_size != stack_stat.st_size
            or new_stat.st_mtime_ns != stack_stat.st_mtime_ns):
        raise ValueError("denoising input changed during moment formation")
    timings["input_revalidation"] = _elapsed_seconds(phase_started)

    phase_started = time.monotonic()
    m1, m2, m3 = (
        empirical[f"mbar{order}"] for order in (1, 2, 3))
    if config.augment_reflections:
        # Mirroring an image conjugates its stored q>=0 FB coefficients, so
        # averaging every image with its mirror leaves exactly the real
        # parts of the invariant moments; no images are duplicated.
        m1, m2, m3 = (np.ascontiguousarray(m.real).astype(np.complex128)
                      for m in (m1, m2, m3))
    basis = empirical["basis"]
    n_frequencies = basis.n_coefficients
    if (tuple(np.shape(m) for m in (m1, m2, m3)) != basis.shapes
            or len(m3) != expected_m3):
        raise RuntimeError("FB moments do not match the expected layout")
    n_m3 = len(m3)
    basis_fields = _fb_archive_fields(basis)
    norms = [float(np.linalg.norm(moment)) for moment in (m1, m2, m3)]
    if not all(np.isfinite(value) and value > 0 for value in norms):
        raise FloatingPointError("empirical moments have invalid norms")
    pixel = manifest["native_res"] * manifest["native_pixel"] \
        / config.moment_res
    timings["moment_validation_and_metadata"] = _elapsed_seconds(
        phase_started)

    output_started = time.monotonic()
    moments_path = output / "moments.npz"
    np.savez(
        moments_path,
        mbar1=m1, mbar2=m2, mbar3=m3,
        # Retain these two fixed legacy fields so existing prepared archives,
        # fit archives, and resume validation remain byte-schema compatible.
        support=np.arange(basis.n_coefficients, dtype=np.int64),
        triples=np.empty((0, 3), dtype=np.int64),
        selected_positions=positions, selected_source_rows=rows,
        selected_image_names=names,
        n_images=config.n_images, ds_res=config.moment_res,
        dataset=manifest["dataset"], pixel_size=pixel,
        native_side=manifest["native_res"],
        native_pixel_size=manifest["native_pixel"],
        denoise_side=manifest["denoise_res"],
        denoised_pixel_size=manifest["denoise_pixel"],
        dc_normalized=False, dc_excluded=False,
        augment_reflections=bool(config.augment_reflections),
        **basis_fields,
    )
    shutil.copy2(selection_path, output / "selection.npz")
    if selection_report_path.is_file():
        shutil.copy2(
            selection_report_path, output / "selection_report.json")
    shutil.copy2(manifest_path, output / "denoising_manifest.npz")
    metadata = empirical["metadata"]
    metadata.update(
        moment_norms=norms,
        m2_hermitian_relative_error=None,
        exact_dc_retained=True,
        dc_normalization_applied=False,
        dc_removal_applied=False,
    )
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        preparation_version=FB_PREPARATION_VERSION,
        moment_basis="fb",
        fb_ell_max=config.fb_ell_max,
        augment_reflections=bool(config.augment_reflections),
        dataset=manifest["dataset"],
        denoising_directory=str(denoising_dir),
        denoised_stack=str(stack_path),
        denoising_manifest=str(manifest_path),
        denoising_manifest_sha256=manifest_hash,
        denoising_report=str(denoising_report),
        selection_file=str(selection_path),
        selection_sha256=_sha256(selection_path),
        selection_report=(
            str(selection_report_path.resolve())
            if selection_report_path.is_file() else None),
        selection_uses_reference=(
            None if selection_report is None
            else selection_report["uses_reference_map"]),
        selection_uses_relion_poses=(
            None if selection_report is None
            else selection_report["uses_relion_poses_for_selection"]),
        n_images=config.n_images,
        selected_positions_are_stack_positions=True,
        denoise_res=config.denoise_res,
        moment_res=config.moment_res,
        pixel_size_angstrom=pixel,
        native_side=manifest["native_res"],
        native_pixel_size_angstrom=manifest["native_pixel"],
        fourier_disk_radius=None,
        n_frequencies=n_frequencies,
        n_m2_entries=int(m2.size),
        n_canonical_m3_entries=n_m3,
        max_m3_entries=config.max_m3_entries,
        moment_compression="in-plane-invariant steerable FB blocks",
        moment_centering="none; uncentered image moments",
        dc_handling="retained exactly; no image scaling or removal",
        contrast_handling="none after denoising",
        shift_handling="observed shifts retained; fits fix sigma to zero",
        ctf_correction_again=False,
        noise_debias_again=False,
        source=dict(
            preprocessing_version=str(manifest["preprocessing_version"]),
            star_path=str(manifest["star_path"]),
            data_folder=str(manifest["data_folder"]),
            stack_prefix=str(manifest["stack_prefix"]),
            source_fingerprint=str(manifest["source_fingerprint"]),
            contrast_estimate_use=(
                "selection stratification and provenance only; no image "
                "division"),
            contrast_estimate_sha256=_sha256_array(
                manifest["contrast_est"]),
            contrast_estimate_range=[
                float(manifest["contrast_est"].min()),
                float(manifest["contrast_est"].max()),
            ],
        ),
        metadata=metadata,
        config={**asdict(config),
                "denoising_dir": str(config.denoising_dir),
                "selection_file": str(config.selection_file),
                "output_dir": str(config.output_dir)},
        prepared_file=str(moments_path),
        prepared_sha256=_sha256(moments_path),
        kam_cryoem_commit=_git_commit(),
        module=str(Path(__file__).resolve()),
        module_sha256=_sha256(__file__),
        implementation=_implementation_fingerprint(),
    )
    report_path = output / "report.json"
    timings["output_serialization"] = _elapsed_seconds(output_started)
    elapsed = _elapsed_seconds(started)
    timings["total"] = elapsed
    report["timings_seconds"] = dict(timings)
    report["elapsed_seconds"] = elapsed
    # Write once, then include a complete report write in the measured output
    # phase.  The second JSON rewrite only updates those timing values.
    _write_json(report_path, report)
    timings["output_serialization"] = _elapsed_seconds(output_started)
    elapsed = _elapsed_seconds(started)
    timings["total"] = elapsed
    report["timings_seconds"] = dict(timings)
    report["elapsed_seconds"] = elapsed
    _write_json(report_path, report)
    return PreparedMomentsResult(
        output_dir=output,
        moments_path=moments_path,
        report_path=report_path,
        dataset=manifest["dataset"],
        n_images=config.n_images,
        denoise_res=config.denoise_res,
        moment_res=config.moment_res,
        pixel_size=pixel,
        n_frequencies=n_frequencies,
        n_m2_entries=int(m2.size),
        n_m3_entries=n_m3,
        elapsed_seconds=elapsed,
    )


def load_prepared_moments(path, order):
    """Load and validate one active target from a prepared moment archive."""

    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2 or 3")
    path = Path(path).resolve(strict=True)
    report_path = path / "report.json"
    archive = path / "moments.npz"
    report = json.loads(report_path.read_text())
    if report.get("implementation") != _implementation_fingerprint():
        raise ValueError(
            "prepared moments were produced by a different implementation; "
            "prepare them again before reconstruction")
    if _sha256(archive) != report["prepared_sha256"]:
        raise ValueError("prepared moment archive hash differs from report")
    with np.load(archive, allow_pickle=False) as saved:
        values = {key: np.asarray(saved[key]).copy() for key in saved.files}
    required = {
        "mbar1", "mbar2", "mbar3", "support", "triples", "n_images",
        "ds_res", "denoise_side", "native_side", "native_pixel_size",
        "pixel_size", "dataset", "dc_normalized", "dc_excluded",
    }
    missing = required.difference(values)
    if missing:
        raise ValueError(
            f"prepared moment archive is missing fields: {sorted(missing)}")
    stored_support = np.asarray(values["support"], dtype=np.int64)
    stored_triples = np.asarray(values["triples"], dtype=np.int64)
    n_images = int(values["n_images"].reshape(()))
    side = int(values["ds_res"].reshape(()))
    denoise_side = int(values["denoise_side"].reshape(()))
    _even_side(side, "prepared moment_res")
    _even_side(denoise_side, "prepared denoise_res")
    moment_basis = str(values.get(
        "moment_basis", np.asarray("")).reshape(()))
    if moment_basis != "fb":
        raise ValueError(
            "prepared moments must use the Fourier-Bessel representation")
    if (n_images < 1 or denoise_side < side
            or bool(values["dc_normalized"].reshape(()))
            or bool(values["dc_excluded"].reshape(()))):
        raise ValueError("prepared moments do not match the full-moment setup")
    fb_basis = _load_fb_basis(values, side)
    shapes = fb_basis.shapes
    if (not np.array_equal(
            stored_support, np.arange(fb_basis.n_coefficients))
            or stored_triples.shape != (0, 3)):
        raise ValueError("prepared FB coefficient layout is inconsistent")
    native_side = int(values["native_side"].reshape(()))
    native_pixel = float(values["native_pixel_size"].reshape(()))
    pixel = float(values["pixel_size"].reshape(()))
    if (native_side < denoise_side or native_pixel <= 0
            or not math.isclose(
                pixel, native_side * native_pixel / side,
                rel_tol=1e-12, abs_tol=0.0)):
        raise ValueError("prepared particle geometry is inconsistent")
    target = np.asarray(values[f"mbar{order}"], dtype=np.complex128)
    if (target.shape != shapes[order - 1]
            or not np.all(np.isfinite(target))):
        raise ValueError(f"prepared M{order} target is invalid")
    augmented = bool(values.get(
        "augment_reflections", np.asarray(False)).reshape(()))
    if augmented and np.any(np.iscomplex(target)):
        raise ValueError(
            "reflection-augmented moments must be real; the archive is stale")
    return PreparedMomentData(
        directory=path, report=report, values=values,
        target=target, order=order,
        side=side, pixel_size=pixel, n_images=n_images,
        moment_basis=moment_basis, fb_basis=fb_basis, shapes=shapes,
        augment_reflections=augmented,
    )


def _load_fb_basis(values, side):
    """Rebuild the stored FB radial basis and verify its fingerprint."""

    required = {
        "fb_ell_max", "fb_radial_nodes", "fb_radial_weights",
        "fb_radial_basis", "fb_radial_counts", "fb_basis_sha256",
    }
    missing = required.difference(values)
    if missing:
        raise ValueError(
            f"prepared FB archive is missing fields: {sorted(missing)}")
    basis = FBMomentBasis(
        side, int(values["fb_ell_max"].reshape(())),
        radial=tuple(values[key] for key in (
            "fb_radial_nodes", "fb_radial_weights", "fb_radial_basis",
            "fb_radial_counts")))
    if basis.fingerprint() != str(values["fb_basis_sha256"].reshape(())):
        raise ValueError("prepared FB radial basis fingerprint differs")
    return basis


def deterministic_random_starts(n_parameters, n_init, seed=0):
    """Return reproducible Gaussian coefficient directions and provenance."""

    n_parameters = _positive_integer(n_parameters, "n_parameters")
    n_init = _positive_integer(n_init, "n_init")
    if isinstance(seed, (bool, np.bool_)) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    starts = []
    for index in range(n_init):
        child_seed = int(np.random.SeedSequence(
            [int(seed), 2, index]).generate_state(1)[0])
        direction = np.random.default_rng(child_seed).standard_normal(
            n_parameters)
        starts.append((
            index, child_seed, direction, _sha256_array(direction)))
    return starts


def select_best_fit(records):
    """Return the record with minimum finite normalized moment objective.

    Ties retain the first record.  No reference-derived field is inspected.
    """

    records = list(records)
    if not records:
        raise ValueError("at least one fit record is required")
    best = records[0]
    best_loss = float(best["objective"])
    if not np.isfinite(best_loss):
        raise ValueError("fit objectives must be finite")
    for record in records[1:]:
        loss = float(record["objective"])
        if not np.isfinite(loss):
            raise ValueError("fit objectives must be finite")
        if loss < best_loss:
            best, best_loss = record, loss
    return best


def _config_payload(
        config, prepared, param, density, grid, target_norm,
        warm_fit_paths=()):
    initial_fit = (None if config.initial_fit is None
                   else str(config.initial_fit.resolve(strict=True)))
    return dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        dataset=str(prepared.values["dataset"].reshape(())),
        order=config.order,
        moment_basis=prepared.moment_basis,
        fb_basis=(None if prepared.fb_basis is None
                  else prepared.fb_basis.summary()),
        objective=f"in-plane-invariant steerable FB M{config.order} only",
        objective_normalization="1 / squared Frobenius norm of active target",
        prepared_directory=str(prepared.directory),
        prepared_sha256=prepared.report["prepared_sha256"],
        selected_images=prepared.n_images,
        moment_res=prepared.side,
        pixel_size_angstrom=prepared.pixel_size,
        native_side=int(prepared.values["native_side"].reshape(())),
        native_pixel_size_angstrom=float(
            prepared.values["native_pixel_size"].reshape(())),
        disk_frequencies=None,
        fb_coefficients=prepared.fb_basis.n_coefficients,
        n_m2_entries=int(np.prod(prepared.shapes[1])),
        canonical_m3_entries=int(np.prod(prepared.shapes[2])),
        moment_compression="in-plane-invariant FB blocks, |q| <= ell_max",
        ell_max=config.ell_max,
        n_volume_parameters=param.n_params,
        viewing_representation="positive full-real-SH SOS density",
        viewing_degree=config.viewing_degree,
        n_view_parameters=density.n_parameters,
        sphere_node_request=config.sphere_nodes,
        sphere_degree_request=config.sphere_degree,
        reflection_augmented=prepared.augment_reflections,
        sphere_nodes=grid.Ns,
        inplane_angles=grid.n_gamma,
        so3_atoms=grid.M,
        shift_sigma=0.0,
        fit_shift=False,
        contrast_scales=[1.0, 1.0, 1.0],
        image_dc_normalized=False,
        image_dc_excluded=False,
        weighting="FB blocks scaled to the complete-tensor Frobenius norm",
        target_norm=target_norm,
        target_sha256=_sha256_array(prepared.target),
        n_initializations=config.n_init,
        maxiter=config.maxiter,
        initialization=(
            "deterministic Gaussian coefficient directions from "
            "SeedSequence([seed, 2, index]), rescaled to the "
            f"M{config.order} target norm"
            if not warm_fit_paths else
            f"M{config.order - 1} init_i volume and viewing density, with "
            "the volume analytically rescaled by least squares of order "
            f"{config.order} against M{config.order}"),
        initialization_seed=config.seed,
        initial_fit=initial_fit,
        initial_fit_files=[str(path) for path in warm_fit_paths],
        initial_fit_sha256=[_sha256(path) for path in warm_fit_paths],
        selection=(
            "strict minimum normalized active-moment loss; reference never "
            "selects a start"),
        reference_used_for_optimization=False,
        kam_cryoem_commit=_git_commit(),
        module=str(Path(__file__).resolve()),
        module_sha256=_sha256(__file__),
        implementation=_implementation_fingerprint(),
    )


def _resume_keys(payload):
    return (
        "dataset", "order", "moment_basis", "prepared_sha256",
        "selected_images",
        "moment_res", "ell_max", "viewing_degree", "sphere_node_request",
        "sphere_nodes", "inplane_angles", "so3_atoms",
        "n_volume_parameters", "n_view_parameters", "n_initializations",
        "maxiter", "initialization_seed", "target_sha256",
        "initial_fit_sha256",
        "implementation",
    )


def _read_completed_fit(output, index, expected_order):
    directory = output / f"init_{index}"
    summary_path = directory / "summary.json"
    fit_path = directory / "fit.npz"
    recon_path = directory / "recon.mrc"
    if not (summary_path.is_file() and fit_path.is_file()
            and recon_path.is_file()):
        return None
    record = json.loads(summary_path.read_text())
    if (int(record.get("init_index", -1)) != index
            or int(record.get("moment_order", -1)) != expected_order
            or not np.isfinite(float(record.get("objective", np.nan)))):
        raise ValueError(f"invalid completed fit record in {directory}")
    with np.load(fit_path, allow_pickle=False) as saved:
        if (int(np.asarray(saved["moment_order"]).reshape(()))
                != expected_order
                or not np.isfinite(
                    float(np.asarray(saved["objective"]).reshape(())))):
            raise ValueError(f"invalid completed fit archive in {directory}")
        archive_loss = float(np.asarray(saved["objective"]).reshape(()))
    if not math.isclose(
            archive_loss, float(record["objective"]),
            rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"fit objective and summary disagree in {directory}")
    return record


def _copy_best_fit(output, record):
    source = output / f"init_{int(record['init_index'])}"
    shutil.copy2(source / "fit.npz", output / "fit.npz")
    shutil.copy2(source / "recon.mrc", output / "recon.mrc")


def _load_m2_warm_start(path, prepared, config, param, density, grid):
    """Load and validate one previous-order fit for the next moment."""

    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "theta", "beta", "moment_order", "shift_sigma", "fit_shift",
            "viewing_degree", "viewing_antipodal", "ell_max", "grid_rots",
            "grid_xyz", "grid_q", "support", "prepared_sha256",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(
                f"M2 warm start is missing fields: {sorted(missing)}")
        theta = np.asarray(saved["theta"], dtype=np.float64)
        beta = np.asarray(saved["beta"], dtype=np.float64)
        order = int(np.asarray(saved["moment_order"]).reshape(()))
        shift_sigma = float(np.asarray(saved["shift_sigma"]).reshape(()))
        fit_shift = bool(np.asarray(saved["fit_shift"]).reshape(()))
        viewing_degree = int(
            np.asarray(saved["viewing_degree"]).reshape(()))
        viewing_antipodal = bool(
            np.asarray(saved["viewing_antipodal"]).reshape(()))
        ell_max = int(np.asarray(saved["ell_max"]).reshape(()))
        prepared_sha256 = str(
            np.asarray(saved["prepared_sha256"]).reshape(()))
        support = np.asarray(saved["support"], dtype=np.int64)
        rots = np.asarray(saved["grid_rots"], dtype=np.float64)
        xyz = np.asarray(saved["grid_xyz"], dtype=np.float64)
        q = np.asarray(saved["grid_q"], dtype=np.float64)
    if (order != config.order - 1 or theta.shape != (param.n_params,)
            or beta.shape != (density.n_parameters,)
            or not np.all(np.isfinite(theta))
            or not np.all(np.isfinite(beta))
            or shift_sigma != 0.0 or fit_shift
            or viewing_degree != config.viewing_degree
            or viewing_antipodal != density.antipodal
            or ell_max != config.ell_max
            or prepared_sha256 != prepared.report["prepared_sha256"]):
        raise ValueError(
            f"initial_fit is not a compatible fixed-zero-shift "
            f"M{config.order - 1} fit")
    np.testing.assert_array_equal(
        support, np.arange(prepared.fb_basis.n_coefficients))
    np.testing.assert_allclose(rots, grid.rots, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(xyz, grid.xyz, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(q, grid.q, rtol=0.0, atol=0.0)
    return path, theta, beta


def _resolve_m2_fit_paths(initial_fit, n_init):
    """Resolve one compatible M2 fit per requested M3 initialization."""

    initial_fit = Path(initial_fit).resolve(strict=True)
    if initial_fit.is_dir():
        directory = initial_fit
    elif initial_fit.is_file() and (initial_fit.parent / "init_0").is_dir():
        directory = initial_fit.parent
    elif initial_fit.is_file() and n_init == 1:
        return [initial_fit]
    else:
        raise ValueError(
            "initial_fit must be an M2 output directory containing "
            "init_i/fit.npz for every requested M3 start")
    paths = [
        (directory / f"init_{index}" / "fit.npz").resolve(strict=True)
        for index in range(n_init)
    ]
    if len({str(path) for path in paths}) != n_init:
        raise ValueError("M2 warm-start fit paths must be distinct")
    return paths


def _m3_warm_starts(
        warm_fits, config, prepared, param, density, grid, targets, lambdas):
    """Rescale each previous-order initialization for the next objective."""

    if len(warm_fits) != config.n_init:
        raise ValueError(
            f"M{config.order} requires one completed M{config.order - 1} "
            "fit per initialization")
    starts = []
    for index, (path, candidate, warm_beta) in enumerate(warm_fits):
        norm = float(np.linalg.norm(candidate))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("M2 warm-start volume has invalid norm")
        p = density.pmf(warm_beta)
        _, _, _, info = joint_objective_and_grad(
            candidate, p, param, grid, *targets,
            lambdas=lambdas,
            contrast_scales=np.ones(3), return_info=True,
        )
        prediction = info[f"m{config.order}"]
        denominator = float(np.vdot(prediction, prediction).real)
        numerator = float(np.vdot(prediction, prepared.target).real)
        if (not np.isfinite(denominator) or denominator <= 0
                or not np.isfinite(numerator)):
            raise FloatingPointError(
                f"warm M{config.order} start has invalid amplitude fit")
        amplitude = numerator / denominator
        scale = float(np.sign(amplitude)
                      * np.abs(amplitude) ** (1.0 / config.order))
        if not np.isfinite(scale) or scale == 0.0:
            raise FloatingPointError(
                f"warm start is orthogonal to the M{config.order} target")
        starts.append((
            index, -1, scale * candidate, warm_beta.copy(), scale,
            float(np.linalg.norm(prediction)), _sha256_array(candidate),
            f"analytic M{config.order} rescaling of {path}", 0.0, amplitude,
        ))
    return starts


def reconstruct_single_moment(config: ReconstructionConfig):
    """Fit one moment order and select the start with minimum moment loss."""

    if not isinstance(config, ReconstructionConfig):
        raise TypeError("config must be ReconstructionConfig")
    output = config.output_dir.resolve()
    nonempty = output.exists() and any(output.iterdir())
    if nonempty and not config.resume:
        raise FileExistsError(
            "use a new or empty reconstruction directory, or set resume=True")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    setup_started = time.monotonic()
    prepared = load_prepared_moments(config.prepared_dir, config.order)
    target_norm = float(np.linalg.norm(prepared.target))
    if not np.isfinite(target_norm) or target_norm <= 0:
        raise ValueError(f"M{config.order} target has invalid norm")

    use_device(config.device)
    if prepared.fb_basis.ell_max != config.ell_max:
        raise ValueError(
            f"FB moments were prepared with ell_max="
            f"{prepared.fb_basis.ell_max}; reconstruct with the same ell_max "
            "or prepare them again")
    # FB moments are already in-plane averaged: viewing directions only.
    grid = so3_sphere_grid(config.sphere_nodes, 1, degree=config.sphere_degree)
    param = SphFBParam(
        prepared.side, config.ell_max, grid.rots,
        support=None,
        fb_basis=prepared.fb_basis,
    )
    if param.ell_max != config.ell_max:
        raise ValueError(
            f"ell_max={config.ell_max} is unsupported at moment_res="
            f"{prepared.side}; radial truncation supports at most "
            f"ell_max={param.ell_max}")
    density = LowDegreeViewingDensity(
        grid, degree=config.viewing_degree,
        antipodal=bool(prepared.augment_reflections))
    targets = [np.zeros(shape, dtype=np.complex128)
               for shape in prepared.shapes]
    targets[config.order - 1] = prepared.target
    lambdas = [0.0, 0.0, 0.0]
    lambdas[config.order - 1] = 1.0 / target_norm**2
    beta_uniform = np.zeros(density.n_parameters)
    if config.initial_fit is None:
        p_uniform = density.pmf(beta_uniform)
        starts = []
        for index, child_seed, direction, direction_hash in \
                deterministic_random_starts(
                    param.n_params, config.n_init, config.seed):
            _, _, _, info = joint_objective_and_grad(
                direction, p_uniform, param, grid, *targets,
                lambdas=lambdas, contrast_scales=np.ones(3),
                return_info=True,
            )
            prediction = info[f"m{config.order}"]
            prediction_norm = float(np.linalg.norm(prediction))
            if not np.isfinite(prediction_norm) or prediction_norm <= 0:
                raise FloatingPointError(
                    f"random M{config.order} start has invalid moment norm")
            scale = (target_norm / prediction_norm) ** (1.0 / config.order)
            starts.append((
                index, child_seed, scale * direction, beta_uniform.copy(),
                scale, prediction_norm, direction_hash,
                f"random Gaussian direction scaled to M{config.order} "
                "target norm",
                0.0, np.nan,
            ))
        warm_path = None
        warm_fit_paths = []
    else:
        warm_fit_paths = _resolve_m2_fit_paths(
            config.initial_fit, config.n_init)
        warm_fits = [
            _load_m2_warm_start(
                path, prepared, config, param, density, grid)
            for path in warm_fit_paths
        ]
        starts = _m3_warm_starts(
            warm_fits, config, prepared, param, density, grid, targets,
            lambdas,
        )
        warm_path = Path(config.initial_fit).resolve(strict=True)

    payload = _config_payload(
        config, prepared, param, density, grid, target_norm,
        warm_fit_paths=warm_fit_paths)
    payload["init_seeds"] = [start[1] for start in starts]
    payload["initialization_sources"] = [start[7] for start in starts]
    payload["m3_warm_start_semantics"] = (
        None if config.initial_fit is None else
        f"M{config.order} start i is M{config.order - 1} init_i with its "
        "fitted viewing density; each volume receives the analytic "
        f"order-{config.order} least-squares amplitude against "
        f"M{config.order}")
    config_path = output / "config.json"
    if nonempty:
        if not config_path.is_file():
            raise ValueError("resume directory is missing config.json")
        previous = json.loads(config_path.read_text())
        mismatched = {
            key: (previous.get(key), payload.get(key))
            for key in _resume_keys(payload)
            if previous.get(key) != payload.get(key)
        }
        if mismatched:
            raise ValueError(
                f"resume configuration differs from existing fit: {mismatched}")
        payload["created_utc"] = previous.get(
            "created_utc", payload["created_utc"])
    else:
        _write_json(config_path, payload)
        np.savez(
            output / "grid.npz",
            grid_rots=grid.rots, grid_xyz=grid.xyz, grid_q=grid.q,
            support=np.arange(prepared.fb_basis.n_coefficients),
        )
        if warm_path is not None:
            _write_json(
                output / "m2_warm_starts.json",
                {"source": str(warm_path),
                 "fit_files": [str(path) for path in warm_fit_paths],
                 "sha256": [_sha256(path) for path in warm_fit_paths]},
            )

    completed = {}
    if config.resume:
        for index in range(config.n_init):
            record = _read_completed_fit(output, index, config.order)
            if record is not None:
                completed[index] = record
    setup_elapsed = _elapsed_seconds(setup_started)
    fit_loop_started = time.monotonic()
    optimization_elapsed_this_run = 0.0
    completed_this_run = 0
    records = []
    for (index, child_seed, theta0, beta0, scale, prediction_norm,
         direction_hash, start_source, perturbation_norm,
         cubic_amplitude) in starts:
        if index in completed:
            records.append(completed[index])
            print(
                f"M{config.order} start {index + 1}/{len(starts)}: "
                "using completed resume record",
                flush=True,
            )
            continue
        init_started = time.monotonic()
        init = output / f"init_{index}"
        init.mkdir(exist_ok=True)
        initial_loss = float(joint_objective_and_grad(
            theta0, density.pmf(beta0), param, grid, *targets,
            lambdas=lambdas,
            contrast_scales=np.ones(3),
        )[0])
        np.savez(
            init / "start.npz",
            theta=theta0, beta=beta0, p=density.pmf(beta0),
            viewing_antipodal=density.antipodal,
            seed=child_seed, moment_order=config.order,
            scale=scale,
            prediction_norm_before_scaling=prediction_norm,
            target_norm=target_norm,
            direction_sha256=direction_hash,
            objective=initial_loss,
            source=start_source,
            perturbation_norm=perturbation_norm,
            cubic_amplitude=cubic_amplitude,
        )
        print(
            f"M{config.order} start {index + 1}/{len(starts)}: "
            f"initial normalized loss={initial_loss:.8e}",
            flush=True,
        )
        optimization_started = time.monotonic()
        result = fit_single_moment_low_degree(
            param, grid, targets, lambdas, theta0,
            beta0=beta0, degree=config.viewing_degree,
            maxiter=config.maxiter, print_every=config.print_every,
            reflection_augmented=prepared.augment_reflections,
        )
        optimization_elapsed = _elapsed_seconds(optimization_started)
        optimization_elapsed_this_run += optimization_elapsed
        optimizer = result["res"]
        loss = float(optimizer.fun)
        expected_size = param.n_params + density.n_parameters
        relative_error = float(
            result["stages"][0]["rel_misfit"][config.order - 1])
        normalized_loss = relative_error**2
        if (result["x"].shape != (expected_size,)
                or result["stages"][0]["name"] != f"m{config.order}"
                or not np.isfinite(loss)
                or not math.isclose(loss, normalized_loss,
                                    rel_tol=2e-10, abs_tol=1e-14)):
            raise RuntimeError(
                "fit departed from the normalized single-moment fixed-zero "
                "objective")
        np.savez(
            init / "fit.npz",
            theta=result["theta"], beta=result["beta"], p=result["p"],
            viewing_antipodal=density.antipodal,
            init_index=index, initialization_source=start_source,
            objective=loss, normalized_active_moment_loss=loss,
            relative_moment_error=relative_error,
            moment_order=config.order,
            shift_sigma=0.0, shift_variance=0.0, fit_shift=False,
            viewing_degree=config.viewing_degree,
            ell_max=config.ell_max,
            grid_rots=grid.rots, grid_xyz=grid.xyz, grid_q=grid.q,
            support=np.arange(prepared.fb_basis.n_coefficients),
            prepared_sha256=prepared.report["prepared_sha256"],
        )
        save_mrc(
            param.volume(result["theta"]).transpose(2, 1, 0),
            init / "recon.mrc", prepared.pixel_size,
        )
        _write_json(
            init / "gradient_checks.json", result["gradient_checks"])
        record = dict(
            init_index=index,
            seed=child_seed,
            moment_order=config.order,
            initial_objective=initial_loss,
            initialization_source=start_source,
            perturbation_norm=perturbation_norm,
            cubic_amplitude=(
                None if not np.isfinite(cubic_amplitude)
                else float(cubic_amplitude)),
            objective=loss,
            normalized_active_moment_loss=loss,
            relative_moment_error=relative_error,
            nit=int(optimizer.nit),
            nfev=int(optimizer.nfev),
            status=int(optimizer.status),
            success=bool(optimizer.success),
            message=str(optimizer.message),
            gradient_norm=float(np.linalg.norm(optimizer.jac)),
            viewing=result["viewing_diagnostics"],
            optimization_elapsed_seconds=optimization_elapsed,
        )
        record["elapsed_seconds"] = _elapsed_seconds(init_started)
        record["nonoptimization_elapsed_seconds"] = max(
            0.0, record["elapsed_seconds"] - optimization_elapsed)
        _write_json(init / "summary.json", record)
        completed_this_run += 1
        records.append(record)
        best = select_best_fit(records)
        _copy_best_fit(output, best)
        partial = dict(payload)
        partial.update(
            fits=sorted(records, key=lambda item: item["init_index"]),
            best_init_by_loss=int(best["init_index"]),
            fit_complete=len(records) == len(starts),
            complete=False,
        )
        _write_json(output / "summary.json", partial)
        print("Fit result: " + json.dumps(record), flush=True)

    fit_loop_elapsed = _elapsed_seconds(fit_loop_started)
    final_output_started = time.monotonic()
    records.sort(key=lambda item: item["init_index"])
    if len(records) != len(starts):
        raise RuntimeError("not every requested initialization completed")
    best = select_best_fit(records)
    _copy_best_fit(output, best)
    with np.load(output / "fit.npz", allow_pickle=False) as saved:
        beta = np.asarray(saved["beta"], dtype=np.float64)
        selected_loss = float(np.asarray(saved["objective"]).reshape(()))
    if not math.isclose(
            selected_loss, float(best["objective"]),
            rel_tol=1e-12, abs_tol=1e-15):
        raise RuntimeError("top-level fit does not match the minimum-loss start")
    _save_viewing_plot(output, density, beta)

    timed_optimization_records = [
        record for record in records
        if "optimization_elapsed_seconds" in record
    ]
    timed_initialization_records = [
        record for record in records if "elapsed_seconds" in record
    ]
    optimization_total = float(sum(
        float(record["optimization_elapsed_seconds"])
        for record in timed_optimization_records
    ))
    initializations_total = float(sum(
        float(record["elapsed_seconds"])
        for record in timed_initialization_records
    ))
    for name, value in (
            ("optimization_total", optimization_total),
            ("initializations_total", initializations_total)):
        if not np.isfinite(value) or value < 0.0:
            raise RuntimeError(f"{name} timing is invalid")

    summary = dict(payload)
    summary.update(
        fits=records,
        best_init_by_loss=int(best["init_index"]),
        best_normalized_active_moment_loss=float(best["objective"]),
        fit_complete=True,
        timing_coverage=dict(
            initialization_records=len(timed_initialization_records),
            optimization_records=len(timed_optimization_records),
            requested_initializations=len(records),
            completed_in_this_invocation=completed_this_run,
            note=(
                "Older resumed records may lack timing fields; totals over "
                "initializations cover only records carrying those fields."
            ),
        ),
    )
    final_output_elapsed = _elapsed_seconds(final_output_started)
    elapsed = _elapsed_seconds(started)
    timings = dict(
        setup=setup_elapsed,
        initialization_fit_loop=fit_loop_elapsed,
        optimization_total=optimization_total,
        optimization_this_invocation=optimization_elapsed_this_run,
        initializations_total=initializations_total,
        final_output=final_output_elapsed,
        total=elapsed,
    )
    summary["timings_seconds"] = timings
    summary["elapsed_seconds"] = elapsed
    summary["complete"] = True
    _write_json(output / "summary.json", summary)

    # Include one complete summary write in the top-level output and elapsed
    # timings.  The second JSON rewrite only updates the measured values.
    final_output_elapsed = _elapsed_seconds(final_output_started)
    elapsed = _elapsed_seconds(started)
    timings["final_output"] = final_output_elapsed
    timings["total"] = elapsed
    summary["timings_seconds"] = timings
    summary["elapsed_seconds"] = elapsed
    _write_json(output / "summary.json", summary)
    return ReconstructionResult(
        output_dir=output,
        fit_path=output / "fit.npz",
        reconstruction_path=output / "recon.mrc",
        summary_path=output / "summary.json",
        order=config.order,
        best_init=int(best["init_index"]),
        objective=float(best["objective"]),
        relative_moment_error=float(best["relative_moment_error"]),
        n_volume_parameters=param.n_params,
        n_view_parameters=density.n_parameters,
        actual_sphere_nodes=grid.Ns,
        so3_atoms=grid.M,
        elapsed_seconds=elapsed,
    )


def _alignment_voxel_size(value, name):
    """Validate one isotropic sampling value in Angstrom per voxel."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _load_alignment_volume(path, pixel_size=None):
    """Read a cubic map, retaining header sampling beside an optional override."""
    volume, header_pixel = load_volume(path)
    if len(set(volume.shape)) != 1:
        raise ValueError(f"alignment map must be cubic: {path}")
    effective_pixel = _alignment_voxel_size(
        header_pixel if pixel_size is None else pixel_size, "voxel size")
    return volume, effective_pixel, {
        "header_pixel_size_angstrom": float(header_pixel),
        "effective_pixel_size_angstrom": effective_pixel,
        "pixel_size_overridden": pixel_size is not None,
        "pixel_size_source": "MRC header" if pixel_size is None else "explicit override",
        "side": int(volume.shape[0]),
        "effective_box_size_angstrom": float(volume.shape[0] * effective_pixel),
        "source_header_modified": False,
    }


def _fourier_resample_real(volume, target_side):
    """Resample a centered real cube by separable Fourier interpolation."""
    volume = np.asarray(volume, dtype=np.float64)
    if (volume.ndim != 3 or len(set(volume.shape)) != 1
            or not np.all(np.isfinite(volume))):
        raise ValueError("Fourier resampling requires a finite real cube")
    target_side = int(target_side)
    if target_side <= 0:
        raise ValueError("Fourier-resampling target side must be positive")
    if volume.shape[0] == target_side:
        return np.array(volume, copy=True), 0.0

    result = np.fft.ifftshift(volume)
    for axis in range(3):
        result = signal.resample(result, target_side, axis=axis, window=None)
    result = np.fft.fftshift(result)
    imaginary_norm = float(np.linalg.norm(np.imag(result)))
    result = np.asarray(np.real(result), dtype=np.float64)
    relative_imaginary = imaginary_norm / max(
        float(np.linalg.norm(result)), np.finfo(float).tiny)
    if relative_imaginary > 1e-12 or not np.all(np.isfinite(result)):
        raise FloatingPointError(
            "Fourier interpolation did not produce a finite real map")
    return result, relative_imaginary


def _alignment_evaluation_grid(
        reconstruction, pixel_size, reference, reference_pixel_size):
    """Fourier-upsample a reconstruction to the original reference grid.

    The reference samples are retained exactly.  The reconstruction's field
    of view is held fixed while it is Fourier-interpolated to the reference
    side. Small box-size discrepancies are left for the bounded magnification
    fit; larger discrepancies indicate incorrect sampling metadata. No support
    mask or low-pass is applied to either evaluation map.
    """
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    for volume, name in ((reconstruction, "reconstruction"),
                         (reference, "reference")):
        if (volume.ndim != 3 or len(set(volume.shape)) != 1
                or not np.all(np.isfinite(volume))):
            raise ValueError(f"{name} must be a finite cubic real volume")
    pixel_size = _alignment_voxel_size(
        pixel_size, "reconstruction voxel size")
    reference_pixel_size = _alignment_voxel_size(
        reference_pixel_size, "reference voxel size")

    source_side = int(reconstruction.shape[0])
    reference_side = int(reference.shape[0])
    source_box = source_side * pixel_size
    reference_box = reference_side * reference_pixel_size
    relative_box_difference = abs(source_box - reference_box) / max(
        source_box, reference_box)
    if relative_box_difference > _ALIGNMENT_BOX_SIZE_RTOL:
        raise ValueError(
            "reconstruction and reference physical box sizes are incompatible: "
            f"{source_box:.6g} A versus {reference_box:.6g} A "
            f"({100 * relative_box_difference:.2f}% difference; maximum "
            f"{100 * _ALIGNMENT_BOX_SIZE_RTOL:.1f}%). Check the MRC voxel-size "
            "headers or provide --voxel-size/--reference-voxel-size overrides."
        )

    if source_side > reference_side:
        raise ValueError(
            "reconstruction side exceeds the original reference side; "
            "alignment will not downsample the reconstruction or alter the "
            "native reference")
    coarse = source_side < reference_side
    target_side = reference_side
    interpolation_pixel = source_box / target_side
    # Both arrays are evaluated and saved on the original reference physical
    # grid. A small source/reference box mismatch is absorbed by the bounded
    # magnification fit rather than by relabeling the reference sampling.
    target_pixel = reference_pixel_size
    reconstruction_evaluation, reconstruction_imaginary = \
        _fourier_resample_real(reconstruction, target_side)
    reference_evaluation = np.array(reference, dtype=np.float64, copy=True)
    reference_imaginary = 0.0

    reconstruction_resampled = source_side != target_side
    maximum_supported_shell = (
        source_side // 2 - 1
        if reconstruction_resampled and source_side % 2 == 0
        else source_side // 2
    )

    provenance = {
        "applied": bool(reconstruction_resampled),
        "method": (
            "centered separable Fourier interpolation to the native reference side"
            if reconstruction_resampled else "none (native/reference grid retained)"
        ),
        "evaluation_grid_rule": (
            "use the original native reference side; Fourier-upsample only "
            "the reconstruction and never downsample the reference"
        ),
        "coarse_reconstruction": bool(coarse),
        "source_side": source_side,
        "source_pixel_size_angstrom": float(pixel_size),
        "source_box_size_angstrom": float(source_box),
        "reference_side": reference_side,
        "reference_pixel_size_angstrom": float(reference_pixel_size),
        "reference_box_size_angstrom": float(reference_box),
        "source_to_reference_box_size_ratio": float(
            source_box / reference_box),
        "relative_box_size_difference": float(relative_box_difference),
        "box_size_compatibility_tolerance_fraction": float(
            _ALIGNMENT_BOX_SIZE_RTOL),
        "target_side": target_side,
        "target_pixel_size_angstrom": float(target_pixel),
        "target_box_size_angstrom": float(target_side * target_pixel),
        "fourier_interpolation_pixel_size_angstrom": float(
            interpolation_pixel),
        "fourier_interpolation_box_size_angstrom": float(source_box),
        "reference_target_pixel_size_angstrom": float(
            reference_pixel_size),
        "reference_target_box_size_angstrom": float(
            reference_box),
        "fourier_interpolation_field_of_view_preserved": True,
        "placed_on_original_reference_physical_grid": True,
        "physical_grid_scale_reconciled_by_alignment_magnification": bool(
            relative_box_difference > 0),
        "relative_imaginary_norm": float(reconstruction_imaginary),
        "reference_fourier_resampled": False,
        "reference_relative_imaginary_norm": float(reference_imaginary),
        "reference_antialiasing": "not required; original samples retained",
        "original_native_reference_retained": True,
        "support_mask_applied": False,
        "evaluation_lowpass_applied": False,
        "new_fourier_information_created": False,
        "source_bandlimit": {
            "maximum_supported_fsc_shell": int(maximum_supported_shell),
            "definition": (
                "floor(source_side/2)-1 for an upsampled even-grid ball; "
                "shells above this value are marked unavailable"
            ),
        },
    }
    reference_provenance = {
        "method": "none (original native reference retained exactly)",
        "source_shape": list(reference.shape),
        "source_pixel_size": float(reference_pixel_size),
        "source_box_size_angstrom": float(reference_box),
        "target_shape": [target_side] * 3,
        "target_pixel_size": float(reference_pixel_size),
        "target_box_size_angstrom": float(reference_box),
        "coordinate_scale_output_to_input": 1.0,
        "physical_field_of_view_preserved": True,
        "relative_imaginary_norm": float(reference_imaginary),
        "original_native_samples_retained": True,
        "support_mask_applied": False,
        "evaluation_lowpass_applied": False,
        "antialiasing": provenance["reference_antialiasing"],
    }
    return (reconstruction_evaluation, reference_evaluation, target_pixel,
            provenance, reference_provenance)


def _resample_reference_to_particle_box(
        volume, source_pixel, target_side, target_pixel):
    """Center-resample a cubic reference into the physical particle box."""

    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError("evaluation reference must be a cubic 3-D map")
    source_pixel = _alignment_voxel_size(source_pixel, "reference voxel size")
    target_pixel = _alignment_voxel_size(target_pixel, "target voxel size")
    scale = float(target_pixel / source_pixel)
    source_center = (np.asarray(volume.shape, dtype=float) - 1.0) / 2.0
    target_center = (np.full(3, target_side, dtype=float) - 1.0) / 2.0
    result = ndimage.affine_transform(
        volume, np.eye(3) * scale,
        offset=source_center - scale * target_center,
        output_shape=(target_side,) * 3,
        order=3, mode="constant", cval=0.0, prefilter=True,
    )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(
            "reference resampling produced nonfinite values")
    return result, dict(
        method="center-preserving cubic-spline resampling in real space",
        source_shape=list(volume.shape),
        source_pixel_size=float(source_pixel),
        source_box_size_angstrom=float(volume.shape[0] * source_pixel),
        target_shape=[target_side] * 3,
        target_pixel_size=float(target_pixel),
        target_box_size_angstrom=float(target_side * target_pixel),
        coordinate_scale_output_to_input=scale,
    )


def _hard_lowpass(volume, pixel_size, resolution):
    if not np.isfinite(resolution) or resolution <= 2.0 * pixel_size:
        raise ValueError("comparison low-pass must be above Nyquist resolution")
    side = volume.shape[0]
    coordinates = np.arange(side, dtype=float) - side // 2
    kx, ky, kz = np.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij")
    cutoff = side * pixel_size / resolution
    mask = kx * kx + ky * ky + kz * kz <= cutoff * cutoff
    filtered = np.real(centered_ifftn(centered_fftn(volume) * mask))
    return filtered, dict(
        type="spherical hard Fourier low-pass",
        resolution_angstrom=float(resolution),
        cutoff_shell_cycles_per_box=float(cutoff),
        retained_fourier_voxels=int(mask.sum()),
        maximum_fully_retained_fsc_shell=int(math.floor(cutoff)),
    )


def _correlation(first, second):
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return -np.inf
    return float(np.sum(first * second) / denominator)


def _alignment_search_side(side, pixel_size, lowpass_resolution):
    """Choose a compact grid that still resolves the alignment bandwidth.

    The physical box is unchanged by Fourier resampling.  A search grid must
    therefore resolve the retained cutoff ``box_size / lowpass_resolution``.
    The 2.5 samples-per-cycle margin keeps the cutoff comfortably below the
    coarse Nyquist boundary.  Typical 80-A comparisons of native cryo-EM maps
    use a 32--48 voxel search grid.
    """
    cutoff = side * pixel_size / lowpass_resolution
    required = int(math.ceil(_ALIGNMENT_SEARCH_OVERSAMPLING * cutoff))
    if required % 2:
        required += 1
    target = max(_ALIGNMENT_SEARCH_MIN_SIDE, required)
    return min(int(side), target)


def _alignment_search_volumes(
        reference, reconstruction, pixel_size, lowpass_resolution):
    """Fourier-resample two low-pass maps onto the alignment-search grid."""
    if reference.shape != reconstruction.shape:
        raise ValueError("alignment volumes must have the same shape")
    side = reference.shape[0]
    search_side = _alignment_search_side(
        side, pixel_size, lowpass_resolution)
    if search_side == side:
        reference_search = reference
        reconstruction_search = reconstruction
    else:
        reference_search = fourier_crop(reference, search_side)
        reconstruction_search = fourier_crop(reconstruction, search_side)
    search_pixel = pixel_size * side / search_side
    return reference_search, reconstruction_search, dict(
        method=(
            "native grid" if search_side == side
            else "centered Fourier resampling at fixed physical field of view"
        ),
        native_side=int(side),
        native_pixel_size_angstrom=float(pixel_size),
        search_side=int(search_side),
        search_pixel_size_angstrom=float(search_pixel),
        physical_box_size_angstrom=float(side * pixel_size),
        retained_cutoff_cycles_per_box=float(
            side * pixel_size / lowpass_resolution),
        cutoff_fraction_of_search_nyquist=float(
            2.0 * side * pixel_size
            / (lowpass_resolution * search_side)),
        search_transform_interpolation="cubic spline",
        final_transform_interpolation="cubic spline on native grid",
    )


_RIGID_REFINEMENT_LADDER = (40.0, 20.0, 12.0, 8.0)
_RIGID_REFINEMENT_BEAM = (6, 3, 2, 1)


def _rigid_refinement_side(side, pixel_size, resolution):
    """Smallest even Fourier-cropped side that still carries `resolution`."""
    needed = int(math.ceil(2.0 * side * pixel_size / resolution))
    return int(min(side, max(24, needed + (needed % 2))))


def _coarse_orientation_candidates(reference, moving, n_grid=12, top_k=6,
                                   separation_degrees=25.0):
    """Top distinct grid orientations per handedness, scored by NCC.

    The coarse scan runs on a heavily low-passed grid where pseudo-symmetric
    envelopes score almost alike, so keeping only its argmax can lock the
    comparison into a wrong basin that no local refinement escapes. Several
    well-separated candidates are returned instead and resolved later at
    higher resolution.
    """
    from scipy.spatial.transform import Rotation

    grid = _orientation_grid(n_grid)
    candidates = []
    for reflected in (False, True):
        volume = reflect_volume(moving) if reflected else moving
        scores = np.array([_ncc(reference, rotate_volume_spline(volume, R))
                           for R in grid])
        order = np.argsort(scores)[::-1]
        kept = []
        for index in order:
            rotation = grid[index]
            if any(Rotation.from_matrix(rotation @ other.T).magnitude()
                   < np.radians(separation_degrees) for other in kept):
                continue
            kept.append(rotation)
            candidates.append((float(scores[index]), rotation, bool(reflected)))
            if len(kept) >= top_k:
                break
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _refine_rigid_alignment(
        reference, candidates, pixel_size, ladder=_RIGID_REFINEMENT_LADDER,
        beam=_RIGID_REFINEMENT_BEAM, maxfev=600, refine_scale=True,
        scale_bound=0.05, bandlimit_angstrom=None):
    """Beam-refine rigid candidates up a resolution ladder.

    ``candidates`` are ``(rotation, translation, magnification)`` triples on
    the native grid. Every stage Fourier-crops both maps to the smallest grid
    carrying its resolution, refines rotation, translation and (optionally)
    an isotropic scale, then keeps the best few candidates for the next, finer
    stage. One degree of residual tilt already destroys agreement beyond
    ``radius * angle``, roughly 8 A for a 150 A radius, and a one-percent
    scale error does the same, so both must be fitted against high-resolution
    content rather than at the search low-pass.
    """
    from scipy.spatial.transform import Rotation

    side = reference.shape[0]
    # Stages finer than what the moving map can carry would only fit noise.
    floor = 2.5 * pixel_size
    if bandlimit_angstrom is not None:
        floor = max(floor, float(bandlimit_angstrom) / 1.5)
    ladder = tuple(value for value in ladder if value > floor) or (
        (max(ladder[0], floor * 1.5),))
    state = [dict(index=index, rotation=np.asarray(rotation, dtype=float),
                  translation=np.asarray(translation, dtype=float),
                  magnification=float(magnification), moving=moving,
                  correlation=-np.inf, history=[])
             for index, (rotation, translation, magnification, moving)
             in enumerate(candidates)]
    stages = []
    for index, resolution in enumerate(ladder):
        keep = beam[min(index, len(beam) - 1)]
        stage_side = _rigid_refinement_side(side, pixel_size, resolution)
        scale = stage_side / side
        reference_stage = (reference if stage_side == side
                           else fourier_crop(reference, stage_side))
        reference_stage = reference_stage - reference_stage.mean()
        for item in state:
            moving_stage = (item["moving"] if stage_side == side
                            else fourier_crop(item["moving"], stage_side))

            def objective(parameters, moving_stage=moving_stage, item=item):
                delta = Rotation.from_rotvec(parameters[:3]).as_matrix()
                relative = (float(np.clip(parameters[6], -scale_bound,
                                          scale_bound))
                            if refine_scale else 0.0)
                moved = similarity_transform_volume(
                    moving_stage, delta @ item["rotation"],
                    item["magnification"] * (1.0 + relative),
                    item["translation"] * scale + parameters[3:6], order=3)
                return -_correlation(reference_stage, moved - moved.mean())

            start = objective(np.zeros(7))
            result = minimize(objective, np.zeros(7), method="Nelder-Mead",
                              options=dict(maxfev=maxfev, xatol=1e-4,
                                           fatol=1e-9))
            improved = result.fun < start
            best = result.x if improved else np.zeros(7)
            item["rotation"] = (Rotation.from_rotvec(best[:3]).as_matrix()
                                @ item["rotation"])
            item["translation"] = item["translation"] + best[3:6] / scale
            if refine_scale:
                item["magnification"] *= 1.0 + float(
                    np.clip(best[6], -scale_bound, scale_bound))
            item["correlation"] = float(-min(result.fun, start))
            item["history"].append(dict(
                resolution_angstrom=float(resolution),
                grid_side=int(stage_side),
                correlation_before=float(-start),
                correlation_after=item["correlation"],
                accepted=bool(improved)))
        state.sort(key=lambda item: item["correlation"], reverse=True)
        stages.append(dict(
            resolution_angstrom=float(resolution), grid_side=int(stage_side),
            candidates_in=len(state), candidates_kept=int(min(keep, len(state))),
            correlations=[round(item["correlation"], 4) for item in state]))
        state = state[:max(1, keep)]
    winner = state[0]
    return winner, dict(
        model=("multi-candidate beam refinement of rotation, translation and "
               "isotropic scale up a resolution ladder"),
        ladder_angstrom=[float(value) for value in ladder],
        beam_widths=[int(value) for value in beam],
        scale_refined=bool(refine_scale),
        final_correlation=float(winner["correlation"]),
        final_magnification=float(winner["magnification"]),
        stages=stages,
        winner_history=winner["history"],
    )


def _scale_and_translation_score(reference, moving, magnification):
    """Register one isotropically scaled candidate and return its NCC."""
    scaled = similarity_transform_volume(
        moving, np.eye(3), magnification, np.zeros(3), order=3)
    translation, registration_error, phase = phase_cross_correlation(
        reference, scaled,
        upsample_factor=_ALIGNMENT_TRANSLATION_UPSAMPLE,
        normalization=None,
    )
    translation = np.asarray(translation, dtype=float)
    shifted = ndimage.shift(
        scaled, translation, order=3, mode="constant", cval=0.0,
        prefilter=True,
    )
    shifted -= shifted.mean()
    return (
        _ncc(reference, shifted), translation,
        float(registration_error), float(phase),
    )


def _estimate_scale_and_translation(
        reference, moving, scale_bounds=_ALIGNMENT_SCALE_BOUNDS):
    """Estimate bounded magnification and subpixel translation.

    A short deterministic scale scan prevents a bounded scalar optimizer from
    settling on the wrong local peak.  At each scale, translation is solved by
    upsampled Fourier cross-correlation.  The final scalar search is restricted
    to the interval neighboring the best scan point.
    """
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if (reference.shape != moving.shape or reference.ndim != 3
            or len(set(reference.shape)) != 1):
        raise ValueError(
            "scale/translation estimation requires equal cubic volumes")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(moving)):
        raise ValueError("alignment volumes must be finite")
    lower, upper = map(float, scale_bounds)
    if not 0.0 < lower < upper:
        raise ValueError("invalid isotropic-scale bounds")

    reference = reference - reference.mean()
    moving = moving - moving.mean()
    scales = np.linspace(lower, upper, _ALIGNMENT_SCALE_GRID_SIZE)
    scan = [
        _scale_and_translation_score(reference, moving, scale)
        for scale in scales
    ]
    best_index = int(np.argmax([item[0] for item in scan]))
    left = scales[max(0, best_index - 1)]
    right = scales[min(len(scales) - 1, best_index + 1)]
    if left == right:
        best_scale = float(scales[best_index])
    else:
        optimum = minimize_scalar(
            lambda scale: -_scale_and_translation_score(
                reference, moving, scale)[0],
            bounds=(float(left), float(right)), method="bounded",
            options={"xatol": 1e-5, "maxiter": 80},
        )
        best_scale = float(optimum.x)
    score, translation, error, phase = _scale_and_translation_score(
        reference, moving, best_scale)
    return dict(
        correlation=float(score),
        magnification=best_scale,
        translation=translation,
        phase_correlation_error=error,
        phase_difference_radians=phase,
        scale_scan_best=float(scales[best_index]),
        scale_scan_best_correlation=float(scan[best_index][0]),
        scale_bounds=[lower, upper],
        scale_grid_size=int(_ALIGNMENT_SCALE_GRID_SIZE),
        translation_upsample_factor=int(
            _ALIGNMENT_TRANSLATION_UPSAMPLE),
    )


def _similarity_refinement_volumes(
        reference, reconstruction, pixel_size, coarse_lowpass_resolution):
    """Build the low/intermediate-frequency grid for similarity refinement."""
    # Twenty Angstrom retains enough shape detail for subpixel registration
    # while remaining well below the reliable resolution of the initial maps.
    # Four native pixels keeps this refinement safely below Nyquist for data
    # sets with unusually coarse sampling.
    resolution = max(
        4.0 * pixel_size,
        min(float(coarse_lowpass_resolution),
            _ALIGNMENT_SIMILARITY_RESOLUTION),
    )
    reference_lp, description = _hard_lowpass(
        reference, pixel_size, resolution)
    reconstruction_lp, _ = _hard_lowpass(
        reconstruction, pixel_size, resolution)
    reference_search, reconstruction_search, grid = \
        _alignment_search_volumes(
            reference_lp, reconstruction_lp, pixel_size, resolution)
    description.update(
        purpose=("bounded isotropic-scale and subpixel-translation "
                 "estimation only; not applied to saved maps or FSC"),
    )
    return reference_search, reconstruction_search, description, grid


def _evaluate_reconstruction_volume(
        reconstruction, *, output_dir, pixel_size, reference_map,
        lowpass_resolution, allow_sign, label, ball_project_reference,
        reference_pixel_size=None, fourier_upsample_to_reference=False):
    """Align one volume and compare the original, unfiltered maps.

    Low-pass copies are used only to estimate rotation, handedness, optional
    global sign, isotropic magnification, and translation.  The selected
    similarity transform is applied once to the unfiltered evaluation map.
    For Fourier-upsampled reconstructions, FSC shells beyond the original
    source bandlimit are explicitly unavailable rather than interpreted as
    newly recovered information.
    """
    output = Path(output_dir)
    reference_map = Path(reference_map).resolve(strict=True)
    pixel_size = _alignment_voxel_size(pixel_size, "reconstruction voxel size")
    reconstruction_native = np.array(
        reconstruction, dtype=np.float64, copy=True)
    if (reconstruction_native.ndim != 3
            or len(set(reconstruction_native.shape)) != 1
            or not np.all(np.isfinite(reconstruction_native))):
        raise ValueError("reconstruction must be a finite cubic real volume")
    side = reconstruction_native.shape[0]
    native, reference_pixel, reference_sampling = _load_alignment_volume(
        reference_map, reference_pixel_size)
    if fourier_upsample_to_reference:
        (reconstruction_native, reference_native, pixel_size,
         reconstruction_resampling, resampling) = _alignment_evaluation_grid(
             reconstruction_native, pixel_size,
             native.transpose(2, 1, 0), reference_pixel)
        side = reconstruction_native.shape[0]
        imaginary_relative = float(
            resampling["relative_imaginary_norm"])
    else:
        reconstruction_resampling = {
            "applied": False,
            "method": "disabled for internal post-fit evaluation",
            "source_side": int(side),
            "source_pixel_size_angstrom": float(pixel_size),
            "source_box_size_angstrom": float(side * pixel_size),
            "target_side": int(side),
            "target_pixel_size_angstrom": float(pixel_size),
            "target_box_size_angstrom": float(side * pixel_size),
            "physical_field_of_view_preserved": True,
            "support_mask_applied": False,
            "evaluation_lowpass_applied": False,
            "new_fourier_information_created": False,
        }
        imaginary_relative = 0.0
    if not fourier_upsample_to_reference:
        reference_mrc, resampling = _resample_reference_to_particle_box(
            native, reference_pixel, side, pixel_size)
        if ball_project_reference:
            reference_fourier = centered_fftn(
                reference_mrc.transpose(2, 1, 0))
            coordinates = np.arange(side, dtype=float) - side // 2
            kx, ky, kz = np.meshgrid(
                coordinates, coordinates, coordinates, indexing="ij")
            radius = side / 2 - 1 if side % 2 == 0 else (side - 1) / 2
            reference_fourier *= kx*kx + ky*ky + kz*kz <= radius*radius
            reference_complex = centered_ifftn(reference_fourier)
            reference_native = np.real(reference_complex)
            imaginary_relative = float(
                np.linalg.norm(reference_complex.imag)
                / max(np.linalg.norm(reference_native), np.finfo(float).tiny))
            if imaginary_relative > 1e-10:
                raise FloatingPointError("ball-projected reference is not real")
        else:
            reference_native = np.asarray(
                reference_mrc.transpose(2, 1, 0), dtype=np.float64)
            imaginary_relative = 0.0
    means = dict(
        reference=float(reference_native.mean()),
        reconstruction=float(reconstruction_native.mean()),
    )
    reference_centered = reference_native - means["reference"]
    reconstruction_centered = (
        reconstruction_native - means["reconstruction"])

    # Filtering is restricted to transform estimation.  Nothing below this
    # search block uses the filtered arrays for output or evaluation.
    reference_lp, alignment_lowpass = _hard_lowpass(
        reference_centered, pixel_size, lowpass_resolution)
    reconstruction_lp, _ = _hard_lowpass(
        reconstruction_centered, pixel_size, lowpass_resolution)

    reference_search, reconstruction_search, alignment_search = \
        _alignment_search_volumes(
            reference_lp, reconstruction_lp, pixel_size,
            lowpass_resolution)

    (reference_similarity, reconstruction_similarity,
     similarity_lowpass, similarity_grid) = _similarity_refinement_volumes(
         reference_centered, reconstruction_centered, pixel_size,
         lowpass_resolution)

    refinement_side = reference_similarity.shape[0]
    candidates = []
    starts = []
    for sign in ((1.0, -1.0) if allow_sign else (1.0,)):
        for o3_correlation, rotation, reflected in \
                _coarse_orientation_candidates(
                    reference_search, sign * reconstruction_search):
            candidate = sign * reconstruction_similarity
            if reflected:
                candidate = reflect_volume(candidate)
            candidate = rotate_volume_spline(candidate, rotation)
            similarity = _estimate_scale_and_translation(
                reference_similarity, candidate)
            transform_input = sign * reconstruction_native
            if reflected:
                transform_input = reflect_volume(transform_input)
            starts.append((
                rotation,
                np.asarray(similarity["translation"], dtype=float)
                * side / refinement_side,
                float(similarity["magnification"]), transform_input))
            candidates.append((
                float(similarity["correlation"]), rotation, bool(reflected),
                sign, float(o3_correlation), similarity))
    order = np.argsort([-candidate[0] for candidate in candidates])
    candidates = [candidates[index] for index in order]
    starts = [starts[index] for index in order]
    keep = int(_RIGID_REFINEMENT_BEAM[0])
    source_bandlimit = None
    if reconstruction_resampling["applied"]:
        supported = int(reconstruction_resampling["source_bandlimit"]
                        ["maximum_supported_fsc_shell"])
        if supported > 0:
            source_bandlimit = side * pixel_size / supported
    winner, rigid_refinement = _refine_rigid_alignment(
        reference_centered, starts[:keep], pixel_size,
        bandlimit_angstrom=source_bandlimit)
    rigid_refinement["source_bandlimit_angstrom"] = source_bandlimit
    chosen = int(winner["index"])
    (search_correlation, _, reflected, sign, o3_correlation, similarity) = \
        candidates[chosen]
    rigid_refinement["candidates_scored"] = len(candidates)
    rigid_refinement["coarse_rank_of_winner"] = chosen
    rotation = winner["rotation"]
    translation_native = winner["translation"]
    magnification = float(winner["magnification"])

    # Reflection is an exact permutation.  Apply the interpolating rotation,
    # magnification, and translation together, once, on the unfiltered map.
    aligned_native = similarity_transform_volume(
        starts[chosen][3], rotation, magnification, translation_native,
        order=3)
    aligned_centered = aligned_native - aligned_native.mean()
    correlation = _correlation(reference_centered, aligned_centered)
    shells, curve = fsc(reference_centered, aligned_centered)
    curve[0] = np.nan
    if reconstruction_resampling["applied"]:
        maximum_supported_shell = int(
            reconstruction_resampling["source_bandlimit"]
            ["maximum_supported_fsc_shell"])
        curve[shells > maximum_supported_shell] = np.nan
    else:
        maximum_supported_shell = int(shells[-1])
    finite_shells = shells[(shells >= 1) & np.isfinite(curve)]
    alignment_lowpass.update(
        purpose=("rotation, handedness, and optional sign estimation only; "
                 "not applied to saved maps, correlation, or FSC"),
    )
    fsc_05 = fsc_resolution(shells, curve, pixel_size, side)
    if reconstruction_resampling["applied"]:
        fsc_domain = (
            "the original native-reference grid within the original "
            "reconstruction bandlimit")
    else:
        fsc_domain = "the full original native-reference frequency range"
    fsc_status = (
        f"0.5 crossing measured on {fsc_domain}"
        if fsc_05 is not None else f"no 0.5 crossing on {fsc_domain}")
    save_mrc(
        reference_native.transpose(2, 1, 0),
        output / "reference_particle_box.mrc", pixel_size,
    )
    save_mrc(
        aligned_native.transpose(2, 1, 0),
        output / "recon_aligned.mrc", pixel_size,
    )
    np.savez(
        output / "fsc_full.npz",
        k=shells, fsc=curve,
        spatial_frequency=shells / (side * pixel_size),
        fsc_05_angstrom=np.nan if fsc_05 is None else fsc_05,
        fsc_05_status=fsc_status,
        maximum_fsc_shell=int(shells[-1]),
        maximum_supported_fsc_shell=maximum_supported_shell,
        finite_fsc_shells=finite_shells,
        evaluation_sign=sign,
        full_frequency_comparison=(
            not reconstruction_resampling["applied"]),
        gold_standard_fsc=False,
    )
    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    curve_label = (
        f"{label} ({fsc_05:.1f} Å)" if fsc_05 is not None
        else f"{label} (no 0.5 crossing)")
    axis.plot(shells / (side * pixel_size), curve, label=curve_label)
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Spatial frequency (Å⁻¹)", ylabel="FSC",
        ylim=(-1.05, 1.05),
    )
    axis.legend(frameon=False, loc="upper right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "fsc_full.pdf")
    plt.close(figure)
    resampling.update(
        source_map=str(reference_map),
        source_sha256=_sha256(reference_map),
        working_side=side,
        working_pixel_size=pixel_size,
        reference_ball_projected=bool(ball_project_reference),
        ball_projection_imaginary_relative_norm=imaginary_relative,
        source_sampling=reference_sampling,
    )
    similarity_transform = dict(
        model=("output[y] = reflected_signed_input[center + R.T @ "
               "(y - center - translation) / magnification]"),
        reflection_applied_before_interpolation=bool(reflected),
        interpolation=("one cubic-spline affine resampling on the original "
                       "unfiltered evaluation-grid map"),
        isotropic_magnification=magnification,
        output_to_input_coordinate_scale=1.0 / magnification,
        scale_bounds=similarity["scale_bounds"],
        scale_grid_size=similarity["scale_grid_size"],
        scale_scan_best=similarity["scale_scan_best"],
        scale_scan_best_correlation=(
            similarity["scale_scan_best_correlation"]),
        translation_voxels_array_axes=translation_native.tolist(),
        translation_angstrom_array_axes=(
            translation_native * pixel_size).tolist(),
        translation_search_grid_voxels=(
            np.asarray(similarity["translation"], dtype=float).tolist()),
        translation_convention=("positive values move map contents toward "
                                "increasing array indices"),
        phase_correlation_error=similarity["phase_correlation_error"],
        phase_difference_radians=similarity["phase_difference_radians"],
        translation_upsample_factor=(
            similarity["translation_upsample_factor"]),
        estimation_lowpass=similarity_lowpass,
        estimation_grid=similarity_grid,
        fit_correlation=float(search_correlation),
    )
    return dict(
        correlation=correlation,
        search_correlation=float(search_correlation),
        o3_search_correlation=float(o3_correlation),
        reflected=reflected,
        rotation=rotation.tolist(),
        isotropic_magnification=magnification,
        translation_voxels_array_axes=translation_native.tolist(),
        translation_angstrom_array_axes=(
            translation_native * pixel_size).tolist(),
        evaluation_sign=sign,
        fsc_05_angstrom=fsc_05,
        fsc_05_status=fsc_status,
        alignment_lowpass=alignment_lowpass,
        full_frequency_comparison=(
            not reconstruction_resampling["applied"]),
        comparison_frequency_domain=(
            "source-bandlimited native-reference-grid FSC"
            if reconstruction_resampling["applied"] else
            "full original native-reference frequency range"),
        source_bandlimit_enforced=bool(reconstruction_resampling["applied"]),
        maximum_supported_fsc_shell=maximum_supported_shell,
        evaluation_dc_excluded=True,
        original_real_space_means=means,
        gold_standard_fsc=False,
        reference_resampling=resampling,
        reconstruction_fourier_upsampling=reconstruction_resampling,
        reference_sampling=reference_sampling,
        alignment_search=alignment_search,
        similarity_transform=similarity_transform,
        rigid_refinement=rigid_refinement,
    )


def evaluate_reference(
        param, theta, order, output_dir, pixel_size, reference_map,
        lowpass_resolution, side):
    """Backward-compatible post-fit evaluator; reconstruction no longer calls it."""
    reconstruction = param.volume(theta)
    if reconstruction.shape != (side, side, side):
        raise ValueError("parameterized reconstruction has the wrong side length")
    return _evaluate_reconstruction_volume(
        reconstruction,
        output_dir=output_dir,
        pixel_size=pixel_size,
        reference_map=reference_map,
        lowpass_resolution=lowpass_resolution,
        allow_sign=(order == 2),
        label=f"M{order}",
        ball_project_reference=True,
    )


def align_map_to_reference(
        map_path, reference_map, output_dir, lowpass_resolution,
        *, allow_sign=False, label="reconstruction", pixel_size=None,
        reference_pixel_size=None):
    """Align unfiltered maps using header sampling or independent overrides.

    Sampling overrides describe physical voxel spacing; source MRC files are
    never edited. A coarse reconstruction is Fourier-upsampled at fixed field
    of view to the unchanged native reference side. Saved maps and FSC use
    that grid without a support mask or evaluation low-pass.
    """
    map_path = Path(map_path).expanduser().resolve(strict=True)
    reference_map = Path(reference_map).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("alignment output directory must be new or empty")
    for value, name in ((pixel_size, "reconstruction voxel size"),
                        (reference_pixel_size, "reference voxel size")):
        if value is not None:
            _alignment_voxel_size(value, name)
    reconstruction_mrc, effective_pixel, input_sampling = _load_alignment_volume(
        map_path, pixel_size)
    output.mkdir(parents=True, exist_ok=True)
    evaluation = _evaluate_reconstruction_volume(
        reconstruction_mrc.transpose(2, 1, 0),
        output_dir=output,
        pixel_size=effective_pixel,
        reference_map=reference_map,
        reference_pixel_size=reference_pixel_size,
        lowpass_resolution=lowpass_resolution,
        allow_sign=bool(allow_sign),
        label=label,
        ball_project_reference=False,
        fourier_upsample_to_reference=True,
    )
    report = dict(
        complete=True,
        map=str(map_path),
        map_sha256=_sha256(map_path),
        reference_map=str(reference_map),
        reference_map_sha256=_sha256(reference_map),
        label=str(label),
        input_side=int(reconstruction_mrc.shape[0]),
        input_pixel_size_angstrom=effective_pixel,
        input_header_pixel_size_angstrom=input_sampling["header_pixel_size_angstrom"],
        input_sampling=input_sampling,
        evaluation_side=int(
            evaluation["reconstruction_fourier_upsampling"]["target_side"]),
        evaluation_pixel_size_angstrom=float(
            evaluation["reconstruction_fourier_upsampling"]
            ["target_pixel_size_angstrom"]),
        reconstruction_fourier_upsampling=(
            evaluation["reconstruction_fourier_upsampling"]),
        reference_sampling=evaluation["reference_sampling"],
        alignment_search_lowpass_angstrom=float(lowpass_resolution),
        alignment_model=("global O(3) orientation and handedness search, "
                         "then bounded isotropic magnification and subpixel "
                         "translation, then local six-parameter rigid "
                         "refinement up the resolution ladder"),
        handedness="global O(3) search",
        sign_search=bool(allow_sign),
        evaluation=evaluation,
    )
    _write_json(output / "alignment.json", report)
    return report


def _save_viewing_plot(output, density, beta):
    """Plot the fitted SH density on a dense sphere grid, in sr^-1."""

    longitude_edges = np.linspace(-np.pi, np.pi, 361)
    latitude_edges = np.linspace(-np.pi / 2, np.pi / 2, 181)
    longitude, latitude = np.meshgrid(
        (longitude_edges[:-1] + longitude_edges[1:]) / 2,
        (latitude_edges[:-1] + latitude_edges[1:]) / 2,
    )
    directions = np.column_stack((
        (np.cos(latitude) * np.cos(longitude)).ravel(),
        (np.cos(latitude) * np.sin(longitude)).ravel(),
        np.sin(latitude).ravel(),
    ))
    probability_density = density.density(beta, directions).reshape(
        latitude.shape) / (4 * np.pi)
    figure, axis = plt.subplots(
        subplot_kw={"projection": "mollweide"}, figsize=(6.5, 3.8))
    surface = axis.pcolormesh(
        longitude_edges, latitude_edges, probability_density,
        cmap="viridis", shading="flat", rasterized=True, vmin=0,
    )
    axis.set_longitude_grid(60)
    axis.set_latitude_grid(30)
    axis.grid(alpha=0.2)
    figure.colorbar(
        surface, ax=axis, label="Probability density (sr$^{-1}$)", shrink=0.75)
    figure.tight_layout()
    figure.savefig(output / "viewing_density.pdf")
    plt.close(figure)


def _aligned_viewing_transform(alignment):
    """Map oriented camera normals into the reference-aligned map frame."""

    evaluation = alignment["evaluation"]
    rotation = np.asarray(evaluation["rotation"], dtype=np.float64)
    reflected = evaluation["reflected"]
    if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8,
                               rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8,
                              rtol=0)
            or not isinstance(reflected, (bool, np.bool_))):
        raise ValueError("alignment must contain a proper rotation and a reflection flag")
    reflection = np.diag([-1.0, 1.0, 1.0]) if reflected else np.eye(3)
    map_transform = rotation @ reflection
    # A camera normal is cross(R[:, 0], R[:, 1]), hence an axial vector.
    # det(T)*T preserves a proper camera when the map is reflected by T.
    direction_transform = (-1.0 if reflected else 1.0) * map_transform
    return direction_transform


def plot_aligned_viewing_distributions(
    mom_fit_path, relion_star_path, initial_alignment, refined_alignment,
    output_dir,
):
    """Compare fitted MoM density and final RELION poses in the reference frame.

    The MoM density uses the initial map's alignment; RELION poses use the
    refined map's alignment. RELION is plotted as one point per particle
    without smoothing or symmetry expansion. When the MoM fit was formed
    with reflection augmentation (an antipodal density), each particle's
    mirror image is also plotted: mirroring a projection is the same as
    viewing the molecule from the antipodal direction, so the mirrored
    image's direction is -n. Equal-area pose histograms are retained only
    as numerical diagnostics.
    """

    from aspire.utils import RelionStarFile

    started = time.perf_counter()
    fit_path = Path(mom_fit_path).resolve(strict=True)
    star_path = Path(relion_star_path).resolve(strict=True)
    output = Path(output_dir).resolve()
    transforms = [_aligned_viewing_transform(report) for report in
                  (initial_alignment, refined_alignment)]
    reference_hashes = [report.get("reference_map_sha256") for report in
                        (initial_alignment, refined_alignment)]
    if (all(reference_hashes)
            and reference_hashes[0] != reference_hashes[1]):
        raise ValueError("both map alignments must use the same reference")

    with np.load(fit_path, allow_pickle=False) as saved:
        required = ("grid_xyz", "grid_q", "viewing_degree", "beta")
        missing = [name for name in required if name not in saved]
        if missing:
            raise ValueError(f"MoM fit is missing viewing-density fields: {missing}")
        xyz = np.asarray(saved["grid_xyz"], dtype=np.float64)
        q = np.asarray(saved["grid_q"], dtype=np.float64)
        degree = np.asarray(saved["viewing_degree"]).item()
        beta = np.asarray(saved["beta"], dtype=np.float64)
        antipodal = bool(np.asarray(
            saved["viewing_antipodal"]).reshape(())) \
            if "viewing_antipodal" in saved.files else False
    grid = SimpleNamespace(xyz=xyz, q=q, Ns=len(xyz))
    density = LowDegreeViewingDensity(grid, degree=degree,
                                      antipodal=antipodal)
    density._beta(beta)

    metadata = RelionStarFile(str(star_path)).get_merged_data_block()
    missing = [key for key in ("_rlnAngleRot", "_rlnAngleTilt")
               if key not in metadata]
    if missing:
        raise ValueError(f"RELION STAR is missing pose fields: {missing}")
    rot = np.asarray(metadata["_rlnAngleRot"], dtype=np.float64)
    tilt = np.asarray(metadata["_rlnAngleTilt"], dtype=np.float64)
    if (rot.ndim != 1 or rot.size == 0 or tilt.shape != rot.shape
            or not np.all(np.isfinite(rot)) or not np.all(np.isfinite(tilt))
            or np.any(tilt < -1e-6) or np.any(tilt > 180 + 1e-6)):
        raise ValueError("RELION poses must be finite with tilt in [0, 180] degrees")
    rot = np.deg2rad(np.mod(rot, 360.0))
    tilt = np.deg2rad(np.clip(tilt, 0, 180))
    original_directions = np.column_stack((
        np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt),
    ))
    relion_directions = original_directions @ transforms[1].T
    relion_directions /= np.linalg.norm(relion_directions, axis=1)[:, None]
    # The mirror image of a projection along n is the projection along -n
    # (up to an in-plane rotation), so reflection augmentation adds -n.
    mirrored_directions = (-relion_directions if antipodal
                           else np.empty((0, 3), dtype=np.float64))
    plotted_directions = np.vstack((relion_directions, mirrored_directions))

    longitude_edges = np.linspace(-np.pi, np.pi, 73)
    z_edges = np.linspace(-1.0, 1.0, 37)
    counts, _, _ = np.histogram2d(
        np.clip(plotted_directions[:, 2], -1.0, 1.0),
        np.arctan2(plotted_directions[:, 1], plotted_directions[:, 0]),
        bins=(z_edges, longitude_edges),
    )
    cell_area = np.diff(z_edges)[:, None] * np.diff(longitude_edges)[None, :]
    relion_density = counts / (len(plotted_directions) * cell_area)

    # Evaluate the fitted continuous SH function, not an interpolated node PMF.
    smooth_longitude_edges = np.linspace(-np.pi, np.pi, 361)
    smooth_z_edges = np.linspace(-1.0, 1.0, 181)
    longitude, z = np.meshgrid(
        (smooth_longitude_edges[:-1] + smooth_longitude_edges[1:]) / 2,
        (smooth_z_edges[:-1] + smooth_z_edges[1:]) / 2,
    )
    radius = np.sqrt(1.0 - z**2)
    reference_directions = np.column_stack((
        (radius * np.cos(longitude)).ravel(),
        (radius * np.sin(longitude)).ravel(), z.ravel(),
    ))
    mom_density = density.density(
        beta, reference_directions @ transforms[0],
    ).reshape(z.shape) / (4 * np.pi)
    mom_integral = float(q @ density.density(beta, xyz @ transforms[0]))
    relion_integral = float(np.sum(relion_density * cell_area))
    if (int(counts.sum()) != len(plotted_directions)
            or not np.isclose(mom_integral, 1.0, atol=1e-10, rtol=0)
            or not np.isclose(relion_integral, 1.0, atol=1e-12, rtol=0)):
        raise ValueError("aligned viewing distributions lost probability mass")

    output.mkdir(parents=True, exist_ok=True)
    mom_pdf = output / "viewing_density_mom_aligned.pdf"
    relion_pdf = output / "viewing_density_relion_aligned.pdf"
    data_path = output / "viewing_density_aligned.npz"
    maximum = float(mom_density.max())
    relion_title = "RELION, images and mirrors" if antipodal else "RELION"
    for pdf, title in ((mom_pdf, "MoM"), (relion_pdf, relion_title)):
        figure, axis = plt.subplots(
            subplot_kw={"projection": "mollweide"}, figsize=(6.5, 3.8),
            layout="constrained",
        )
        if pdf == mom_pdf:
            surface = axis.pcolormesh(
                smooth_longitude_edges, np.arcsin(smooth_z_edges), mom_density,
                cmap="viridis", shading="flat", rasterized=True,
                vmin=0, vmax=maximum,
            )
            figure.colorbar(surface, ax=axis, shrink=0.75,
                            label="Probability density (sr$^{-1}$)")
        else:
            axis.scatter(
                np.arctan2(relion_directions[:, 1], relion_directions[:, 0]),
                np.arcsin(np.clip(relion_directions[:, 2], -1.0, 1.0)),
                s=1.5, c="black", marker=".", alpha=1.0, linewidths=0,
                rasterized=True, label="image",
            )
            if antipodal:
                axis.scatter(
                    np.arctan2(mirrored_directions[:, 1],
                               mirrored_directions[:, 0]),
                    np.arcsin(np.clip(mirrored_directions[:, 2], -1.0, 1.0)),
                    s=1.5, c="tab:red", marker=".", alpha=1.0, linewidths=0,
                    rasterized=True, label="mirror image",
                )
                axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12),
                            ncol=2, frameon=False, markerscale=8,
                            handletextpad=0.2)
        axis.set_title(title)
        axis.set_longitude_grid(60)
        axis.set_latitude_grid(30)
        axis.grid(alpha=0.2)
        figure.savefig(pdf, dpi=300)
        plt.close(figure)
    np.savez_compressed(
        data_path,
        mom_longitude_edges=smooth_longitude_edges, mom_z_edges=smooth_z_edges,
        mom_density_per_steradian=mom_density,
        relion_longitude_edges=longitude_edges, relion_z_edges=z_edges,
        relion_counts=counts.astype(np.int64),
        relion_density_per_steradian=relion_density,
        relion_directions_aligned=relion_directions,
        relion_directions_mirrored=mirrored_directions,
        mom_direction_transform=transforms[0],
        relion_direction_transform=transforms[1],
    )
    report = dict(
        status="complete", mom_pdf=str(mom_pdf), relion_pdf=str(relion_pdf),
        data=str(data_path),
        mom_fit=str(fit_path), mom_fit_sha256=_sha256(fit_path),
        relion_star=str(star_path), relion_star_sha256=_sha256(star_path),
        reference_map=initial_alignment.get("reference_map"),
        reference_map_sha256=reference_hashes[0],
        frame="reference map XYZ axes, longitude=atan2(y,x), latitude=asin(z)",
        direction_convention="oriented camera normal; RELION A[2,:], MoM R[:,2]",
        transform_convention=("T=alignment.rotation @ diag(-1,1,1) if reflected, "
                              "otherwise T=alignment.rotation; "
                              "n_aligned=det(T)*T@n; no antipodal folding; "
                              "mirror images plotted at -n_aligned when the "
                              "MoM density is antipodal"),
        mom_direction_transform=transforms[0].tolist(),
        relion_direction_transform=transforms[1].tolist(),
        mom_viewing_degree=int(degree), relion_particle_count=int(rot.size),
        mom_plot_kind="continuous spherical-harmonic density",
        mom_antipodal=bool(antipodal),
        relion_plot_kind=("scatter of every aligned per-particle direction"
                          + (" plus the antipodal direction of its mirror image"
                             if antipodal else "")),
        relion_mirror_images_included=bool(antipodal),
        relion_scatter=dict(count=int(rot.size), color="black", marker=".",
                            mirrored_count=int(len(mirrored_directions)),
                            mirrored_color="tab:red",
                            size_points_squared=1.5, alpha=1.0,
                            rasterized_in_pdf=True, dpi=300, colorbar=False),
        relion_histogram_shape=list(counts.shape),
        relion_histogram=("equal-solid-angle longitude/z bins, no smoothing; "
                          "numerical diagnostic only, not plotted"),
        density_units="per steradian", shared_color_limits=None,
        mom_color_limits=[0.0, maximum],
        normalization=dict(mom=mom_integral, relion=relion_integral),
        populations=("MoM density fitted to the selected moment images; RELION "
                     "poses include every particle in the final refinement STAR"),
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json(output / "viewing_density_aligned.json", report)
    return report


def smoke():
    """Run inexpensive geometry, parameterization, and selection checks."""

    use_device("cpu")
    dimensions = []
    identity = np.eye(3)[None, ...]
    for side, ell_max in ((8, 2), (12, 3)):
        basis = FBMomentBasis(side, ell_max)
        param = SphFBParam(side, ell_max, identity, fb_basis=basis)
        if param.ell_max != ell_max or param.n_params < 1:
            raise AssertionError("dynamic SphFB parameterization failed")
        dimensions.append(dict(
            moment_res=side, ell_max=ell_max,
            fb_coefficients=basis.n_coefficients,
            n_volume_parameters=param.n_params,
            n_m3_entries=basis.m3_size,
        ))
    grid = so3_sphere_grid(DEFAULT_SPHERE_NODES, 1)
    viewing = {}
    for degree, antipodal, expected_parameters in (
            (0, False, 0), (2, False, 9), (0, True, 0), (2, True, 6)):
        density = LowDegreeViewingDensity(grid, degree=degree,
                                          antipodal=antipodal)
        if density.n_parameters != expected_parameters:
            raise AssertionError("dynamic viewing degree failed")
        p = density.pmf(np.zeros(density.n_parameters))
        np.testing.assert_allclose(p, grid.q, rtol=0.0, atol=1e-15)
        viewing[f"degree{degree}" + ("_antipodal" if antipodal else "")] = \
            density.n_parameters
    records = [
        {"init_index": 7, "objective": 0.2, "correlation": 0.99},
        {"init_index": 4, "objective": 0.1, "correlation": -1.0},
        {"init_index": 9, "objective": 0.1, "correlation": 1.0},
    ]
    if select_best_fit(records)["init_index"] != 4:
        raise AssertionError("best-fit selection did not use first minimum loss")
    return dict(dimensions=dimensions, viewing_parameters=viewing,
                best_init=4)
