"""RELION particle I/O, Fourier downsampling and reference-free particle selection.

Stack loading applies no STAR poses or intensity corrections; particle selection uses no poses or maps.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .moments import centered_fft2


# ===========================================================================
# RELION particle metadata and raw stacks
# ===========================================================================

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

    def __init__(self, star_path, data_folder=None, stack_prefix=None,
                 pixel_size=None):
        from aspire.utils import RelionStarFile

        if stack_prefix is not None:
            stack_prefix = os.fspath(stack_prefix)
        self.stack_prefix = stack_prefix
        self.star_path = Path(star_path).resolve(strict=True)
        root = (self.star_path.parent if data_folder is None
                else Path(data_folder).resolve(strict=True))
        self.metadata = RelionStarFile(str(self.star_path)).get_merged_data_block()
        self.image_names = np.asarray(self.metadata["_rlnImageName"], dtype=str)
        self.n = len(self.image_names)
        parts = [name.split("@", 1) for name in self.image_names]
        self.stack_indices = np.array([int(part[0]) - 1 for part in parts])
        if np.any(self.stack_indices < 0):
            raise ValueError("STAR stack indices must be one-based positive integers")
        names, self.stack_ids = np.unique([part[1] for part in parts],
                                          return_inverse=True)
        if stack_prefix is not None:
            prefix = stack_prefix + "/"
            stripped = []
            for name in names:
                relative = name[len(prefix):]
                stripped.append(relative)
            names = stripped
        self.stack_paths = [root / name for name in names]

        if "_rlnImagePixelSize" in self.metadata:
            pixel_sizes = self.column("_rlnImagePixelSize")
        elif all(key in self.metadata for key in
                 ("_rlnDetectorPixelSize", "_rlnMagnification")):
            pixel_sizes = (1e4 * self.column("_rlnDetectorPixelSize")
                           / self.column("_rlnMagnification"))
        else:
            raise ValueError("STAR must specify the particle pixel size")
        if (not np.all(np.isfinite(pixel_sizes)) or np.any(pixel_sizes <= 0)
                or not np.allclose(pixel_sizes, pixel_sizes[0], rtol=1e-7, atol=0)):
            raise ValueError("particles must have one common positive pixel size")
        self.metadata_pixel_size = float(pixel_sizes[0])
        if pixel_size is None:
            self.pixel_size = self.metadata_pixel_size
        else:
            self.pixel_size = float(pixel_size)
        self.res = None
        for stack_id, path in enumerate(self.stack_paths):
            with mrcfile.mmap(path, mode="r") as stack:
                shape = stack.data.shape
                shape = (1, *shape) if len(shape) == 2 else shape
                self.res = int(shape[1])
        if ("_rlnImageSize" in self.metadata
                and np.any(self.column("_rlnImageSize") != self.res)):
            raise ValueError("STAR image size disagrees with the MRC header")

    def column(self, name, default=None):
        """Return a finite numeric metadata column with one value per particle."""
        if name not in self.metadata:
            if default is None:
                raise ValueError(f"STAR is missing {name}")
            values = np.full(self.n, default, dtype=np.float64)
        else:
            values = np.asarray(self.metadata[name], dtype=np.float64)
        return values

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
# Reference-free low-crowding selection
# ===========================================================================

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
        n_select, *, minimum_neighbor_distance, contrast_bins=4,
        max_score_quantile=0.95, auto_score_ceiling=False,
        smooth_sigma=1.1, inner_radius=0.50,
        outer_radius_min=0.62, outer_radius_max=0.94,
        inner_top_fraction=0.20, outer_top_fraction=0.10, batch_size=256,
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
    n_select, contrast_bins, batch_size = int(n_select), int(contrast_bins), int(batch_size)

    coordinates = np.arange(side, dtype=np.float64) - (side - 1) / 2
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt(xx ** 2 + yy ** 2) / (side / 2)
    inner_mask = radius <= inner_radius
    outer_mask = (radius >= outer_radius_min) & (radius <= outer_radius_max)

    scores = np.empty(count, dtype=np.float64)
    inner_strength = np.empty(count, dtype=np.float64)
    outer_strength = np.empty(count, dtype=np.float64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        batch = np.array(images[start:end], dtype=np.float32, copy=True)
        smooth = gaussian_filter(
            batch, sigma=(0, smooth_sigma, smooth_sigma), mode="nearest")
        positive = np.maximum(smooth, 0.0)
        central = _top_rms(positive[:, inner_mask], inner_top_fraction)
        peripheral = _top_rms(positive[:, outer_mask], outer_top_fraction)
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
