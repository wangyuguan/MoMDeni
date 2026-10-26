"""RELION particle I/O, Fourier downsampling, CTFs, and particle selection.

Merged from real_data.py, ctf.py, and particle_selection.py. Stack loading
applies no STAR poses or intensity corrections; CTFs use the centered,
Fortran-flattened frequency order of the moment operators; particle selection
is reference-free and uses no poses or maps.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile

import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .moments import centered_coords_1d, centered_fft2


# ===========================================================================
# RELION particle metadata and raw stacks (was real_data.py)
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
            if (not isinstance(stack_prefix, str)
                    or os.path.isabs(stack_prefix)
                    or "\\" in stack_prefix
                    or any(part in ("", ".", "..")
                           for part in stack_prefix.split("/"))):
                raise ValueError("stack_prefix must be a nonempty relative directory "
                                 "prefix without a trailing slash or '.'/'..' parts")
        self.stack_prefix = stack_prefix
        self.star_path = Path(star_path).resolve(strict=True)
        root = (self.star_path.parent if data_folder is None
                else Path(data_folder).resolve(strict=True))
        self.metadata = RelionStarFile(str(self.star_path)).get_merged_data_block()
        required = ("_rlnImageName", "_rlnVoltage", "_rlnDefocusU",
                    "_rlnDefocusV", "_rlnDefocusAngle",
                    "_rlnSphericalAberration", "_rlnAmplitudeContrast")
        missing = [key for key in required if key not in self.metadata]
        if missing:
            raise ValueError(f"STAR is missing required fields: {missing}")
        self.image_names = np.asarray(self.metadata["_rlnImageName"], dtype=str)
        self.n = len(self.image_names)
        if self.n == 0:
            raise ValueError("STAR contains no particles")
        parts = [name.split("@", 1) for name in self.image_names]
        if any(len(part) != 2 for part in parts):
            raise ValueError("_rlnImageName must have the form 1@stack.mrcs")
        self.stack_indices = np.array([int(part[0]) - 1 for part in parts])
        if np.any(self.stack_indices < 0):
            raise ValueError("STAR stack indices must be one-based positive integers")
        names, self.stack_ids = np.unique([part[1] for part in parts],
                                          return_inverse=True)
        if stack_prefix is not None:
            prefix = stack_prefix + "/"
            stripped = []
            for name in names:
                if not name.startswith(prefix):
                    raise ValueError(f"STAR stack path {name!r} does not start with "
                                     f"the configured stack_prefix {prefix!r}")
                relative = name[len(prefix):]
                if (not relative or Path(relative).is_absolute()
                        or ".." in Path(relative).parts):
                    raise ValueError(f"unsafe STAR stack path after stripping "
                                     f"stack_prefix: {name!r}")
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
            if isinstance(pixel_size, (bool, np.bool_)):
                raise ValueError("pixel_size must be finite and positive")
            try:
                self.pixel_size = float(pixel_size)
            except (TypeError, ValueError) as exc:
                raise ValueError("pixel_size must be finite and positive") from exc
            if not np.isfinite(self.pixel_size) or self.pixel_size <= 0:
                raise ValueError("pixel_size must be finite and positive")
        self.res = None
        for stack_id, path in enumerate(self.stack_paths):
            with mrcfile.mmap(path, mode="r") as stack:
                shape = stack.data.shape
                shape = (1, *shape) if len(shape) == 2 else shape
                if len(shape) != 3 or shape[1] != shape[2]:
                    raise ValueError(f"expected square particle images in {path}")
                if np.iscomplexobj(stack.data):
                    raise ValueError(f"expected real-space particles in {path}")
                if self.res is not None and self.res != shape[1]:
                    raise ValueError("all stacks must have the same image size")
                self.res = int(shape[1])
                if np.any(self.stack_indices[self.stack_ids == stack_id] >= shape[0]):
                    raise ValueError(f"STAR references an out-of-range image in {path}")
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
        if values.shape != (self.n,) or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid numeric STAR column: {name}")
        return values

    def read(self, indices):
        """Read zero-based STAR rows in the requested order, using memory maps."""
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.n)):
            raise ValueError("indices must be a vector of valid zero-based STAR rows")
        images = np.empty((len(indices), self.res, self.res), dtype=np.float32)
        for stack_id in np.unique(self.stack_ids[indices]):
            positions = np.flatnonzero(self.stack_ids[indices] == stack_id)
            stack_indices = self.stack_indices[indices[positions]]
            with mrcfile.mmap(self.stack_paths[stack_id], mode="r") as stack:
                data = stack.data
                if data.ndim == 2:
                    data = data[None]
                images[positions] = data[stack_indices]
        if not np.all(np.isfinite(images)):
            raise ValueError("particle pixels contain NaN or infinity")
        return images

    def fingerprint(self):
        """Identify STAR contents and the referenced stacks without hashing all pixels."""
        digest = hashlib.sha256(self.star_path.read_bytes())
        for path in self.stack_paths:
            stat = path.stat()
            digest.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()

    def write_downsampled_star(self, path, stack_path, indices, ds_res):
        """Export metadata matching a saved, unaligned downsampled particle stack.

        Output rows follow ``indices`` exactly, including repeated source rows.
        Optics groups and physical CTF parameters retain their original meaning.
        Pixel-unit origins are rescaled; Angstrom origins, angles, coordinates,
        defocus, and other physical metadata are not. This only writes metadata:
        no pose, CTF correction, or contrast transformation is applied to pixels.
        The existing output stack must have the matching image count and grid.
        """
        from aspire.utils import RelionStarFile

        indices = np.asarray(indices)
        if (indices.ndim != 1 or len(indices) == 0
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.n)):
            raise ValueError("indices must be a nonempty vector of valid STAR rows")
        if (isinstance(ds_res, (bool, np.bool_))
                or not isinstance(ds_res, (int, np.integer))
                or not 1 <= ds_res <= self.res):
            raise ValueError("ds_res must be an integer between 1 and native image size")
        ds_res = int(ds_res)
        path = Path(path).resolve()
        stack_path = Path(stack_path).resolve(strict=True)
        if path == self.star_path or path in [p.resolve() for p in self.stack_paths]:
            raise ValueError("the exported STAR must not overwrite a source file")
        if path == stack_path:
            raise ValueError("the STAR path must differ from the output stack path")
        with mrcfile.mmap(stack_path, mode="r") as stack:
            shape = stack.data.shape
            shape = (1, *shape) if len(shape) == 2 else shape
            if shape != (len(indices), ds_res, ds_res):
                raise ValueError("output stack shape does not match selected rows and ds_res")

        # Read a separate object: neither the source file nor self.metadata changes.
        exported = RelionStarFile(str(self.star_path))
        particle_block = (exported.data_block_name if exported.relion_version == "3.1"
                          else next(iter(exported.blocks)))
        selected = {key: np.asarray(values)[indices].copy()
                    for key, values in exported.blocks[particle_block].items()}
        exported.blocks[particle_block] = selected
        scale = self.res / ds_res
        pixel_size = self.pixel_size * scale
        if exported.relion_version == "3.1":
            optics = exported.blocks["optics"]
            used = np.isin(optics["_rlnOpticsGroup"], selected["_rlnOpticsGroup"])
            optics = {key: np.asarray(values)[used].copy()
                      for key, values in optics.items()}
            exported.blocks["optics"] = optics
            geometry = optics
        else:
            geometry = selected
        count = len(next(iter(geometry.values())))
        geometry["_rlnImagePixelSize"] = np.full(count, pixel_size)
        geometry["_rlnImageSize"] = np.full(count, ds_res, dtype=int)
        geometry["_rlnImageDimensionality"] = np.full(count, 2, dtype=int)
        for block in exported.blocks.values():
            for key in ("_rlnImagePixelSize", "_rlnImageSize"):
                if key in block:
                    block[key] = np.full(len(block[key]),
                                         pixel_size if key.endswith("PixelSize") else ds_res)
            if "_rlnDetectorPixelSize" in block:
                if "_rlnMagnification" in block:
                    block["_rlnDetectorPixelSize"] = (
                        pixel_size * np.asarray(block["_rlnMagnification"], dtype=float)
                        / 1e4)
                else:
                    block["_rlnDetectorPixelSize"] = (
                        np.asarray(block["_rlnDetectorPixelSize"], dtype=float) * scale)
            for key in ("_rlnOriginX", "_rlnOriginY", "_rlnOriginZ"):
                if key in block:
                    block[key] = np.asarray(block[key], dtype=float) / scale
        stack_name = os.path.relpath(stack_path, path.parent)
        if any(char.isspace() for char in stack_name):
            raise ValueError("the exported stack path must not contain whitespace")
        selected["_rlnImageName"] = np.array([
            f"{row + 1:06d}@{stack_name}" for row in range(len(indices))])
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            exported.write(str(temporary_path))
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path


def downsample_fourier_images(images, ds_res):
    """Fourier crop raw particles, preserving their field of view and amplitude.

    The output contains centered, unnormalized 2-D Fourier coefficients.
    Multiplication by ``(ds_res/native_res)**2`` matches the simulator.
    For even sizes, zero the unpaired Nyquist row/column; the reconstruction
    disk excludes these edges. No mean subtraction or normalization is applied.
    """
    images = np.asarray(images)
    if (images.ndim != 3 or images.shape[-1] != images.shape[-2]
            or np.iscomplexobj(images)):
        raise ValueError("images must be a stack of real square images")
    native_res = images.shape[-1]
    if int(ds_res) != ds_res or not 1 <= ds_res <= native_res:
        raise ValueError("ds_res must be an integer between 1 and native image size")
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
# Contrast transfer functions (was ctf.py)
# ===========================================================================

def radial_ctf_values(ds_res, vox_ds, defocus, support=None,
                      voltage=300.0, Cs=2.0, alpha=0.1, B=0.0):
    """Evaluate a radial CTF on the ``ds_res`` Fourier grid.

    ``defocus`` and ``vox_ds`` are in Angstrom, ``voltage`` in kV, ``Cs`` in
    mm, ``alpha`` is the amplitude contrast, and ``B`` is the envelope factor.
    The result is Fortran-flattened and optionally restricted to ``support``.
    """
    c = centered_coords_1d(ds_res) / float(ds_res)
    kx, ky = np.meshgrid(c, c, indexing="ij")
    kx, ky = kx.ravel(order="F"), ky.ravel(order="F")
    if support is not None:
        kx, ky = kx[np.asarray(support)], ky[np.asarray(support)]
    # Convert cycles per pixel to cycles per nm.
    s = np.sqrt(kx * kx + ky * ky) * (10.0 / float(vox_ds))
    voltage = float(voltage)
    lamb = 1.22639 / np.sqrt(voltage * 1000.0 + 0.97845 * voltage ** 2)  # nm
    defocus_nm = 0.1 * float(defocus)  # angstrom -> nm
    k2 = np.pi * lamb * defocus_nm
    k4 = (np.pi / 2.0) * 1e6 * float(Cs) * lamb ** 3  # Cs: mm -> nm via 1e6
    chi = k4 * s ** 4 - k2 * s ** 2
    alpha = float(alpha)
    h = np.sqrt(1.0 - alpha ** 2) * np.sin(chi) - alpha * np.cos(chi)
    B = float(B)
    if B:
        h = h * np.exp(-B * s ** 2)
    return np.asarray(h, dtype=np.float64)


def relion_ctf_values(ds_res, pixel_size, defocus_u, defocus_v,
                      defocus_angle=0.0, voltage=300.0, Cs=2.0, alpha=0.1,
                      phase_shift=0.0, support=None):
    """Evaluate RELION's astigmatic CTF for one particle or a particle batch.

    Parameters use STAR units: ``pixel_size`` is the *downsampled* pixel size
    in Angstrom, ``defocus_u`` and ``defocus_v`` are in Angstrom, ``voltage``
    is in kV, and ``Cs`` is in mm. ``alpha`` is the amplitude-contrast fraction
    in [0, 1]. ``defocus_angle`` and ``phase_shift`` are in degrees. Positive
    defocus denotes underfocus; negative (overfocus) values are also allowed.
    All physical parameters may be scalars or broadcast-compatible 1-D arrays.

    The result is float64, with shape ``(n_pixels,)`` for scalar parameters or
    ``(n_particles, n_pixels)`` when any parameter is a vector. Each image is
    centered and Fortran-flattened. ``support`` optionally selects pixels using
    a 1-D integer index array or a boolean mask of length ``ds_res**2``.

    NumPy image axis 0 is y (rows), axis 1 is x (columns); the STAR defocus
    angle is measured from x toward y. The undamped RELION convention is
    ``-sin(chi - phase_shift - arcsin(alpha))`` with
    ``chi = pi*Cs*lambda**3*s**4/2 - pi*lambda*defocus*s**2``.
    Thus the zero-phase DC value is +alpha, the opposite sign to
    ``radial_ctf_values`` and ASPIRE/recovar. The wavelength constants also
    follow RELION, rather than the rounded constants in the legacy evaluator.
    No B-factor, scale factor, or higher-order aberrations are applied.

    This samples the continuous CTF. On even grids, Nyquist-edge samples have
    aliased Friedel mates; omit those edges when exact discrete Hermitian
    symmetry is required for astigmatic filters.

    Reference: RELION ``CTF::initialise`` and ``CTF::getCTF`` in
    https://github.com/3dem/relion/blob/master/src/ctf.cpp and ``src/ctf.h``.
    """
    if (isinstance(ds_res, (bool, np.bool_))
            or not isinstance(ds_res, (int, np.integer)) or ds_res < 1):
        raise ValueError("ds_res must be a positive integer")

    parameter_values = {
        "pixel_size": pixel_size,
        "defocus_u": defocus_u,
        "defocus_v": defocus_v,
        "defocus_angle": defocus_angle,
        "voltage": voltage,
        "Cs": Cs,
        "alpha": alpha,
        "phase_shift": phase_shift,
    }
    parameters = []
    for name, value in parameter_values.items():
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
        value = np.asarray(value, dtype=np.float64)
        if value.ndim > 1 or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite scalar or 1-D array")
        parameters.append(value)
    try:
        (pixel_size, defocus_u, defocus_v, defocus_angle, voltage,
         Cs, alpha, phase_shift) = np.broadcast_arrays(*parameters)
    except ValueError as exc:
        raise ValueError("CTF parameter vectors must have compatible lengths") from exc
    if np.any(pixel_size <= 0) or np.any(voltage <= 0):
        raise ValueError("pixel_size and voltage must be positive")
    if np.any(Cs < 0):
        raise ValueError("Cs must be nonnegative")
    if np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("alpha must be between zero and one")

    coords = centered_coords_1d(ds_res) / float(ds_res)
    kx, ky = np.meshgrid(coords, coords, indexing="xy")
    kx, ky = kx.ravel(order="F"), ky.ravel(order="F")
    if support is not None:
        support = np.asarray(support)
        if support.ndim != 1:
            raise ValueError("support must be a 1-D index array or boolean mask")
        if support.dtype.kind == "b":
            if support.size != ds_res ** 2:
                raise ValueError("boolean support must have length ds_res**2")
        elif support.size == 0:
            support = support.astype(np.intp)
        elif (support.dtype.kind not in "iu" or np.any(support < 0)
              or np.any(support >= ds_res ** 2)):
            raise ValueError("support indices must be integers in [0, ds_res**2)")
        kx, ky = kx[support], ky[support]

    # Shared frequency polynomials avoid per-particle angle grids and trig.
    s2 = kx * kx + ky * ky
    defocus_mean = 0.5 * (defocus_u + defocus_v)
    defocus_diff = 0.5 * (defocus_u - defocus_v)
    twice_angle = np.deg2rad(2.0 * np.remainder(defocus_angle, 180.0))
    chi = defocus_mean[..., None] * s2
    chi += (defocus_diff * np.cos(twice_angle))[..., None] * (kx * kx - ky * ky)
    chi += (defocus_diff * np.sin(twice_angle))[..., None] * (2.0 * kx * ky)

    # RELION's relativistic wavelength in Angstrom; physical frequency is
    # cycles/pixel divided by pixel_size, not divided by the original sampling.
    voltage_v = voltage * 1000.0
    wavelength = 12.2643247 / np.sqrt(voltage_v * (1.0 + 0.978466e-6 * voltage_v))
    inv_pixel_squared = 1.0 / pixel_size ** 2
    chi *= (-np.pi * wavelength * inv_pixel_squared)[..., None]
    chi += ((np.pi / 2.0) * Cs * 1e7 * wavelength ** 3
            * inv_pixel_squared ** 2)[..., None] * s2 ** 2
    chi -= np.deg2rad(np.remainder(phase_shift, 360.0))[..., None]
    return (-np.sqrt(1.0 - alpha ** 2)[..., None] * np.sin(chi)
            + alpha[..., None] * np.cos(chi))


# ===========================================================================
# Reference-free low-crowding selection (was particle_selection.py)
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
    nearest_neighbor_distances: np.ndarray
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
    if names.ndim != 1 or len(names) == 0:
        raise ValueError("micrograph_names must be a nonempty vector")
    x = _finite_vector(coordinate_x, "coordinate_x", len(names))
    y = _finite_vector(coordinate_y, "coordinate_y", len(names))

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
        max_score_quantile=0.95, smooth_sigma=1.1, inner_radius=0.50,
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

    The selection is deterministic: ties are resolved by input position and
    returned positions are sorted.  No random sampling is performed here.
    """

    shape = getattr(images, "shape", None)
    if (shape is None or len(shape) != 3 or shape[0] < 1
            or shape[1] != shape[2] or shape[1] < 1):
        raise ValueError("images must have shape (n, side, side) with n and side positive")
    count, side, _ = map(int, shape)
    names = np.asarray(micrograph_names, dtype=str)
    if names.shape != (count,):
        raise ValueError("micrograph_names must contain one value per image")
    distances = np.asarray(nearest_neighbor_distances, dtype=np.float64)
    if (distances.shape != (count,) or np.any(np.isnan(distances))
            or np.any(distances < 0)):
        raise ValueError(
            "nearest_neighbor_distances must be a nonnegative vector without NaN")
    contrast = _finite_vector(contrast, "contrast", count)

    n_select = _positive_integer(n_select, "n_select")
    contrast_bins = _positive_integer(contrast_bins, "contrast_bins")
    batch_size = _positive_integer(batch_size, "batch_size")
    if n_select > count:
        raise ValueError("n_select cannot exceed the number of images")
    if (isinstance(minimum_neighbor_distance, (bool, np.bool_))
            or not np.isfinite(minimum_neighbor_distance)
            or minimum_neighbor_distance <= 0):
        raise ValueError("minimum_neighbor_distance must be finite and positive")
    if not np.isfinite(smooth_sigma) or smooth_sigma < 0:
        raise ValueError("smooth_sigma must be finite and nonnegative")
    if not 0 < max_score_quantile <= 1:
        raise ValueError("max_score_quantile must lie in (0, 1]")
    if not (0 < inner_radius < outer_radius_min < outer_radius_max <= 1):
        raise ValueError("radial masks must be ordered inside the unit disk")
    if not (0 < inner_top_fraction <= 1 and 0 < outer_top_fraction <= 1):
        raise ValueError("top fractions must lie in (0, 1]")
    finite_parameters = (
        max_score_quantile, inner_radius, outer_radius_min, outer_radius_max,
        inner_top_fraction, outer_top_fraction,
    )
    if not np.all(np.isfinite(finite_parameters)):
        raise ValueError("selection parameters must be finite")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")

    coordinates = np.arange(side, dtype=np.float64) - (side - 1) / 2
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt(xx ** 2 + yy ** 2) / (side / 2)
    inner_mask = radius <= inner_radius
    outer_mask = (radius >= outer_radius_min) & (radius <= outer_radius_max)
    if not np.any(inner_mask) or not np.any(outer_mask):
        raise ValueError("radial masks must each contain at least one pixel")

    scores = np.empty(count, dtype=np.float64)
    inner_strength = np.empty(count, dtype=np.float64)
    outer_strength = np.empty(count, dtype=np.float64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        batch = np.array(images[start:end], dtype=np.float32, copy=True)
        if not np.all(np.isfinite(batch)):
            raise ValueError("images contain NaN or infinity")
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

    score_ceiling = float(np.quantile(scores, max_score_quantile))
    geometry_pass = ((distances >= minimum_neighbor_distance)
                     & (scores <= score_ceiling))
    if int(geometry_pass.sum()) < n_select:
        raise RuntimeError(
            f"only {geometry_pass.sum()} particles pass the neighbor and score gates; "
            f"{n_select} requested")

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
    if len(selected_positions) != n_select:
        raise RuntimeError("stratified selector returned the wrong count")
    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[selected_positions] = True

    return LowCrowdingSelection(
        selected_positions=selected_positions,
        rejected_positions=np.flatnonzero(~selected_mask),
        geometry_pass=geometry_pass,
        crowding_scores=scores,
        inner_strength=inner_strength,
        outer_strength=outer_strength,
        nearest_neighbor_distances=distances.copy(),
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


def _finite_vector(values, name, size):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return values


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _top_rms(values, fraction):
    count = max(1, int(np.ceil(fraction * values.shape[1])))
    partitioned = np.partition(values, values.shape[1] - count, axis=1)
    return np.sqrt(np.mean(partitioned[:, -count:] ** 2, axis=1))


def _capped_proportional_quotas(population, capacity, total):
    """Allocate proportional integer quotas subject to stratum capacities."""

    population = np.asarray(population, dtype=np.int64)
    capacity = np.asarray(capacity, dtype=np.int64)
    if (population.shape != capacity.shape or np.any(population < 0)
            or np.any(capacity < 0) or np.any(capacity > population)
            or total < 0 or total > int(capacity.sum())):
        raise ValueError("invalid population, capacity, or requested total")
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
    if int(quotas.sum()) != total or np.any(quotas > capacity):
        raise RuntimeError("failed to allocate exact capped quotas")
    return quotas
