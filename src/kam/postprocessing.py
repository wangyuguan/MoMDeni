"""Volume alignment and FSC, and moment preparation and reconstruction (independent SphFB M1/M2/M3 matching)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import finufft
import mrcfile
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation
from skimage.registration import phase_cross_correlation

from .moments import (
    FBMomentBasis,
    fb_objective_and_grad,
    UPSAMPFAC,
    centered_coords_1d,
    centered_fftn,
    centered_ifftn,
    fourier_crop,
    viewing_grid,
    use_device,
    empirical_moments,
)
from .optimization import SphFBParam
from .utils import ROOT, fourier_resample_real, git_commit, load_volume, save_mrc, sha256_file
from .viewing import LowDegreeViewingDensity, fit_single_moment_low_degree


# ===========================================================================
# Volume alignment and FSC
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
    magnification = float(magnification)
    shift = (np.zeros(3, dtype=float) if translation is None
             else np.asarray(translation, dtype=float))
    rotation = np.asarray(R, dtype=float)
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
# Moment preparation and reconstruction
# ===========================================================================

DEFAULT_DENOISE_RES = 64
DEFAULT_MOMENT_RES = 64
DEFAULT_N_SELECT = 10_000
DEFAULT_ELL_MAX = 7
DEFAULT_VIEWING_DEGREE = 2
# Fixed settings of the M1 -> M2 -> M3 fits: N_INIT random starts of M1 and M2 (M3 runs once, from the M2 solution with the smallest
# M3 residual), and the L-BFGS iteration limits of M1/M2 and of M3.
N_INIT = 10
MAXITER = 10_000
M3_MAXITER = 500
DEFAULT_SHIFT_STEP_PX = 0.5
DEFAULT_ALIGNMENT_LOWPASS = 80.0


@dataclass(frozen=True)
class MomentPreparationConfig:
    """Configuration for :func:`prepare_moments`."""

    denoising_dir: Path
    selection_file: Path
    output_dir: Path
    n_images: int = DEFAULT_N_SELECT
    denoise_res: int = DEFAULT_DENOISE_RES
    moment_res: int = DEFAULT_MOMENT_RES
    batch_size: int = 256
    device: str = "gpu"
    fb_ell_max: int = DEFAULT_ELL_MAX


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration for one independent M2-only or M3-only fit."""

    prepared_dir: Path
    output_dir: Path
    order: int
    ell_max: int = DEFAULT_ELL_MAX
    viewing_degree: int = DEFAULT_VIEWING_DEGREE
    symmetry: str = "C1"
    n_init: int = N_INIT
    maxiter: int = MAXITER
    device: str = "gpu"
    print_every: int = 100
    seed: int = 0
    initial_fit: Path | None = None
    resume: bool = False


@dataclass(frozen=True)
class ReconstructionResult:
    """Summary of a completed independent moment fit."""

    fit_path: Path
    reconstruction_path: Path
    summary_path: Path
    order: int
    best_init: int
    objective: float
    relative_moment_error: float


@dataclass(frozen=True)
class PreparedMomentData:
    """Arrays and geometry loaded from ``moments.npz``."""

    directory: Path
    report: dict
    values: dict
    target: np.ndarray
    order: int
    side: int
    pixel_size: float
    n_images: int
    fb_basis: FBMomentBasis
    shapes: tuple


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _elapsed_seconds(start):
    return time.monotonic() - start


def _scalar(saved, key):
    value = np.asarray(saved[key])
    return value.reshape(()).item()


def _sha256_array(value):
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes()).hexdigest()


def _manifest_path(directory, value):
    path = Path(str(value))
    if not path.is_absolute():
        path = directory / path
    return path.resolve(strict=True)


def read_denoising_output(directory):
    """The files of a denoising output and the content of its manifest."""

    directory = Path(directory).resolve(strict=True)
    manifest_path = directory / "manifest.npz"
    with np.load(manifest_path, allow_pickle=False) as saved:
        provenance = {
            key: _scalar(saved, key)
            for key in ("star_path", "data_folder",
                        "stack_prefix", "source_fingerprint", "seed")
        }
        provenance.update(
            dataset=str(_scalar(saved, "dataset")),
            n_total=int(_scalar(saved, "n_total")),
            n_images=int(_scalar(saved, "n_images")),
            native_res=int(_scalar(saved, "native_res")),
            native_pixel=float(_scalar(saved, "vox_native")),
            denoise_res=int(_scalar(saved, "denoise_res")),
            denoise_pixel=float(_scalar(saved, "vox_denoise")),
            source_rows=np.asarray(saved["source_rows"], dtype=np.int64),
            image_names=np.asarray(saved["image_names"]),
            contrast_est=np.asarray(saved["contrast_est"], dtype=np.float64),
        )
        stack_path = _manifest_path(directory, _scalar(saved, "denoised_stack"))
    return directory, stack_path, manifest_path, directory / "report.json", provenance


def load_selection(selection_file):
    """The selected positions, source rows and image names of a selection file."""

    selection_file = Path(selection_file).resolve(strict=True)
    with np.load(selection_file, allow_pickle=False) as saved:
        positions = np.asarray(saved["selected_positions"], dtype=np.int64)
        rows = np.asarray(saved["selected_source_rows"], dtype=np.int64)
        names = np.asarray(saved["selected_image_names"])
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

    side = moment_res
    basis, (m1, m2, m3) = empirical_moments(images, side, ell_max, batch_size=batch_size)
    metadata = dict(
        n_images=len(images), denoise_res=int(np.shape(images)[1]),
        moment_res=side, fb=basis.summary(), batch_size=batch_size,
    )
    return dict(
        mbar1=m1, mbar2=m2, mbar3=m3, metadata=metadata, basis=basis,
    )


def _fb_archive_fields(basis):
    nodes, weights, radial, counts = basis.radial_arrays()
    return dict(
        fb_ell_max=basis.ell_max,
        fb_radial_nodes=nodes, fb_radial_weights=weights,
        fb_radial_basis=radial, fb_radial_counts=counts,
    )


def prepare_moments_from_images(images, pixel_size, output, *, moment_res, ell_max=DEFAULT_ELL_MAX, batch_size=256, dataset="",
                                source=""):
    """Write ``moments.npz`` and ``report.json`` (the files of ``prepare_moments``) for a stack of denoised images of pixel ``pixel_size``."""

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    empirical = empirical_moments_from_denoised_images(images, moment_res=moment_res, ell_max=ell_max, batch_size=batch_size)
    # Mirroring an image conjugates its stored q>=0 FB coefficients, so averaging every image with its mirror leaves
    # exactly the real parts of the invariant moments.
    m1, m2, m3 = (np.ascontiguousarray(empirical[f"mbar{order}"].real).astype(np.complex128) for order in (1, 2, 3))
    side = int(np.shape(images)[-1])
    pixel = pixel_size * side / moment_res
    moments_path = output / "moments.npz"
    np.savez(moments_path, mbar1=m1, mbar2=m2, mbar3=m3, n_images=len(images), ds_res=moment_res, dataset=dataset,
             pixel_size=pixel, native_side=side, native_pixel_size=pixel_size, **_fb_archive_fields(empirical["basis"]))
    _write_json(output / "report.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), dataset=dataset, denoised_stack=str(source),
        n_images=len(images), image_side=side, image_pixel_size_angstrom=pixel_size, moment_res=moment_res,
        pixel_size_angstrom=pixel, fb_ell_max=ell_max, n_frequencies=empirical["basis"].n_coefficients,
        n_m2_entries=int(m2.size), n_canonical_m3_entries=len(m3), metadata=empirical["metadata"],
        prepared_file=str(moments_path), prepared_sha256=sha256_file(moments_path), kam_cryoem_commit=git_commit(ROOT),
        elapsed_seconds=_elapsed_seconds(started)))
    return output

def prepare_moments(config: MomentPreparationConfig):
    """Prepare complete M1--M3 from a selected denoised image subset."""

    output = config.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new or empty moment preparation directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timings = {}

    phase_started = time.monotonic()
    use_device(config.device)
    timings["setup"] = _elapsed_seconds(phase_started)

    phase_started = time.monotonic()
    denoising_dir, stack_path, manifest_path, denoising_report, manifest = read_denoising_output(config.denoising_dir)
    selection_path, positions, rows, names = load_selection(config.selection_file)
    timings["input_loading"] = _elapsed_seconds(phase_started)

    phase_started = time.monotonic()
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
    m1, m2, m3 = (
        empirical[f"mbar{order}"] for order in (1, 2, 3))
    # Mirroring an image conjugates its stored q>=0 FB coefficients, so averaging every image with its mirror leaves
    # exactly the real parts of the invariant moments; no images are duplicated.
    m1, m2, m3 = (np.ascontiguousarray(m.real).astype(np.complex128) for m in (m1, m2, m3))
    basis = empirical["basis"]
    n_frequencies = basis.n_coefficients
    n_m3 = len(m3)
    basis_fields = _fb_archive_fields(basis)
    norms = [float(np.linalg.norm(moment)) for moment in (m1, m2, m3)]
    pixel = manifest["native_res"] * manifest["native_pixel"] \
        / config.moment_res
    timings["moment_metadata"] = _elapsed_seconds(
        phase_started)

    output_started = time.monotonic()
    moments_path = output / "moments.npz"
    np.savez(
        moments_path,
        mbar1=m1, mbar2=m2, mbar3=m3,
        selected_positions=positions, selected_source_rows=rows,
        selected_image_names=names,
        n_images=len(positions), ds_res=config.moment_res,
        dataset=manifest["dataset"], pixel_size=pixel,
        native_side=manifest["native_res"],
        native_pixel_size=manifest["native_pixel"],
        denoise_side=manifest["denoise_res"],
        denoised_pixel_size=manifest["denoise_pixel"],
        **basis_fields,
    )
    metadata = empirical["metadata"]
    metadata.update(moment_norms=norms)
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        fb_ell_max=config.fb_ell_max,
        dataset=manifest["dataset"],
        denoising_directory=str(denoising_dir),
        denoised_stack=str(stack_path),
        denoising_manifest=str(manifest_path),
        denoising_manifest_sha256=sha256_file(manifest_path),
        denoising_report=str(denoising_report),
        selection_file=str(selection_path),
        selection_sha256=sha256_file(selection_path),
        n_images=len(positions),
        n_requested=config.n_images,
        denoise_res=config.denoise_res,
        moment_res=config.moment_res,
        pixel_size_angstrom=pixel,
        native_side=manifest["native_res"],
        native_pixel_size_angstrom=manifest["native_pixel"],
        n_frequencies=n_frequencies,
        n_m2_entries=int(m2.size),
        n_canonical_m3_entries=n_m3,
        source=dict(
            star_path=str(manifest["star_path"]),
            data_folder=str(manifest["data_folder"]),
            stack_prefix=str(manifest["stack_prefix"]),
            source_fingerprint=str(manifest["source_fingerprint"]),
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
        prepared_sha256=sha256_file(moments_path),
        kam_cryoem_commit=git_commit(ROOT),
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

def load_prepared_moments(path, order):
    """Load one active target from a prepared moment archive."""

    path = Path(path).resolve(strict=True)
    report_path = path / "report.json"
    archive = path / "moments.npz"
    report = json.loads(report_path.read_text())
    with np.load(archive, allow_pickle=False) as saved:
        values = {key: np.asarray(saved[key]).copy() for key in saved.files}
    n_images = int(values["n_images"].reshape(()))
    side = int(values["ds_res"].reshape(()))
    fb_basis = _load_fb_basis(values, side)
    shapes = fb_basis.shapes
    pixel = float(values["pixel_size"].reshape(()))
    target = np.asarray(values[f"mbar{order}"], dtype=np.complex128)
    return PreparedMomentData(
        directory=path, report=report, values=values,
        target=target, order=order,
        side=side, pixel_size=pixel, n_images=n_images,
        fb_basis=fb_basis, shapes=shapes,
    )


def _load_fb_basis(values, side):
    """Rebuild the stored FB radial basis."""

    return FBMomentBasis(
        side, int(values["fb_ell_max"].reshape(())),
        radial=tuple(values[key] for key in (
            "fb_radial_nodes", "fb_radial_weights", "fb_radial_basis",
            "fb_radial_counts")))


def deterministic_random_starts(n_parameters, n_init, seed=0):
    """Return reproducible Gaussian coefficient directions and provenance."""

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
        fb_basis=prepared.fb_basis.summary(),
        objective=f"in-plane-invariant steerable FB M{config.order} only",
        prepared_directory=str(prepared.directory),
        prepared_sha256=prepared.report["prepared_sha256"],
        selected_images=prepared.n_images,
        moment_res=prepared.side,
        pixel_size_angstrom=prepared.pixel_size,
        native_side=int(prepared.values["native_side"].reshape(())),
        native_pixel_size_angstrom=float(
            prepared.values["native_pixel_size"].reshape(())),
        fb_coefficients=prepared.fb_basis.n_coefficients,
        n_m2_entries=int(np.prod(prepared.shapes[1])),
        canonical_m3_entries=int(np.prod(prepared.shapes[2])),
        ell_max=config.ell_max,
        n_volume_parameters=param.n_params,
        symmetry=config.symmetry,
        n_free_volume_parameters=param.n_params if param.basis is None else param.basis.shape[1],
        viewing_degree=config.viewing_degree,
        n_view_parameters=density.n_parameters,
        sphere_nodes=grid.Ns,
        target_norm=target_norm,
        target_sha256=_sha256_array(prepared.target),
        n_initializations=config.n_init,
        maxiter=config.maxiter,
        initialization=(
            "deterministic Gaussian coefficient directions from "
            "SeedSequence([seed, 2, index]), rescaled to the "
            f"M{config.order} target norm"
            if not warm_fit_paths else
            f"M{config.order - 1} init_i volume and viewing density, with the volume analytically rescaled by least squares of order "
            f"{config.order} against M{config.order}"),
        initialization_seed=config.seed,
        initial_fit=initial_fit,
        initial_fit_files=[str(path) for path in warm_fit_paths],
        initial_fit_sha256=[sha256_file(path) for path in warm_fit_paths],
        kam_cryoem_commit=git_commit(ROOT),
    )


# Keys of config.json that must agree for a partially finished fit to continue (the others follow from them).
_RESUME_KEYS = (
    "order", "prepared_sha256", "ell_max", "viewing_degree", "symmetry",
    "n_initializations", "maxiter", "initialization_seed", "initial_fit_sha256",
)


def _read_completed_fit(output, index):
    """The record of initialization ``index`` if it finished before an interruption (its summary is written last)."""
    summary = output / f"init_{index}" / "summary.json"
    return json.loads(summary.read_text()) if summary.is_file() else None


def _copy_best_fit(output, record):
    source = output / f"init_{int(record['init_index'])}"
    shutil.copy2(source / "fit.npz", output / "fit.npz")
    shutil.copy2(source / "recon.mrc", output / "recon.mrc")


def _load_m2_warm_start(path):
    """The path, volume coefficients and viewing parameters of one previous-order fit."""

    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as saved:
        return path, np.asarray(saved["theta"], dtype=np.float64), np.asarray(saved["beta"], dtype=np.float64)


def _resolve_m2_fit_paths(initial_fit, n_init):
    """Resolve one previous-order fit per requested initialization."""

    directory = Path(initial_fit).resolve(strict=True)
    return [(directory / f"init_{index}" / "fit.npz").resolve(strict=True) for index in range(n_init)]


def _m3_warm_starts(
        warm_fits, config, prepared, param, density, targets, lambdas):
    """Rescale each previous-order initialization for the next objective."""

    starts = []
    for index, (path, candidate, warm_beta) in enumerate(warm_fits):
        p = density.pmf(warm_beta)
        _, _, _, info = fb_objective_and_grad(
            candidate, p, param, *targets,
            lambdas=lambdas, return_info=True,
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
            f"analytic M{config.order} rescaling of {path}",
        ))
    return starts


def reconstruct_single_moment(config: ReconstructionConfig):
    """Fit one moment order and select the start with minimum moment loss."""

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

    use_device(config.device)
    # FB moments are already in-plane averaged: viewing directions only; the rule integrates the M3 model exactly.
    grid = viewing_grid(3 * config.ell_max + config.viewing_degree)
    param = SphFBParam(prepared.side, config.ell_max, grid.rots, fb_basis=prepared.fb_basis, symmetry=config.symmetry)
    density = LowDegreeViewingDensity(grid, degree=config.viewing_degree)
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
            direction = param.project(direction)
            _, _, _, info = fb_objective_and_grad(
                direction, p_uniform, param, *targets,
                lambdas=lambdas, return_info=True,
            )
            prediction = info[f"m{config.order}"]
            prediction_norm = float(np.linalg.norm(prediction))
            scale = (target_norm / prediction_norm) ** (1.0 / config.order)
            starts.append((
                index, child_seed, scale * direction, beta_uniform.copy(),
                scale, prediction_norm, direction_hash,
                f"random Gaussian direction scaled to M{config.order} "
                "target norm",
            ))
        warm_fit_paths = []
    else:
        # M{order} start i is M{order-1} init_i.
        warm_fit_paths = _resolve_m2_fit_paths(config.initial_fit, config.n_init)
        warm_fits = [_load_m2_warm_start(path) for path in warm_fit_paths]
        starts = _m3_warm_starts(
            warm_fits, config, prepared, param, density, targets, lambdas)

    payload = _config_payload(
        config, prepared, param, density, grid, target_norm,
        warm_fit_paths=warm_fit_paths)
    payload["init_seeds"] = [start[1] for start in starts]
    payload["initialization_sources"] = [start[7] for start in starts]
    payload["m3_warm_start_semantics"] = (
        None if config.initial_fit is None else
        f"M{config.order} start i is M{config.order - 1} init_i with its fitted viewing density; each volume receives the analytic "
        f"order-{config.order} least-squares amplitude against "
        f"M{config.order}")
    config_path = output / "config.json"
    if nonempty:
        previous = json.loads(config_path.read_text())
        mismatched = {
            key: (previous.get(key), payload.get(key))
            for key in _RESUME_KEYS
            if previous.get(key) != payload.get(key)
        }
        if mismatched:
            raise ValueError(
                f"resume configuration differs from existing fit: {mismatched}")
        payload["created_utc"] = previous.get(
            "created_utc", payload["created_utc"])
    else:
        _write_json(config_path, payload)

    completed = {}
    if config.resume:
        for index in range(config.n_init):
            record = _read_completed_fit(output, index)
            if record is not None:
                completed[index] = record
    setup_elapsed = _elapsed_seconds(setup_started)
    fit_loop_started = time.monotonic()
    optimization_elapsed_this_run = 0.0
    completed_this_run = 0
    records = []
    for (index, child_seed, theta0, beta0, scale, prediction_norm,
         direction_hash, start_source) in starts:
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
        initial_loss = float(fb_objective_and_grad(
            theta0, density.pmf(beta0), param, *targets,
            lambdas=lambdas,
        )[0])
        np.savez(
            init / "start.npz",
            theta=theta0, beta=beta0, p=density.pmf(beta0),
            seed=child_seed, moment_order=config.order,
            scale=scale,
            prediction_norm_before_scaling=prediction_norm,
            target_norm=target_norm,
            direction_sha256=direction_hash,
            objective=initial_loss,
            source=start_source,
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
        )
        optimization_elapsed = _elapsed_seconds(optimization_started)
        optimization_elapsed_this_run += optimization_elapsed
        optimizer = result["res"]
        loss = float(optimizer.fun)
        relative_error = result["relative_error"]
        np.savez(
            init / "fit.npz",
            theta=result["theta"], beta=result["beta"], p=result["p"],
            init_index=index, initialization_source=start_source,
            objective=loss, normalized_active_moment_loss=loss,
            relative_moment_error=relative_error,
            moment_order=config.order,
            viewing_degree=config.viewing_degree,
            ell_max=config.ell_max,
            grid_rots=grid.rots, grid_xyz=grid.xyz, grid_q=grid.q,
            prepared_sha256=prepared.report["prepared_sha256"],
        )
        save_mrc(
            param.volume(result["theta"]).transpose(2, 1, 0),
            init / "recon.mrc", prepared.pixel_size,
        )
        record = dict(
            init_index=index,
            seed=child_seed,
            moment_order=config.order,
            initial_objective=initial_loss,
            initialization_source=start_source,
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
    best = select_best_fit(records)
    _copy_best_fit(output, best)
    with np.load(output / "fit.npz", allow_pickle=False) as saved:
        beta = np.asarray(saved["beta"], dtype=np.float64)
    _save_viewing_plot(output, density, beta)

    optimization_total = float(sum(record["optimization_elapsed_seconds"] for record in records))
    initializations_total = float(sum(record["elapsed_seconds"] for record in records))

    summary = dict(payload)
    summary.update(
        fits=records,
        best_init_by_loss=int(best["init_index"]),
        best_normalized_active_moment_loss=float(best["objective"]),
        fit_complete=True,
        completed_in_this_invocation=completed_this_run,
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
        fit_path=output / "fit.npz",
        reconstruction_path=output / "recon.mrc",
        summary_path=output / "summary.json",
        order=config.order,
        best_init=int(best["init_index"]),
        objective=float(best["objective"]),
        relative_moment_error=float(best["relative_moment_error"]),
    )


def _load_alignment_volume(path, pixel_size=None):
    """Read a cubic map, retaining header sampling beside an optional override."""
    volume, header_pixel = load_volume(path)
    effective_pixel = float(header_pixel if pixel_size is None else pixel_size)
    return volume, effective_pixel, {
        "header_pixel_size_angstrom": float(header_pixel),
        "effective_pixel_size_angstrom": effective_pixel,
        "pixel_size_overridden": pixel_size is not None,
        "pixel_size_source": "MRC header" if pixel_size is None else "explicit override",
        "side": int(volume.shape[0]),
        "effective_box_size_angstrom": float(volume.shape[0] * effective_pixel),
    }


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
        fourier_resample_real(reconstruction, int(target_side))
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
        "physical_grid_scale_reconciled_by_alignment_magnification": bool(
            relative_box_difference > 0),
        "relative_imaginary_norm": float(reconstruction_imaginary),
        "reference_relative_imaginary_norm": float(reference_imaginary),
        "source_bandlimit": {
            "maximum_supported_fsc_shell": int(maximum_supported_shell),
        },
    }
    reference_provenance = {
        "source_shape": list(reference.shape),
        "source_pixel_size": float(reference_pixel_size),
        "source_box_size_angstrom": float(reference_box),
        "target_shape": [target_side] * 3,
        "target_pixel_size": float(reference_pixel_size),
        "target_box_size_angstrom": float(reference_box),
        "relative_imaginary_norm": float(reference_imaginary),
    }
    return (reconstruction_evaluation, reference_evaluation, target_pixel,
            provenance, reference_provenance)


def _hard_lowpass(volume, pixel_size, resolution):
    side = volume.shape[0]
    coordinates = np.arange(side, dtype=float) - side // 2
    kx, ky, kz = np.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij")
    cutoff = side * pixel_size / resolution
    mask = kx * kx + ky * ky + kz * kz <= cutoff * cutoff
    filtered = np.real(centered_ifftn(centered_fftn(volume) * mask))
    return filtered, dict(
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
    lower, upper = map(float, scale_bounds)

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
        lowpass_resolution, allow_sign, label, reference_pixel_size=None):
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
    reconstruction_native = np.array(
        reconstruction, dtype=np.float64, copy=True)
    native, reference_pixel, reference_sampling = _load_alignment_volume(
        reference_map, reference_pixel_size)
    (reconstruction_native, reference_native, pixel_size,
     reconstruction_resampling, resampling) = _alignment_evaluation_grid(
         reconstruction_native, pixel_size,
         native.transpose(2, 1, 0), reference_pixel)
    side = reconstruction_native.shape[0]
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
        source_sha256=sha256_file(reference_map),
        working_side=side,
        working_pixel_size=pixel_size,
        source_sampling=reference_sampling,
    )
    similarity_transform = dict(
        reflection_applied_before_interpolation=bool(reflected),
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
        original_real_space_means=means,
        reference_resampling=resampling,
        reconstruction_fourier_upsampling=reconstruction_resampling,
        reference_sampling=reference_sampling,
        alignment_search=alignment_search,
        similarity_transform=similarity_transform,
        rigid_refinement=rigid_refinement,
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
    )
    report = dict(
        complete=True,
        map=str(map_path),
        map_sha256=sha256_file(map_path),
        reference_map=str(reference_map),
        reference_map_sha256=sha256_file(reference_map),
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
    without smoothing or symmetry expansion. The MoM fit is formed with
    reflection augmentation (an antipodal density), so each particle's
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

    with np.load(fit_path, allow_pickle=False) as saved:
        xyz = np.asarray(saved["grid_xyz"], dtype=np.float64)
        q = np.asarray(saved["grid_q"], dtype=np.float64)
        degree = np.asarray(saved["viewing_degree"]).item()
        beta = np.asarray(saved["beta"], dtype=np.float64)
    grid = SimpleNamespace(xyz=xyz, q=q, Ns=len(xyz))
    density = LowDegreeViewingDensity(grid, degree=degree)

    metadata = RelionStarFile(str(star_path)).get_merged_data_block()
    rot = np.asarray(metadata["_rlnAngleRot"], dtype=np.float64)
    tilt = np.asarray(metadata["_rlnAngleTilt"], dtype=np.float64)
    rot = np.deg2rad(np.mod(rot, 360.0))
    tilt = np.deg2rad(np.clip(tilt, 0, 180))
    original_directions = np.column_stack((
        np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt),
    ))
    relion_directions = original_directions @ transforms[1].T
    relion_directions /= np.linalg.norm(relion_directions, axis=1)[:, None]
    # The mirror image of a projection along n is the projection along -n
    # (up to an in-plane rotation), so reflection augmentation adds -n.
    mirrored_directions = -relion_directions
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

    output.mkdir(parents=True, exist_ok=True)
    mom_pdf = output / "viewing_density_mom_aligned.pdf"
    relion_pdf = output / "viewing_density_relion_aligned.pdf"
    data_path = output / "viewing_density_aligned.npz"
    maximum = float(mom_density.max())
    for pdf, title in ((mom_pdf, "MoM"), (relion_pdf, "RELION, images and mirrors")):
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
        mom_fit=str(fit_path), mom_fit_sha256=sha256_file(fit_path),
        relion_star=str(star_path), relion_star_sha256=sha256_file(star_path),
        reference_map=initial_alignment.get("reference_map"),
        reference_map_sha256=reference_hashes[0],
        mom_direction_transform=transforms[0].tolist(),
        relion_direction_transform=transforms[1].tolist(),
        mom_viewing_degree=int(degree), relion_particle_count=int(rot.size),
        relion_scatter=dict(count=int(rot.size), color="black", marker=".",
                            mirrored_count=int(len(mirrored_directions)),
                            mirrored_color="tab:red",
                            size_points_squared=1.5, alpha=1.0,
                            rasterized_in_pdf=True, dpi=300, colorbar=False),
        relion_histogram_shape=list(counts.shape),
        mom_color_limits=[0.0, maximum],
        normalization=dict(mom=mom_integral, relion=relion_integral),
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json(output / "viewing_density_aligned.json", report)
    return report
