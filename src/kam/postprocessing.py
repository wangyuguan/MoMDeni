"""Moment preparation and reconstruction (SphFB M1/M2/M3 matching)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mrcfile
import numpy as np

from .moments import (
    FBMomentBasis,
    fb_objective_and_grad,
    viewing_grid,
    use_device,
    empirical_moments,
)
from .optimization import SphFBParam
from .utils import ROOT, git_commit, save_mrc, sha256_file
from .viewing import LowDegreeViewingDensity, fit_single_moment_low_degree


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
