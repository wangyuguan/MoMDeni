#!/usr/bin/env python3
"""Denoised-particle Fourier--Bessel moment reconstruction.

    kam pipeline particles.star --outdir results ...
                              denoise, select, reconstruct M2 then M3
    kam align_reference ...   align initial/refined maps and plot FSC

Each pipeline run writes to
``<results>/<EMPIAR-number>_<SLURM-job-id>/kam_pipeline``. Local runs use a
collision-resistant ``local`` identifier. Optional RELION preparation is an
experiment-specific workflow outside ``src``; reference comparison remains a
separate downstream command.
The pipeline saves resolved settings to hyperparameters.json and appends live
progress, subprocess output, and errors to run.log.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle
import mrcfile
import numpy as np

from .centering import (METHOD as CENTERING_METHOD,
                        METHOD_VERSION as CENTERING_METHOD_VERSION,
                        CenteringConfig)
from .denoise import (CENTERING_PREVIEW_DESCRIPTION,
                     CENTERING_PREVIEW_MARKER_SEMANTICS,
                     DEFAULT_FLE_ROOT,
                     FastCWFDenoiseConfig, denoise_relion_particles,
                     save_centering_preview)
from . import postprocessing
from .postprocessing import ReconstructionConfig, reconstruct_single_moment
from .preprocessing import (RelionParticles, nearest_particle_distances,
                           select_low_crowding_particles)
from .utils import capture_run_log, show_particle_image

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
_THIS_FILE = Path(__file__).resolve()
PARTICLE_SELECTION_SOURCE = PACKAGE_DIR / "preprocessing.py"
# ===========================================================================
# MoM pipeline orchestration
# ===========================================================================

PIPELINE_VERSION = "fourier_bessel_mom_v11"

RECONSTRUCTION_IMPLEMENTATION_FILES = {
    "moment_reconstruction": PACKAGE_DIR / "postprocessing.py",
    "viewing": PACKAGE_DIR / "viewing.py",
    "moments": PACKAGE_DIR / "moments.py",
    "optimization": PACKAGE_DIR / "optimization.py",
    "simulation": PACKAGE_DIR / "simulation.py",
    "utils": PACKAGE_DIR / "utils.py",
}
PIPELINE_IMPLEMENTATION_FILES = {
    "pipeline": _THIS_FILE,
    "denoising": PACKAGE_DIR / "denoise.py",
    "centering": PACKAGE_DIR / "centering.py",
    "preprocessing": PACKAGE_DIR / "preprocessing.py",
    **RECONSTRUCTION_IMPLEMENTATION_FILES,
}

DEFAULT_DENOISE_RES = 64
DEFAULT_MOMENT_RES = 16
DEFAULT_N_SELECT = 10_000
DEFAULT_ELL_MAX = 5
DEFAULT_VIEWING_DEGREE = 2
DEFAULT_SPHERE_NODES = 200
DEFAULT_N_INIT = 1
DEFAULT_MAXITER = 5_000
DEFAULT_MAX_M3_ENTRIES = 10_000_000
DEFAULT_NEIGHBOR_FRACTION = 280.0 / 380.0
PIPELINE_HEARTBEAT_SECONDS = 60.0
LOCAL_RUN_ID = (
    "local_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    + f"_{os.getpid()}"
)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json_exclusive(path: Path, value: dict) -> None:
    """Create a JSON file without replacing a concurrent pipeline manifest."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _write_json_atomic(path: Path, value: dict) -> None:
    """Atomically create or replace a JSON report in ``path.parent``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate_optional_timing(record: dict, context: str) -> None:
    """Validate timing metadata when present, while accepting older outputs."""
    if "elapsed_seconds" in record:
        elapsed = record["elapsed_seconds"]
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed)) or float(elapsed) < 0):
            raise ValueError(f"{context} has invalid elapsed_seconds")
    if "timings_seconds" in record:
        timings = record["timings_seconds"]
        if not isinstance(timings, dict):
            raise ValueError(f"{context} has invalid timings_seconds")
        for name, value in timings.items():
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(
                    f"{context} has invalid timing {name!r}: {value!r}")


def _code_identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _fingerprint(files: dict[str, Path]) -> dict:
    """Hash a named set of source files in stable name order."""
    return {
        name: _code_identity(files[name])
        for name in sorted(files)
    }


def _implementation_fingerprint() -> dict:
    return _fingerprint(PIPELINE_IMPLEMENTATION_FILES)


def _reconstruction_implementation_fingerprint() -> dict:
    return _fingerprint(RECONSTRUCTION_IMPLEMENTATION_FILES)


def _validate_recorded_implementation(recorded: object, context: str) -> None:
    """Verify every recorded implementation file against its current bytes."""
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError(f"{context} lacks an implementation fingerprint")
    for name, identity in recorded.items():
        if (not isinstance(identity, dict)
                or not isinstance(identity.get("path"), str)
                or not isinstance(identity.get("sha256"), str)):
            raise ValueError(
                f"{context} has an invalid implementation identity for {name}")
        path = Path(identity["path"])
        if not path.is_file():
            raise ValueError(
                f"{context} implementation file is missing for {name}: {path}")
        actual = _sha256(path)
        if actual != identity["sha256"]:
            raise ValueError(
                f"{context} implementation changed for {name}: {path}")


def _validate_pipeline_manifest(previous: dict, configuration: dict) -> None:
    if previous.get("configuration") != configuration:
        raise ValueError(
            "pipeline arguments differ from the existing pipeline_config.json; "
            "use the original arguments or a new output directory")
    current = _implementation_fingerprint()
    if previous.get("implementation") != current:
        raise ValueError(
            "pipeline implementation differs from pipeline_config.json; "
            "resume in a new output directory to avoid mixing code versions")


def _optional_path(value: Path | None) -> str | None:
    return None if value is None else str(value.expanduser().resolve())


def _dataset_label(args: argparse.Namespace) -> str:
    if args.dataset_label is not None:
        label = args.dataset_label.strip()
    else:
        label = args.star_path.stem
    if not label:
        raise ValueError("dataset label cannot be empty")
    return label


def _dataset_number(args: argparse.Namespace) -> str:
    """Return the numeric EMPIAR identifier used in stable result names."""
    match = re.fullmatch(r"EMPIAR-(\d+)", _dataset_label(args), re.IGNORECASE)
    if match is None:
        raise ValueError(
            "--dataset-label must have the form EMPIAR-10073 so result "
            "directories can be organized under "
            "results/10073_<SLURM_JOB_ID>/kam_pipeline")
    return match.group(1)


def _execution_id() -> str:
    """Return the Slurm job ID, or a collision-resistant local-run ID."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        return LOCAL_RUN_ID
    job_id = job_id.strip()
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise ValueError("SLURM_JOB_ID must contain only decimal digits")
    return job_id


def _centering_algorithm_configuration(args: argparse.Namespace) -> dict | None:
    """Resolve dataset-centerer defaults from the algorithm's own configuration."""
    if not args.centering:
        return None
    from dataclasses import asdict

    return asdict(CenteringConfig(
        support_radius_fraction=args.centering_radius / args.denoise_res,
        residual_max_shift_fraction=args.centering_max_shift / args.denoise_res,
        ctf_mode=args.centering_ctf_mode,
        wiener_balance=args.centering_wiener_balance,
        contrast_sign=-1 if args.flip_sign else 1,
        device=args.device,
        fle_root=str(getattr(args, "fle_root", DEFAULT_FLE_ROOT)),
    ))


def _configuration(args: argparse.Namespace) -> dict:
    """Return the immutable, dataset-agnostic pipeline configuration."""
    return {
        "pipeline_version": PIPELINE_VERSION,
        "dataset_label": _dataset_label(args),
        "inputs": {
            "star_path": str(args.star_path.expanduser().resolve()),
            "data_folder": _optional_path(args.data_folder),
            "stack_prefix": args.stack_prefix,
            "native_pixel_size": args.native_pixel_size,
        },
        "denoising": {
            "sampling": "all_STAR_rows",
            "seed": args.seed,
            "resolution": args.denoise_res,
            "batch_size": args.batch_size,
            "device": args.device,
            "ctf_mean_defocus_bin_angstrom": args.ctf_mean_defocus_bin,
            "background_radius": args.background_radius,
            "flip_sign": args.flip_sign,
            "centering": {
                "enabled": args.centering,
                "method": CENTERING_METHOD if args.centering else "none",
                "method_version": CENTERING_METHOD_VERSION if args.centering else None,
                "algorithm_configuration": _centering_algorithm_configuration(args),
                "radius_denoise_pixels": args.centering_radius,
                "max_shift_denoise_pixels": args.centering_max_shift,
                "ctf_mode": args.centering_ctf_mode if args.centering else None,
                "wiener_balance": (
                    args.centering_wiener_balance
                    if args.centering and args.centering_ctf_mode == "wiener"
                    else None),
                "max_distance_fraction": args.centering_max_distance_fraction,
                "relion_origins_used": False,
                "orientation_angles_used": False,
                "reference_map_used": False,
            },
        },
        "selection": {
            "n_images": args.n_select,
            "neighbor_distance_fraction": args.neighbor_distance_fraction,
            "neighbor_distance_coordinate_pixels": (
                args.neighbor_distance_coordinate_pixels),
            "neighbor_distance_mode": (
                "explicit RELION-coordinate pixels"
                if args.neighbor_distance_coordinate_pixels is not None else
                "fraction of native particle-box side"),
            "maximum_score_quantile": args.max_score_quantile,
            "preview_images": args.n_preview,
        },
        "moments": {
            "resolution": args.moment_res,
            "basis": "fourier_bessel",
            "compression": "in-plane-invariant steerable FB blocks",
            "dc_handling": "retained",
            "available_orders": [1, 2, 3],
            "n_canonical_m3_entries": args.expected_m3_entries,
            "max_m3_entries": args.max_m3_entries,
        },
        "model": {
            "volume_representation": "spherical Fourier-Bessel",
            "ell_max": args.ell_max,
            "viewing_representation": "positive full-real-SH SOS density",
            "viewing_degree": args.viewing_degree,
            "sphere_node_request": args.sphere_nodes,
            "sphere_degree_request": args.sphere_degree,
            "n_inplane": 1,
            "shift_sigma": 0.0,
        },
        "optimization": {
            "n_initializations": args.n_init,
            "maxiter": args.maxiter,
            "seed": args.seed,
            "orders_fit_independently": True,
            "m3_initialization": "M2 init_i warm-starts M3 init_i",
            "result_selection": "minimum normalized active-moment loss",
        },
    }


def _denoise_command(args, paths):
    command = [
        sys.executable, "-u", "-m", "kam.command_line", "denoise",
        str(args.star_path.expanduser().resolve()),
        "--dataset-label", _dataset_label(args),
        "--seed", str(args.seed),
        "--denoise-res", str(args.denoise_res),
        "--batch-size", str(args.batch_size), "--workers", str(args.workers),
        "--device", args.device,
        "--background-radius", str(args.background_radius),
        "--ctf-mean-defocus-bin", str(args.ctf_mean_defocus_bin),
        "--flip-sign" if args.flip_sign else "--no-flip-sign",
        "--centering" if args.centering else "--no-centering",
        "--outdir", str(paths["denoising"]),
    ]
    if args.centering:
        command.extend(["--centering-radius", str(args.centering_radius)])
        command.extend(["--centering-max-shift", str(args.centering_max_shift)])
        command.extend(["--centering-ctf-mode", args.centering_ctf_mode])
        command.extend(["--centering-wiener-balance",
                        str(args.centering_wiener_balance)])
        command.extend(["--centering-output-dir", str(paths["centering"])])
        if args.centering_max_distance_fraction is not None:
            command.extend(["--centering-max-distance-fraction",
                            str(args.centering_max_distance_fraction)])
    if args.data_folder is not None:
        command.extend(["--datadir", str(args.data_folder.expanduser().resolve())])
    if args.stack_prefix is not None:
        command.extend(["--strip-prefix", args.stack_prefix])
    if args.native_pixel_size is not None:
        command.extend(["--native-pixel-size", str(args.native_pixel_size)])
    return command


def _selection_command(args, paths):
    command = [
        sys.executable, "-u", "-m", "kam.command_line", "select",
        "--denoising-dir", str(paths["denoising"]),
        "--n-select", str(args.n_select), "--n-preview", str(args.n_preview),
        "--seed", str(args.seed), "--batch-size", str(args.batch_size),
        "--max-score-quantile", str(args.max_score_quantile),
        "--outdir", str(paths["selection"]),
    ]
    if args.neighbor_distance_coordinate_pixels is None:
        command.extend([
            "--neighbor-distance-fraction", str(args.neighbor_distance_fraction)])
    else:
        command.extend([
            "--neighbor-distance-coordinate-pixels",
            str(args.neighbor_distance_coordinate_pixels)])
    return command


def _report_value(value):
    """Normalize resolved settings without reading particle data."""
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {str(key): _report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(item) for item in value]
    if isinstance(value, np.generic):
        return _report_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"unsupported hyperparameter report value: {value!r}")


def _hyperparameters(args: argparse.Namespace, paths: dict[str, Path]) -> dict:
    """Record CLI values and resolved stage defaults before processing data."""
    from dataclasses import asdict

    denoise_command = _denoise_command(args, paths)
    denoise_arguments = _denoise_parse_args(
        denoise_command[denoise_command.index("denoise") + 1:])
    denoise_config = FastCWFDenoiseConfig(
        **vars(denoise_arguments), driver_path=_THIS_FILE)
    selection_command = _selection_command(args, paths)
    selection = _selection_parser().parse_args(
        selection_command[selection_command.index("select") + 1:])
    moment_config = postprocessing.MomentPreparationConfig(
        denoising_dir=paths["denoising"],
        selection_file=paths["selection"] / "selection.npz",
        output_dir=paths["prepared"],
        expected_dataset=_dataset_label(args), n_images=args.n_select,
        denoise_res=args.denoise_res, moment_res=args.moment_res,
        batch_size=args.batch_size, device=args.device,
        max_m3_entries=args.max_m3_entries, fb_ell_max=args.ell_max,
        augment_reflections=args.augment_reflections,
    )
    reconstruction = {
        f"M{order}": asdict(ReconstructionConfig(
            prepared_dir=paths["prepared"], output_dir=paths[f"m{order}"],
            order=order, ell_max=args.ell_max,
            viewing_degree=args.viewing_degree, sphere_nodes=args.sphere_nodes,
            sphere_degree=args.sphere_degree,
            n_init=args.n_init,
            maxiter=args.maxiter,
            device=args.device, seed=args.seed, resume=args.resume,
            initial_fit=paths[f"m{order - 1}"] if order > 1 else None,
        ))
        for order in (1, 2, 3)
    }
    environment_names = (
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    return _report_value({
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            {"star_path": "particles", "data_folder": "datadir",
             "results_dir": "outdir", "stack_prefix": "strip_prefix"}.get(key, key): value
            for key, value in vars(args).items()
        },
        "configuration": _configuration(args),
        "resolved_stage_configs": {
            "denoise": asdict(denoise_config),
            "centering": _centering_algorithm_configuration(args),
            "selection": vars(selection),
            "moment_preparation": asdict(moment_config),
            **reconstruction,
        },
        "output_paths": paths,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": Path.cwd(),
            "process_argv": list(sys.argv),
            "requested_device": args.device,
            "workers": args.workers,
            "environment": {name: os.environ[name] for name in environment_names
                            if name in os.environ},
        },
        "implementation": _implementation_fingerprint(),
    })


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    results = args.results_dir.expanduser().resolve()
    dataset_number = _dataset_number(args)
    run_root = results / f"{dataset_number}_{_execution_id()}"
    root = run_root / "kam_pipeline"
    return {
        "run_root": run_root,
        "root": root,
        "manifest": root / "pipeline_config.json",
        "hyperparameters": root / "hyperparameters.json",
        "log": root / "run.log",
        "timing_report": root / "timing_report.json",
        "centering": root / "centering",
        "denoising": root / "denoise",
        "selection": root / "denoise",
        "prepared": root / "MoM" / "prepared",
        "m1": root / "MoM" / "M1",
        "m2": root / "MoM" / "M2",
        "m3": root / "MoM" / "M3",
    }


def _ensure_manifest(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    configuration = _configuration(args)
    manifest_path = paths["manifest"]
    paths["root"].mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        previous = _read_json(manifest_path)
        _validate_pipeline_manifest(previous, configuration)
        return
    # Slurm creates stdout/stderr before the payload starts.  Keeping those
    # logs in the MoM directory must not make a fresh pipeline look like an
    # unrelated, partially populated run.
    def scheduler_log(path: Path) -> bool:
        return (path.is_file() and (path.name == "run.log" or
                (path.name.startswith("slurm_")
                 and path.suffix in {".out", ".err"})))

    unrelated = [
        path for path in paths["root"].iterdir()
        if path != manifest_path and not scheduler_log(path)
    ]
    if unrelated:
        raise FileExistsError(
            f"output directory is nonempty but has no pipeline manifest: "
            f"{paths['root']}")
    value = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "implementation": _implementation_fingerprint(),
    }
    try:
        _write_json_exclusive(manifest_path, value)
    except FileExistsError:
        previous = _read_json(manifest_path)
        try:
            _validate_pipeline_manifest(previous, configuration)
        except ValueError as error:
            raise ValueError(
                "a concurrent invocation created an incompatible manifest") \
                from error


def _run(command: list[str], *, dry_run: bool) -> None:
    print("[pipeline] " + shlex.join(command), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, check=True, cwd=ROOT, env=environment)


def _denoising_complete(path: Path, args: argparse.Namespace) -> bool:
    report_path = path / "report.json"
    manifest_path = path / "manifest.npz"
    stack_path = path / "denoised_images.mrcs"
    if not all(item.is_file() for item in (
            report_path, manifest_path, stack_path,
            path / "denoising_preview.pdf")):
        return False
    report = _read_json(report_path)
    _validate_optional_timing(report, "completed denoising stage")
    _validate_recorded_implementation(
        report.get("implementation"), "completed denoising stage")
    expected_backend = (
        "cupy_cufinufft_fle" if args.device == "gpu"
        else "numpy_scipy_finufft_fle")
    if (report.get("dataset") != _dataset_label(args)
            or int(report.get("denoise_res", -1)) != args.denoise_res
            or report.get("device") != args.device
            or report.get("fastpca_backend") != expected_backend
            or report.get("fle_precision") != "single"
            or bool(report.get("flip_sign")) != args.flip_sign
            or int(report.get("n_images_before_centering_filter", report.get("n_images", -1)))
            != int(report.get("n_total", -2))
            or report.get("sampling") != (
                "all_STAR_rows_then_centering_distance_filter"
                if args.centering_max_distance_fraction is not None else "all_STAR_rows")
            or not math.isclose(
                float(report.get("ctf_mean_defocus_bin_angstrom", math.nan)),
                args.ctf_mean_defocus_bin, rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("completed denoising output does not match pipeline config")
    with np.load(manifest_path, allow_pickle=False) as saved:
        count = int(np.asarray(saved["n_images"]).reshape(()))
        side = int(np.asarray(saved["denoise_res"]).reshape(()))
        stack_size = int(np.asarray(saved["stack_size"]).reshape(()))
        stack_mtime_ns = int(
            np.asarray(saved["stack_mtime_ns"]).reshape(()))
        manifest_device = str(
            np.asarray(saved["device"]).reshape(()).item())
        manifest_backend = str(
            np.asarray(saved["fastpca_backend"]).reshape(()).item())
        manifest_precision = str(
            np.asarray(saved["fle_precision"]).reshape(()).item())
        _validate_denoising_centering(report, saved, args)
    if count != int(report["n_images"]) or side != args.denoise_res:
        raise ValueError("denoising report and manifest disagree")
    if (manifest_device != args.device
            or manifest_backend != expected_backend
            or manifest_precision != "single"):
        raise ValueError(
            "denoising manifest has a different backend or precision")
    stack_stat = stack_path.stat()
    if (stack_stat.st_size != stack_size
            or stack_stat.st_mtime_ns != stack_mtime_ns):
        raise ValueError("denoised image stack changed after its manifest")
    if args.centering:
        _ensure_centering_diagnostic(path, report)
    return stack_stat.st_size > 0


def _ensure_centering_diagnostic(denoising_path, denoising_report):
    """Validate centering provenance and restore missing diagnostic PDFs.

    A restored before/after preview marks the pre-translation estimated-center
    coordinate in both rows, matching newly generated diagnostics.
    """
    directory = denoising_path.parent / "centering"
    expected = {
        "directory": str(directory),
        "preview": str(directory / "before_after.pdf"),
        "preview_cache": str(directory / "preview.npz"),
        "report": str(directory / "report.json"),
        "centers_scatter": str(directory / "estimated_centers.pdf"),
        "estimates": str(directory / "estimates.npz"),
    }
    if denoising_report.get("centering_diagnostic") != expected:
        raise ValueError("denoising lacks its separate centering diagnostics")
    if not all(Path(expected[key]).is_file() for key in ("preview_cache", "report", "estimates")):
        raise FileNotFoundError(f"centering preview cache or report is missing: {directory}")
    report = _read_json(Path(expected["report"]))
    n_images = int(denoising_report["n_images"])
    n_input = int(denoising_report.get("n_images_before_centering_filter", n_images))
    if (report.get("n_images_centered") != n_input
            or report.get("n_images_retained") != n_images
            or report.get("centering_filter") != denoising_report.get("centering_filter")
            or report.get("all_images_entering_covariance_centered") is not True
            or report.get("recentering") != denoising_report.get("recentering")):
        raise ValueError("centering diagnostics disagree with the denoising provenance")
    preview_path = Path(expected["preview"])
    preview_semantics_changed = (
        report.get("preview_marker_semantics")
        != CENTERING_PREVIEW_MARKER_SEMANTICS
    )
    if not preview_path.is_file() or preview_semantics_changed:
        with np.load(expected["preview_cache"], allow_pickle=False) as saved:
            before = np.asarray(saved["before"])
            after = np.asarray(saved["after"])
            cached_centers = (
                np.asarray(saved["estimated_centers_yx_denoise_pixels"])
                if "estimated_centers_yx_denoise_pixels" in saved.files else None
            )
            side = int(denoising_report["denoise_res"])
            if before.shape != (min(8, n_input), side, side):
                raise ValueError("centering preview cache has an unexpected image shape")
        if cached_centers is None:
            with np.load(denoising_path / "manifest.npz", allow_pickle=False) as saved:
                cached_centers = np.asarray(
                    saved["centering_centers_yx_denoise_pixels"]
                )[:len(before)]
        save_centering_preview(
            before, after, preview_path,
            estimated_centers_yx=cached_centers,
        )
        if preview_semantics_changed:
            report["preview_description"] = CENTERING_PREVIEW_DESCRIPTION
            report["preview_marker_semantics"] = CENTERING_PREVIEW_MARKER_SEMANTICS
            _write_json_atomic(Path(expected["report"]), report)
        action = "updated" if preview_semantics_changed else "restored"
        print(f"[pipeline] {action} centering preview: {preview_path}", flush=True)
    scatter_path = Path(expected["centers_scatter"])
    if not scatter_path.is_file():
        from .denoise import save_centering_centers_plot

        with np.load(expected["estimates"], allow_pickle=False) as saved:
            centers = np.asarray(saved["centers_yx"])
            if centers.shape != (n_input, 2):
                raise ValueError("centering scatter requires all input-image centers")
            side = int(denoising_report["denoise_res"])
            save_centering_centers_plot(
                centers, saved["valid"], saved["boundary_hit"],
                (side, side), scatter_path,
                retained=saved["retained"],
                max_distance_pixels=denoising_report["centering_filter"][
                    "max_distance_denoise_pixels"])
        print(f"[pipeline] restored centering scatter: {scatter_path}", flush=True)


def _validate_denoising_centering(report, saved, args) -> None:
    """Never reuse a stack with different centering or pose provenance."""
    expected_method = CENTERING_METHOD if args.centering else "none"
    expected_radius = args.centering_radius if args.centering else None
    expected_distance = args.centering_max_distance_fraction
    if report.get("centering_max_distance_fraction") != expected_distance:
        raise ValueError("completed denoising used a different centering-distance threshold")
    for key in ("relion_poses_used", "relion_orientation_angles_used",
                "relion_origins_used", "recenter_from_star", "reference_used"):
        if (report.get(key) is not False or key not in saved
                or np.asarray(saved[key]).shape != ()
                or np.asarray(saved[key]).dtype.kind != "b"
                or bool(saved[key])):
            raise ValueError(
                f"completed denoising has missing or incompatible provenance: {key}")
    expected_max_shift = args.centering_max_shift if args.centering else None
    expected_ctf_mode = args.centering_ctf_mode if args.centering else None
    expected_balance = (args.centering_wiener_balance
                        if args.centering and expected_ctf_mode == "wiener" else None)
    if (report.get("image_centering_enabled") is not args.centering
            or report.get("centering_method") != expected_method
            or report.get("centering_radius_denoise_pixels") != expected_radius
            or report.get("centering_max_shift_denoise_pixels") != expected_max_shift
            or report.get("centering_ctf_mode") != expected_ctf_mode
            or report.get("centering_wiener_balance") != expected_balance):
        raise ValueError("completed denoising used different image centering settings")
    expected_manifest = {
        "image_centering_enabled": args.centering,
        "centering_method": expected_method,
        "centering_radius_denoise_pixels": (
            expected_radius if args.centering else -1),
        "centering_max_shift_denoise_pixels": (
            -1 if expected_max_shift is None else expected_max_shift),
        "centering_ctf_mode": (
            "none" if expected_ctf_mode is None else expected_ctf_mode),
        "centering_wiener_balance": (
            -1.0 if expected_balance is None else expected_balance),
        "centering_max_distance_fraction": (
            -1.0 if expected_distance is None else expected_distance),
    }
    if args.centering:
        expected_manifest["centering_method_version"] = CENTERING_METHOD_VERSION
        if report.get("centering_method_version") != CENTERING_METHOD_VERSION:
            raise ValueError("completed denoising used an older centering algorithm")
        if report["implementation"].get("centering") != _code_identity(
                PACKAGE_DIR / "centering.py"):
            raise ValueError("completed denoising lacks the current centering implementation")
    for key, expected in expected_manifest.items():
        if (key not in saved or np.asarray(saved[key]).shape != ()
                or np.asarray(saved[key]).item() != expected):
            raise ValueError(f"denoising manifest has different centering metadata: {key}")
    recentering = report.get("recentering")
    if (not isinstance(recentering, dict)
            or recentering.get("method") != expected_method
            or recentering.get("enabled") is not args.centering
            or any(recentering.get(key) is not False for key in (
                "orientation_angles_used", "relion_origins_used", "reference_map_used"))):
        raise ValueError("completed denoising has incompatible centering provenance")
    if args.centering:
        algorithm = recentering.get("algorithm")
        if (not isinstance(algorithm, dict)
                or algorithm.get("configuration") != _centering_algorithm_configuration(args)):
            raise ValueError("completed denoising used different dataset-centering defaults")
    if "recentering" not in saved:
        raise ValueError("denoising manifest lacks centering provenance")
    try:
        manifest_recentering = json.loads(np.asarray(saved["recentering"]).item())
    except (TypeError, ValueError) as error:
        raise ValueError("denoising manifest has invalid centering provenance") from error
    if manifest_recentering != recentering:
        raise ValueError("denoising report and manifest disagree on centering provenance")
    filtering = report.get("centering_filter")
    if (not isinstance(filtering, dict)
            or filtering.get("enabled") is not (expected_distance is not None)
            or filtering.get("max_distance_fraction") != expected_distance
            or filtering.get("n_retained") != int(report["n_images"])
            or filtering.get("n_input") != int(report["n_images_before_centering_filter"])
            or filtering.get("n_input") - filtering.get("n_rejected", -1)
            != filtering.get("n_retained")
            or json.loads(np.asarray(saved["centering_filter"]).item()) != filtering):
        raise ValueError("denoising report and manifest disagree on centering rejection")
    rejected_rows = np.asarray(saved["centering_rejected_source_rows"])
    if (rejected_rows.shape != (filtering["n_rejected"],)
            or np.intersect1d(rejected_rows, saved["source_rows"]).size):
        raise ValueError("discarded particles overlap the retained image stack")
    if args.centering:
        count = int(report["n_images"])
        for key in ("applied_shifts_xy_denoise_pixels",
                    "centering_centers_yx_denoise_pixels"):
            if (key not in saved or np.asarray(saved[key]).shape != (count, 2)
                    or not np.all(np.isfinite(saved[key]))):
                raise ValueError(f"denoising manifest has invalid centering coordinates: {key}")
        # The common estimate and shrunk residuals are generally subpixel.
        # Validate their geometry without rounding them to the legacy integer
        # residual-centerer grid or bounding them by the residual search limit.
        centers = np.asarray(saved["centering_centers_yx_denoise_pixels"])
        shifts = np.asarray(saved["applied_shifts_xy_denoise_pixels"])
        if not np.allclose(centers, args.denoise_res // 2 - shifts[:, ::-1],
                           rtol=0, atol=1e-6):
            raise ValueError("denoising manifest centers disagree with its total applied shifts")
        for key in ("centering_valid", "centering_boundary_hit"):
            if (key not in saved or np.asarray(saved[key]).shape != (count,)
                    or np.asarray(saved[key]).dtype.kind != "b"):
                raise ValueError(f"denoising manifest has invalid centering flags: {key}")
        if expected_distance is not None:
            from .centering import centering_distance_mask

            keep, _ = centering_distance_mask(
                saved["centering_centers_yx_denoise_pixels"], saved["centering_valid"],
                (args.denoise_res, args.denoise_res), expected_distance)
            if not np.all(keep):
                raise ValueError("discarded centering estimates leaked into denoising")


def _selection_complete(path: Path, args: argparse.Namespace) -> bool:
    report_path = path / "selection_report.json"
    archive = path / "selection.npz"
    diagnostic_pdf = path / "selection_diagnostic.pdf"
    if (not report_path.is_file() or not archive.is_file()
            or not diagnostic_pdf.is_file()
            or not (path / "selection_scores.pdf").is_file()):
        return False
    report = _read_json(report_path)
    _validate_optional_timing(report, "completed selection stage")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("completed selection lacks implementation provenance")
    expected_selection_hashes = {
        "selector_sha256": _sha256(_THIS_FILE),
        "selection_module_sha256": _sha256(
            PACKAGE_DIR / "preprocessing.py"),
    }
    for key, expected in expected_selection_hashes.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"completed selection implementation changed: {key}")
    denoising_manifest = path / "manifest.npz"
    if (not denoising_manifest.is_file()
            or report.get("input_manifest_sha256")
            != _sha256(denoising_manifest)):
        raise ValueError(
            "completed selection does not match the current denoising manifest")
    if report.get("selection_sha256") != _sha256(archive):
        raise ValueError("completed selection archive changed after its report")
    if (int(report.get("n_selected", -1)) != args.n_select
            or report.get("dataset") != _dataset_label(args)
            or bool(report.get("uses_reference_map", True))
            or bool(report.get("uses_relion_poses_for_selection", True))):
        raise ValueError("completed selection does not match pipeline config")
    if args.neighbor_distance_coordinate_pixels is not None:
        distance_matches = math.isclose(
            float(report.get(
                "minimum_neighbor_distance_coordinate_pixels", math.nan)),
            args.neighbor_distance_coordinate_pixels,
            rel_tol=0.0, abs_tol=1e-12)
    else:
        distance_matches = math.isclose(
            float(report.get(
                "minimum_neighbor_distance_fraction_of_native_box",
                math.nan)),
            args.neighbor_distance_fraction, rel_tol=0.0, abs_tol=1e-12)
    score_matches = math.isclose(
        float(report.get("maximum_crowding_score_quantile", math.nan)),
        args.max_score_quantile, rel_tol=0.0, abs_tol=1e-12)
    if not distance_matches or not score_matches:
        raise ValueError("completed selection has different selection thresholds")
    with np.load(archive, allow_pickle=False) as saved:
        positions = np.asarray(saved["selected_positions"])
        diagnostic_positions = np.asarray(
            saved["diagnostic_positions"], dtype=np.int64)
        diagnostic_selected = np.asarray(
            saved["diagnostic_selected"], dtype=bool)
    if positions.shape != (args.n_select,) or len(np.unique(positions)) != len(positions):
        raise ValueError("selection archive has invalid particle identities")
    if (diagnostic_positions.shape != (args.n_preview,)
            or diagnostic_selected.shape != (args.n_preview,)
            or len(np.unique(diagnostic_positions)) != args.n_preview):
        raise ValueError("selection archive has an invalid random diagnostic")
    if not np.array_equal(
            diagnostic_selected, np.isin(diagnostic_positions, positions)):
        raise ValueError("selection diagnostic labels disagree with selected particles")
    diagnostic_report = report.get("selection_diagnostic")
    if (not isinstance(diagnostic_report, dict)
            or int(diagnostic_report.get("n_images", -1)) != args.n_preview
            or int(diagnostic_report.get("n_selected", -1))
            != int(np.sum(diagnostic_selected))
            or int(diagnostic_report.get("n_not_selected", -1))
            != int(np.sum(~diagnostic_selected))
            or diagnostic_report.get("sampling")
            != "uniform_without_replacement_from_all_denoised_images"
            or diagnostic_report.get("seed_sequence")
            != [args.seed, 0x53454C]
            or diagnostic_report.get("pdf_sha256") != _sha256(diagnostic_pdf)):
        raise ValueError("selection report has an invalid random diagnostic")
    return True


def _prepared_complete(path: Path, args: argparse.Namespace) -> bool:
    report_path = path / "report.json"
    archive = path / "moments.npz"
    if not report_path.is_file() or not archive.is_file():
        return False
    report = _read_json(report_path)
    _validate_optional_timing(report, "completed moment-preparation stage")
    if report.get("implementation") != \
            _reconstruction_implementation_fingerprint():
        raise ValueError(
            "moment-preparation implementation differs from its report")
    denoising_manifest = path.parents[1] / "denoise" / "manifest.npz"
    selection_archive = path.parents[1] / "denoise" / "selection.npz"
    if (not denoising_manifest.is_file()
            or report.get("denoising_manifest_sha256")
            != _sha256(denoising_manifest)):
        raise ValueError(
            "prepared moments do not match the current denoising manifest")
    if (not selection_archive.is_file()
            or report.get("selection_sha256") != _sha256(selection_archive)):
        raise ValueError(
            "prepared moments do not match the current particle selection")
    if report.get("prepared_sha256") != _sha256(archive):
        raise ValueError("prepared moment archive changed after its report")
    report_moment_res = report.get("moment_res", report.get("side", -1))
    report_basis = report.get("moment_basis")
    if (int(report.get("n_images", -1)) != args.n_select
            or int(report_moment_res) != args.moment_res
            or report.get("dataset") != _dataset_label(args)
            or report_basis != "fb"
            or report.get("fb_ell_max") != args.ell_max
            or bool(report.get("augment_reflections", False))
            is not args.augment_reflections):
        raise ValueError("prepared moments do not match pipeline config")
    with np.load(archive, allow_pickle=False) as saved:
        if (int(np.asarray(saved["n_images"]).reshape(())) != args.n_select
                or int(np.asarray(saved["ds_res"]).reshape(())) != args.moment_res):
            raise ValueError("prepared report and moment archive disagree")
    return True


def _fit_complete(path: Path, args: argparse.Namespace, order: int) -> bool:
    summary_path = path / "summary.json"
    fit_path = path / "fit.npz"
    if not all(item.is_file() for item in (
            summary_path, fit_path, path / "config.json", path / "recon.mrc")):
        return False
    summary = _read_json(summary_path)
    _validate_optional_timing(summary, f"completed M{order} reconstruction")
    if not summary.get("complete", False):
        return False
    if summary.get("implementation") != \
            _reconstruction_implementation_fingerprint():
        raise ValueError(
            f"completed M{order} implementation differs from its summary")
    prepared_archive = path.parent / "prepared" / "moments.npz"
    if (not prepared_archive.is_file()
            or summary.get("prepared_sha256") != _sha256(prepared_archive)):
        raise ValueError(
            f"completed M{order} fit does not match the current moments")
    expected = {
        "order": order,
        "ell_max": args.ell_max,
        "viewing_degree": args.viewing_degree,
        "sphere_node_request": args.sphere_nodes,
        "sphere_degree_request": args.sphere_degree,
        "inplane_angles": 1,
        "n_initializations": args.n_init,
        "maxiter": args.maxiter,
        "initialization_seed": args.seed,
        "initial_fit": (str((path.parent / f"M{order - 1}").resolve())
                        if order > 1 else None),
    }
    mismatch = {key: (summary.get(key), value) for key, value in expected.items()
                if summary.get(key) != value}
    if mismatch:
        raise ValueError(f"completed M{order} fit has different config: {mismatch}")
    if order > 1:
        warm_files = summary.get("initial_fit_files")
        warm_hashes = summary.get("initial_fit_sha256")
        if (not isinstance(warm_files, list)
                or len(warm_files) != args.n_init
                or not isinstance(warm_hashes, list)
                or len(warm_hashes) != args.n_init
                or not summary.get("m3_warm_start_semantics")):
            raise ValueError(
                f"completed M{order} fit lacks one audited M{order - 1} "
                "warm start per init")
        for filename, expected_hash in zip(warm_files, warm_hashes):
            warm_path = Path(filename)
            if (not warm_path.is_file()
                    or _sha256(warm_path) != expected_hash):
                raise ValueError(
                    f"completed M{order} fit has a changed or missing "
                    f"M{order - 1} warm start")
    records = summary.get("fits")
    if not isinstance(records, list) or len(records) != args.n_init:
        raise ValueError(f"completed M{order} fit has an invalid start table")
    objectives = np.asarray([record["objective"] for record in records], dtype=float)
    if not np.all(np.isfinite(objectives)):
        raise ValueError(f"completed M{order} fit contains nonfinite losses")
    best_position = int(np.argmin(objectives))
    best_index = int(records[best_position]["init_index"])
    if int(summary.get("best_init_by_loss", -1)) != best_index:
        raise ValueError(f"M{order} result was not selected by minimum moment loss")
    with np.load(fit_path, allow_pickle=False) as saved:
        saved_objective = float(np.asarray(saved["objective"]).reshape(()))
    if not math.isclose(saved_objective, float(objectives[best_position]),
                        rel_tol=1e-11, abs_tol=1e-14):
        raise ValueError(f"top-level M{order} fit is not the minimum-loss start")
    best_fit = path / f"init_{best_index}" / "fit.npz"
    if not best_fit.is_file() or _sha256(best_fit) != _sha256(fit_path):
        raise ValueError(
            f"top-level M{order} fit is not the archived minimum-loss fit")
    return True


def _nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _stage_report_path(paths: dict[str, Path], name: str) -> Path:
    if name == "denoise":
        return paths["denoising"] / "report.json"
    if name == "select":
        return paths["selection"] / "selection_report.json"
    if name == "prepare":
        return paths["prepared"] / "report.json"
    if name in {"m1", "m2", "m3"}:
        return paths[name] / "summary.json"
    raise ValueError(f"unknown pipeline stage {name!r}")


def _stage_complete(name: str, paths: dict[str, Path],
                    args: argparse.Namespace) -> bool:
    if name == "denoise":
        return _denoising_complete(paths["denoising"], args)
    if name == "select":
        return _selection_complete(paths["selection"], args)
    if name == "prepare":
        return _prepared_complete(paths["prepared"], args)
    if name in {"m1", "m2", "m3"}:
        return _fit_complete(paths[name], args, int(name[1]))
    raise ValueError(f"unknown pipeline stage {name!r}")


def _annotate_new_stage_timing(
        report_path: Path, observed_elapsed: float,
        started_utc: str, finished_utc: str) -> None:
    """Complete and validate timing metadata for a newly finished stage."""
    record = _read_json(report_path)
    record.setdefault("elapsed_seconds", float(observed_elapsed))
    record.setdefault(
        "timings_seconds", {"total": float(record["elapsed_seconds"])})
    record["pipeline_observed_elapsed_seconds"] = float(observed_elapsed)
    record["pipeline_stage_started_utc"] = started_utc
    record["pipeline_stage_finished_utc"] = finished_utc
    _validate_optional_timing(record, str(report_path))
    _write_json_atomic(report_path, record)


def _write_timing_report(
        args: argparse.Namespace, paths: dict[str, Path],
        statuses: dict[str, str], observed: dict[str, float],
        started_utc: str, finished_utc: str,
        invocation_elapsed: float) -> None:
    """Aggregate component reports into one durable root timing report."""
    names = ("denoise", "select", "prepare", "m1", "m2", "m3")
    stage_reports = {}
    for name in names:
        report_path = _stage_report_path(paths, name)
        record = _read_json(report_path) if report_path.is_file() else None
        if record is not None:
            _validate_optional_timing(record, str(report_path))
        elapsed = None if record is None else record.get("elapsed_seconds")
        timings = None if record is None else record.get("timings_seconds")
        stage_reports[name] = {
            "status": statuses.get(
                name, "available_before_invocation"
                if record is not None else "not_available"),
            "report": str(report_path),
            "report_available": record is not None,
            "elapsed_seconds": (
                None if elapsed is None else float(elapsed)),
            "timings_seconds": timings,
            "pipeline_observed_elapsed_seconds": observed.get(name),
        }

    stage_elapsed = {
        name: details["elapsed_seconds"]
        for name, details in stage_reports.items()
    }
    available_elapsed = [
        value for value in stage_elapsed.values() if value is not None]
    requested_names = set(statuses)
    requested_elapsed = [
        stage_elapsed[name] for name in requested_names
        if stage_elapsed[name] is not None]
    invocation = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "full_invocation_elapsed_seconds": float(invocation_elapsed),
        "requested_orders": [1, 2, 3],
        "resume": bool(args.resume),
        "stage_status": dict(statuses),
        "pipeline_observed_stage_elapsed_seconds": dict(observed),
    }
    previous_history = []
    timing_path = paths["timing_report"]
    if timing_path.is_file():
        previous = _read_json(timing_path)
        history = previous.get("invocations", [])
        if isinstance(history, list):
            previous_history = history
    report = {
        "timing_report_version": 1,
        "dataset": _dataset_label(args),
        "pipeline_output": str(paths["root"]),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "full_invocation_elapsed_seconds": float(invocation_elapsed),
        "stage_elapsed_seconds": stage_elapsed,
        "sum_available_stage_elapsed_seconds": float(sum(available_elapsed)),
        "sum_requested_stage_elapsed_seconds": float(sum(requested_elapsed)),
        "stages": stage_reports,
        "latest_invocation": invocation,
        "invocations": [*previous_history, invocation],
    }
    _write_json_atomic(timing_path, report)
    compact = ", ".join(
        f"{name}=" + (
            "n/a" if stage_elapsed[name] is None
            else f"{stage_elapsed[name]:.1f}s")
        for name in names)
    print(
        f"[timing] {compact}; stage-sum={sum(available_elapsed):.1f}s; "
        f"this invocation={invocation_elapsed:.1f}s",
        flush=True,
    )
    print(f"[timing] report={timing_path}", flush=True)


def _skip_or_require_clean(name: str, path: Path, complete: bool,
                           args: argparse.Namespace, *, resumable=False) -> bool:
    if complete:
        if args.resume:
            print(f"[pipeline] {name}: complete, skipping", flush=True)
            return True
        raise FileExistsError(f"{name} output already exists: {path}")
    if _nonempty(path) and not resumable:
        raise RuntimeError(
            f"{name} output is incomplete and will not be overwritten: {path}. "
            "Inspect it, then use a new output directory or remove only that "
            "incomplete stage yourself.")
    return False


def _denoise(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    complete = _denoising_complete(paths["denoising"], args)
    if _skip_or_require_clean("denoising", paths["denoising"], complete, args):
        return
    _run(_denoise_command(args, paths), dry_run=args.dry_run)
    if not args.dry_run and not _denoising_complete(paths["denoising"], args):
        raise RuntimeError("denoising returned without complete outputs")


def _select(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if not args.dry_run and not _denoising_complete(paths["denoising"], args):
        raise FileNotFoundError("selection requires a completed denoising stage")
    complete = _selection_complete(paths["selection"], args)
    if complete and _skip_or_require_clean(
            "selection", paths["selection"], complete, args):
        return
    _require_selection_outputs_absent(paths["selection"])
    _run(_selection_command(args, paths), dry_run=args.dry_run)
    if not args.dry_run and not _selection_complete(paths["selection"], args):
        raise RuntimeError("selection returned without complete outputs")


def _prepare(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if not args.dry_run:
        if not _denoising_complete(paths["denoising"], args):
            raise FileNotFoundError("moment preparation requires denoising outputs")
        if not _selection_complete(paths["selection"], args):
            raise FileNotFoundError("moment preparation requires selection outputs")
    complete = _prepared_complete(paths["prepared"], args)
    if _skip_or_require_clean("moment preparation", paths["prepared"],
                              complete, args):
        return
    description = (
        "prepare_moments(denoising={!s}, selection={!s}, output={!s}, "
        "n_images={}, denoise_res={}, moment_res={}, max_m3_entries={}, "
        "fb_ell_max={})"
    ).format(paths["denoising"], paths["selection"] / "selection.npz",
             paths["prepared"], args.n_select, args.denoise_res,
             args.moment_res, args.max_m3_entries, args.ell_max)
    print("[pipeline] " + description, flush=True)
    if args.dry_run:
        return
    from .postprocessing import MomentPreparationConfig, prepare_moments

    prepare_moments(MomentPreparationConfig(
        denoising_dir=paths["denoising"],
        selection_file=paths["selection"] / "selection.npz",
        output_dir=paths["prepared"],
        expected_dataset=_dataset_label(args), n_images=args.n_select,
        denoise_res=args.denoise_res, moment_res=args.moment_res,
        batch_size=args.batch_size, device=args.device,
        max_m3_entries=args.max_m3_entries, fb_ell_max=args.ell_max,
        augment_reflections=args.augment_reflections,
    ))
    if not _prepared_complete(paths["prepared"], args):
        raise RuntimeError("moment preparation returned without complete outputs")


def _reconstruct(args: argparse.Namespace, paths: dict[str, Path],
                 order: int) -> None:
    if not args.dry_run and not _prepared_complete(paths["prepared"], args):
        raise FileNotFoundError("reconstruction requires prepared moments")
    output = paths[f"m{order}"]
    initial_fit = None
    if order > 1:
        initial_fit = paths[f"m{order - 1}"]
        if not args.dry_run and not _fit_complete(initial_fit, args, order - 1):
            raise FileNotFoundError(
                f"M{order} requires a completed compatible M{order - 1} "
                f"reconstruction in {initial_fit}")
    complete = _fit_complete(output, args, order)
    if _skip_or_require_clean(
            f"M{order} reconstruction", output, complete, args,
            resumable=args.resume):
        return
    description = (
        "reconstruct_single_moment(prepared={!s}, output={!s}, order={}, "
        "L={}, viewing_degree={}, sphere_request={}, sphere_degree={}, "
        "n_inplane=1, n_init={}, maxiter={}, initial_fit={!s})"
    ).format(paths["prepared"], output, order, args.ell_max,
             args.viewing_degree, args.sphere_nodes, args.sphere_degree,
             args.n_init, args.maxiter, initial_fit)
    print("[pipeline] " + description, flush=True)
    if args.dry_run:
        return
    from .postprocessing import ReconstructionConfig, reconstruct_single_moment

    reconstruct_single_moment(ReconstructionConfig(
        prepared_dir=paths["prepared"], output_dir=output, order=order,
        ell_max=args.ell_max, viewing_degree=args.viewing_degree,
        sphere_nodes=args.sphere_nodes, sphere_degree=args.sphere_degree,
        n_init=args.n_init, maxiter=args.maxiter, device=args.device,
        seed=args.seed, resume=args.resume,
        initial_fit=initial_fit,
    ))
    if not _fit_complete(output, args, order):
        raise RuntimeError(f"M{order} reconstruction returned incomplete outputs")


def _resolve_particles_argument(parser, args):
    legacy = vars(args).pop("legacy_star_path", None)
    if legacy is not None:
        if args.star_path is not None:
            parser.error("provide particles either positionally or with --star-path, not both")
        args.star_path = legacy
    if args.star_path is None:
        parser.error("particles STAR file is required")


def _add_centering_arguments(parser):
    parser.add_argument(
        "--centering", action=argparse.BooleanOptionalAction, default=False,
        help=("estimate a guarded dataset-wide shift plus replica-calibrated, "
              "shrunk individual residuals before denoising, without RELION "
              "poses, shifts or reference maps; disabled by default"))
    parser.add_argument(
        "--centering-radius", type=int,
        help=("individual residual-centering support radius in denoising-grid "
              "pixels (default: denoise-res // 4)"))
    parser.add_argument(
        "--centering-max-shift", type=int,
        help=("bound on each individual residual-shift axis in denoising-grid "
              "pixels (default: max(2, centering-radius // 4)); this does not "
              "bound the separate guarded common-shift search"))
    parser.add_argument(
        "--centering-ctf-mode", choices=("flip", "wiener"), default=None,
        help=("individual residual-estimator CTF handling: 'flip' corrects "
              "only the transfer sign; 'wiener' (default) also restores the CTF-suppressed "
              "low-frequency mass with H/(H^2+balance)"))
    parser.add_argument(
        "--centering-wiener-balance", type=float, default=None,
        help="noise-to-signal balance of the Wiener weight (default: 0.05)")
    parser.add_argument(
        "--centering-max-distance-fraction", type=float, default=None,
        help=("discard invalid total centers and centers farther than this "
              "fraction of image side from box center, after common plus "
              "shrunk residual estimation and before PSD/covariance estimation; "
              "omit to keep all centered images"))


def _resolve_centering_arguments(parser, args):
    if not args.centering:
        if args.centering_radius is not None:
            parser.error("--centering-radius cannot be combined with --no-centering")
        if args.centering_max_shift is not None:
            parser.error("--centering-max-shift requires --centering")
        if args.centering_ctf_mode is not None:
            parser.error("--centering-ctf-mode requires --centering")
        if args.centering_wiener_balance is not None:
            parser.error("--centering-wiener-balance requires --centering")
        args.centering_ctf_mode = "wiener"
        args.centering_wiener_balance = 0.05
        if args.centering_max_distance_fraction is not None:
            parser.error("--centering-max-distance-fraction requires --centering")
        return
    if args.centering_radius is None:
        args.centering_radius = args.denoise_res // 4
    if not 1 <= args.centering_radius <= (args.denoise_res - 1) // 2:
        parser.error("centering-radius must be positive and its full support disk must fit denoise-res")
    if args.centering_max_shift is None:
        args.centering_max_shift = max(2, args.centering_radius // 4)
    largest = args.denoise_res // 2 - args.centering_radius
    if not 1 <= args.centering_max_shift <= largest:
        parser.error(f"centering-max-shift must be in [1, {largest}] so every "
                     "candidate disk fits inside the image")
    if args.centering_ctf_mode is None:
        args.centering_ctf_mode = "wiener"
    if args.centering_wiener_balance is None:
        args.centering_wiener_balance = 0.05
    if (not math.isfinite(args.centering_wiener_balance)
            or args.centering_wiener_balance <= 0):
        parser.error("centering-wiener-balance must be finite and positive")
    fraction = args.centering_max_distance_fraction
    if fraction is not None and (not math.isfinite(fraction) or not 0 < fraction <= 0.5):
        parser.error("centering-max-distance-fraction must be in (0, 0.5]")


def _pipeline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kam pipeline", description=__doc__, allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("star_path", metavar="particles", nargs="?", type=Path,
                        help="input particles STAR file (including CTF/optics metadata)")
    parser.add_argument("--star-path", dest="legacy_star_path", type=Path,
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--outdir", "-o", "--results-dir", dest="results_dir", type=Path,
        help=("parent results directory; writes "
              "<EMPIAR-number>_<SLURM-job-id>/kam_pipeline inside it"))
    parser.add_argument("--dataset-label",
                        help="provenance label only; never changes the algorithm")
    parser.add_argument("--datadir", "--data-folder", dest="data_folder", type=Path)
    parser.add_argument("--strip-prefix", "--stack-prefix", dest="stack_prefix")
    parser.add_argument("--native-pixel-size", type=float)
    parser.add_argument("--n-select", type=int, default=DEFAULT_N_SELECT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--denoise-res", type=int, default=DEFAULT_DENOISE_RES,
        help="side length used by FastPCA denoising")
    _add_centering_arguments(parser)
    parser.add_argument(
        "--moment-res", type=int, default=DEFAULT_MOMENT_RES,
        help="side length of images used to form complete moments")
    parser.add_argument(
        "--vol-degree", dest="ell_max", metavar="VOL_DEGREE", type=int,
        default=DEFAULT_ELL_MAX,
        help="maximum SphFB angular degree of the volume")
    parser.add_argument(
        "--viewing-degree", type=int, choices=(0, 2, 4),
        default=DEFAULT_VIEWING_DEGREE,
        help="degree of the positive spherical-harmonic viewing density")
    parser.add_argument("--sphere-nodes", type=int,
                        default=DEFAULT_SPHERE_NODES,
                        help="requested sphere-rule size; actual size is reported")
    parser.add_argument(
        "--sphere-degree", default="auto",
        help=("select the smallest positive-weight quadrature of this degree; "
              "'auto' (the default) uses 3*vol_degree + viewing_degree, the "
              "exactness requirement of the M3 model, and 'none' selects by "
              "--sphere-nodes instead"))
    parser.add_argument(
        "--n-init", type=int, default=DEFAULT_N_INIT,
        help=("random M2 starts; M2 init_i then warm-starts the paired "
              "M3 init_i"))
    parser.add_argument(
        "--maxiter", type=int, default=DEFAULT_MAXITER,
        help="maximum L-BFGS iterations per initialization")
    parser.add_argument(
        "--device", choices=("cpu", "gpu"), default="gpu",
        help="backend for FastPCA FLE transforms and MoM computation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--ctf-mean-defocus-bin", type=float, default=50.0)
    parser.add_argument("--background-radius", type=float, default=0.9)
    parser.add_argument("--flip-sign", action=argparse.BooleanOptionalAction,
                        default=True)
    neighbor_distance = parser.add_mutually_exclusive_group()
    neighbor_distance.add_argument(
        "--neighbor-distance-fraction", type=float,
        default=DEFAULT_NEIGHBOR_FRACTION,
        help="minimum pick separation as a fraction of native box side")
    neighbor_distance.add_argument(
        "--neighbor-distance-coordinate-pixels", type=float,
        help=("explicit minimum pick separation in RELION coordinate-column "
              "units; overrides the fractional default"))
    parser.add_argument("--max-score-quantile", type=float, default=0.95)
    parser.add_argument("--n-preview", type=int, default=100)
    parser.add_argument("--max-m3-entries", type=int,
                        default=DEFAULT_MAX_M3_ENTRIES)
    parser.add_argument(
        "--augment-reflections", action=argparse.BooleanOptionalAction,
        default=False,
        help=("average every image with its mirror when forming moments "
              "(implemented exactly as the real part of the FB moments); "
              "the fit then compares real parts on both sides, equivalent "
              "to an antipodally symmetric viewing density"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_sphere_degree(parser, args):
    """Resolve --sphere-degree: 'auto' matches the M3 exactness requirement."""
    requested = args.sphere_degree
    if requested is None or str(requested).lower() in ("none", "off"):
        args.sphere_degree = None
        return
    if str(requested).lower() == "auto":
        args.sphere_degree = 3 * args.ell_max + args.viewing_degree
        return
    try:
        args.sphere_degree = int(requested)
    except ValueError:
        parser.error("--sphere-degree must be an integer, 'auto', or 'none'")


def _validate_args(parser: argparse.ArgumentParser,
                   args: argparse.Namespace) -> None:
    _resolve_particles_argument(parser, args)
    _resolve_sphere_degree(parser, args)
    if args.star_path is None or args.results_dir is None:
        parser.error("particles and --outdir are required")
    try:
        _dataset_number(args)
    except ValueError as error:
        parser.error(str(error))
    if args.n_select < 1:
        parser.error("n-select must be positive")
    if args.n_preview < 1 or args.batch_size < 1 or args.workers < 1:
        parser.error("preview, batch, and worker counts must be positive")
    if (args.denoise_res < 4 or args.denoise_res % 2
            or args.moment_res < 4 or args.moment_res % 2
            or args.moment_res > args.denoise_res):
        parser.error(
            "denoise-res and moment-res must be even and >=4, with moment-res "
            "not exceeding denoise-res")
    _resolve_centering_arguments(parser, args)
    if (args.ell_max < 0 or args.sphere_nodes < 1
            or args.n_init < 1 or args.maxiter < 1
            or args.max_m3_entries < 1):
        parser.error("model, quadrature, initialization, and iteration counts are invalid")
    if (not 0 < args.background_radius < math.sqrt(2)
            or args.ctf_mean_defocus_bin < 0
            or args.neighbor_distance_fraction <= 0
            or (args.neighbor_distance_coordinate_pixels is not None
                and args.neighbor_distance_coordinate_pixels <= 0)
            or not 0 < args.max_score_quantile <= 1):
        parser.error("continuous hyperparameters are outside their valid ranges")
    if args.native_pixel_size is not None and args.native_pixel_size <= 0:
        parser.error("native pixel size must be positive")
    from .moments import FBMomentBasis
    try:
        args.expected_m3_entries = FBMomentBasis(
            args.moment_res, args.ell_max).m3_size
    except ValueError as error:
        parser.error(str(error))
    if args.expected_m3_entries > args.max_m3_entries:
        parser.error(
            f"side-{args.moment_res} Fourier-Bessel M3 has "
            f"{args.expected_m3_entries:,} entries, exceeding "
            f"--max-m3-entries={args.max_m3_entries:,}")
    if not args.dry_run:
        if not args.star_path.expanduser().is_file():
            parser.error(f"STAR file does not exist: {args.star_path}")
        if args.data_folder is not None and not args.data_folder.expanduser().is_dir():
            parser.error(f"data folder does not exist: {args.data_folder}")


def pipeline_main(argv=None) -> None:
    parser = _pipeline_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    paths = _paths(args)
    invocation_started_utc = datetime.now(timezone.utc).isoformat()
    invocation_started = time.monotonic()
    if args.dry_run:
        print(json.dumps(_configuration(args), indent=2), flush=True)
        _execute_pipeline(args, paths, invocation_started_utc, invocation_started)
        return
    with capture_run_log(paths["log"]):
        print("\n" + "=" * 70, flush=True)
        print(f"[{invocation_started_utc}] PIPELINE START {_dataset_label(args)} "
              f"pid={os.getpid()} slurm_job={os.environ.get('SLURM_JOB_ID', 'none')}",
              flush=True)
        print(f"[pipeline] run_root={paths['run_root']}", flush=True)
        print(f"[pipeline] output={paths['root']} resume={args.resume}", flush=True)
        print(
            f"[pipeline] centering={args.centering} "
            f"method={CENTERING_METHOD if args.centering else 'none'} "
            f"residual_radius_denoise_pixels={args.centering_radius} "
            f"residual_max_shift_denoise_pixels={args.centering_max_shift}; "
            "RELION origins/poses and reference maps are not used for centering",
            flush=True)
        try:
            _ensure_manifest(args, paths)
            parameters = _hyperparameters(args, paths)
            parameters["started_utc"] = invocation_started_utc
            _write_json_atomic(paths["hyperparameters"], parameters)
            print(f"[pipeline] hyperparameters={paths['hyperparameters']}", flush=True)
            print(json.dumps(parameters["arguments"], indent=2), flush=True)
            _execute_pipeline(args, paths, invocation_started_utc, invocation_started)
        except BaseException:
            print(f"[{datetime.now(timezone.utc).isoformat()}] PIPELINE FAILED "
                  f"elapsed={time.monotonic() - invocation_started:.1f}s",
                  file=sys.stderr, flush=True)
            traceback.print_exc()
            raise
        print(f"[{datetime.now(timezone.utc).isoformat()}] PIPELINE COMPLETE "
              f"elapsed={time.monotonic() - invocation_started:.1f}s",
              flush=True)


def _execute_pipeline(args, paths, invocation_started_utc, invocation_started):
    statuses = {}
    observed = {}

    def run_stage(name, function) -> None:
        stage_started_utc = datetime.now(timezone.utc).isoformat()
        stage_started = time.monotonic()
        print(f"[{stage_started_utc}] STEP {len(statuses) + 1}/5 START {name}",
              flush=True)
        stop_heartbeat = threading.Event()

        def heartbeat():
            while not stop_heartbeat.wait(PIPELINE_HEARTBEAT_SECONDS):
                print(f"[{datetime.now(timezone.utc).isoformat()}] "
                      f"{name} RUNNING elapsed={time.monotonic() - stage_started:.1f}s",
                      flush=True)

        monitor = None
        if not args.dry_run:
            monitor = threading.Thread(target=heartbeat, daemon=True)
            monitor.start()
        try:
            already_complete = (
                False if args.dry_run else _stage_complete(name, paths, args))
            had_partial_output = (
                False if args.dry_run
                else (any((paths["selection"] / item).exists()
                          for item in SELECTION_OUTPUT_NAMES)
                      if name == "select"
                      else _nonempty(_stage_report_path(paths, name).parent)))
            function()
            stage_elapsed = time.monotonic() - stage_started
            observed[name] = float(stage_elapsed)
            if args.dry_run:
                statuses[name] = "dry_run"
            else:
                stage_finished_utc = datetime.now(timezone.utc).isoformat()
                if already_complete:
                    statuses[name] = "skipped_completed"
                else:
                    statuses[name] = (
                        "resumed_and_completed" if had_partial_output
                        else "completed_this_invocation")
                    _annotate_new_stage_timing(
                        _stage_report_path(paths, name), stage_elapsed,
                        stage_started_utc, stage_finished_utc)
                if not _stage_complete(name, paths, args):
                    raise RuntimeError(f"{name} lacks complete outputs after its stage")
        except BaseException:
            print(f"[{datetime.now(timezone.utc).isoformat()}] STEP FAILED {name} "
                  f"elapsed={time.monotonic() - stage_started:.1f}s",
                  file=sys.stderr, flush=True)
            raise
        finally:
            stop_heartbeat.set()
            if monitor is not None:
                monitor.join()
        print(f"[{datetime.now(timezone.utc).isoformat()}] STEP END {name} "
              f"status={statuses[name]} elapsed={observed[name]:.1f}s", flush=True)

    run_stage("denoise", lambda: _denoise(args, paths))
    run_stage("select", lambda: _select(args, paths))
    run_stage("prepare", lambda: _prepare(args, paths))
    run_stage("m1", lambda: _reconstruct(args, paths, 1))
    run_stage("m2", lambda: _reconstruct(args, paths, 2))
    run_stage("m3", lambda: _reconstruct(args, paths, 3))
    if not args.dry_run:
        invocation_finished_utc = datetime.now(timezone.utc).isoformat()
        _write_timing_report(
            args, paths, statuses, observed,
            invocation_started_utc, invocation_finished_utc,
            time.monotonic() - invocation_started)
        _write_json_atomic(paths["root"] / "pipeline_report.json", {
            "complete": True,
            "dataset": _dataset_label(args),
            "initializer": str(paths["m3"] / "recon.mrc"),
            "reconstructions": {
                "M2": str(paths["m2"] / "recon.mrc"),
                "M3": str(paths["m3"] / "recon.mrc"),
            },
            "diagnostics": {
                **({"centering_before_after": str(paths["centering"] / "before_after.pdf"),
                    "centering_estimated_centers": str(paths["centering"] / "estimated_centers.pdf")}
                   if args.centering else {}),
                "denoising_preview": str(paths["denoising"] / "denoising_preview.pdf"),
                "selection_diagnostic": str(paths["selection"] / "selection_diagnostic.pdf"),
                "selection_scores": str(paths["selection"] / "selection_scores.pdf"),
            },
            "timing_report": str(paths["timing_report"]),
            "hyperparameters": str(paths["hyperparameters"]),
            "log": str(paths["log"]),
        })
    print(f"[pipeline] finished M2 then M3; output={paths['root']}",
          flush=True)



# ===========================================================================
# FastPCA denoising subcommand (was denoise_cli.py)
# ===========================================================================

def _denoise_parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("star_path", metavar="particles", nargs="?", type=Path,
                        help="input particles STAR file")
    parser.add_argument("--star-path", dest="legacy_star_path", type=Path,
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--datadir", "--data-folder", dest="data_folder",
        type=Path,
        help="Root for relative stack paths; default is the STAR directory",
    )
    parser.add_argument(
        "--strip-prefix", "--stack-prefix", dest="stack_prefix",
        help="Optional leading relative directory to strip from STAR stack paths",
    )
    parser.add_argument(
        "--native-pixel-size",
        type=float,
        help="Optional A/pixel override; default uses RELION metadata",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-res", type=int, default=64)
    _add_centering_arguments(parser)
    parser.add_argument("--centering-output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", choices=("cpu", "gpu"), default="gpu",
        help="FLE/NUFFT backend used by FastPCA denoising",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Parallel native-image readers per bounded loading batch",
    )
    parser.add_argument("--outdir", "-o", "--output-dir", dest="output_dir",
                        type=Path, required=True)
    parser.add_argument("--background-radius", type=float, default=0.9)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument(
        "--ctf-mean-defocus-bin",
        type=float,
        default=50.0,
        metavar="ANGSTROM",
        help="Nearest-bin radial mean defocus; zero keeps exact means",
    )
    parser.add_argument(
        "--flip-sign",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip raw-image sign inside FastPCA (default: enabled)",
    )
    args = parser.parse_args(argv)
    args.n_images = 0
    _resolve_particles_argument(parser, args)
    _resolve_centering_arguments(parser, args)
    del args.centering
    try:
        FastCWFDenoiseConfig(
            dataset_label=args.dataset_label,
            star_path=args.star_path,
            data_folder=args.data_folder,
            stack_prefix=args.stack_prefix,
            native_pixel_size=args.native_pixel_size,
            n_images=args.n_images,
            seed=args.seed,
            denoise_res=args.denoise_res,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            output_dir=args.output_dir,
            background_radius=args.background_radius,
            eps=args.eps,
            ctf_mean_defocus_bin=args.ctf_mean_defocus_bin,
            flip_sign=args.flip_sign,
            centering_radius=args.centering_radius,
            centering_max_shift=args.centering_max_shift,
            centering_ctf_mode=args.centering_ctf_mode,
            centering_wiener_balance=args.centering_wiener_balance,
            centering_output_dir=args.centering_output_dir,
            centering_max_distance_fraction=args.centering_max_distance_fraction,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return args


def denoise_main(argv=None):
    args = _denoise_parse_args(argv)
    return denoise_relion_particles(
        FastCWFDenoiseConfig(
            dataset_label=args.dataset_label,
            star_path=args.star_path,
            data_folder=args.data_folder,
            stack_prefix=args.stack_prefix,
            native_pixel_size=args.native_pixel_size,
            n_images=args.n_images,
            seed=args.seed,
            denoise_res=args.denoise_res,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            output_dir=args.output_dir,
            background_radius=args.background_radius,
            eps=args.eps,
            ctf_mean_defocus_bin=args.ctf_mean_defocus_bin,
            flip_sign=args.flip_sign,
            centering_radius=args.centering_radius,
            centering_max_shift=args.centering_max_shift,
            centering_ctf_mode=args.centering_ctf_mode,
            centering_wiener_balance=args.centering_wiener_balance,
            centering_output_dir=args.centering_output_dir,
            centering_max_distance_fraction=args.centering_max_distance_fraction,
            driver_path=Path(__file__),
        )
    )



# ===========================================================================
# Low-crowding selection subcommand (was selection_cli.py)
# ===========================================================================

def scalar(manifest, name):
    """Read one scalar from an NPZ manifest."""
    if name not in manifest:
        raise ValueError(f"denoising manifest is missing {name!r}")
    value = np.asarray(manifest[name])
    if value.size != 1:
        raise ValueError(f"manifest field {name!r} must be scalar")
    return value.reshape(()).item()


def optional_text(manifest, name):
    """Return an optional scalar path/prefix, accepting NPZ spellings of None."""
    if name not in manifest:
        return None
    value = scalar(manifest, name)
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "none", "null"} else text


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_manifest_path(text, base):
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=True)


def viewing_tv(particles, source_rows, selected_mask):
    """Compute an optional post-hoc viewing diagnostic."""
    required = ("_rlnAngleRot", "_rlnAngleTilt")
    if any(name not in particles.metadata for name in required):
        return None
    rot = np.mod(particles.column("_rlnAngleRot")[source_rows], 360.0)
    tilt = np.deg2rad(particles.column("_rlnAngleTilt")[source_rows])
    azimuth_bin = np.minimum((rot / 30.0).astype(int), 11)
    polar_bin = np.minimum(((np.cos(tilt) + 1.0) * 3.0).astype(int), 5)
    bins = polar_bin * 12 + azimuth_bin
    full = np.bincount(bins, minlength=72).astype(float)
    selected = np.bincount(bins[selected_mask], minlength=72).astype(float)
    full /= full.sum()
    selected /= selected.sum()
    return float(0.5 * np.sum(np.abs(selected - full)))


def save_selection_diagnostic(images, selected, path, scores=None):
    """Save a random audit montage with a check or cross beside each image."""
    images = np.asarray(images)
    selected = np.asarray(selected, dtype=bool)
    if (images.ndim != 3 or images.shape[1] != images.shape[2]
            or selected.shape != (len(images),) or len(images) < 1):
        raise ValueError("diagnostic images and selection labels are incompatible")
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(images),) or not np.all(np.isfinite(scores)):
            raise ValueError("diagnostic scores must be finite, one per image")

    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))
    figure = plt.figure(figsize=(1.45 * columns, 1.25 * rows))
    outer = figure.add_gridspec(
        rows, columns, left=0.005, right=0.995, bottom=0.005, top=0.995,
        wspace=0.08, hspace=0.08)
    for position in range(rows * columns):
        row, column = divmod(position, columns)
        if position >= len(images):
            axis = figure.add_subplot(outer[row, column])
            axis.set_axis_off()
            continue
        inner = outer[row, column].subgridspec(
            1, 2, width_ratios=(5.0, 1.0), wspace=0.02)
        image_axis = figure.add_subplot(inner[0, 0])
        mark_axis = figure.add_subplot(inner[0, 1])
        image = images[position]
        color = "#16883b" if selected[position] else "#c62828"
        show_particle_image(image_axis, image)
        image_axis.add_patch(Rectangle(
            (-0.5, -0.5), image.shape[1], image.shape[0], fill=False,
            edgecolor=color, linewidth=1.8))
        image_axis.set_axis_off()
        if scores is not None:
            image_axis.text(
                0.03, 0.03, f"{scores[position]:.3f}",
                transform=image_axis.transAxes, fontsize=7, color="white",
                bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none",
                      "pad": 1})
        mark_axis.set_xlim(0.0, 1.0)
        mark_axis.set_ylim(0.0, 1.0)
        mark_axis.set_aspect("equal")
        mark_axis.set_axis_off()
        if selected[position]:
            mark_axis.plot(
                (0.10, 0.38, 0.90), (0.50, 0.20, 0.82), color=color,
                linewidth=3.2, solid_capstyle="round", solid_joinstyle="round")
        else:
            mark_axis.plot(
                (0.16, 0.84), (0.16, 0.84), color=color, linewidth=3.2,
                solid_capstyle="round")
            mark_axis.plot(
                (0.16, 0.84), (0.84, 0.16), color=color, linewidth=3.2,
                solid_capstyle="round")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(figure)


SELECTION_OUTPUT_NAMES = (
    "selection.npz", "selection_report.json", "selected_particles.tsv",
    "selection_diagnostic.pdf", "selection_scores.pdf",
)


def _require_selection_outputs_absent(output: Path) -> None:
    """Allow denoising files here without overwriting a previous selection."""
    existing = [name for name in SELECTION_OUTPUT_NAMES
                if (output / name).exists()]
    if existing:
        raise FileExistsError(
            f"selection output already exists in {output}: {', '.join(existing)}")


def save_selection_scores(scores, selected, score_ceiling, path):
    """Compare selected and rejected image scores using the same histogram bins."""
    scores = np.asarray(scores, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    if (scores.ndim != 1 or selected.shape != scores.shape
            or not scores.size or not np.all(np.isfinite(scores))):
        raise ValueError("selection scores must be finite and match the selection")
    edges = np.histogram_bin_edges(scores, bins=60)
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.hist([scores[selected], scores[~selected]], bins=edges, stacked=True,
              color=["#16883b", "#bdbdbd"], label=["Selected", "Not selected"])
    if np.isfinite(score_ceiling):
        axis.axvline(score_ceiling, color="#c62828", linestyle="--",
                     linewidth=1.0, label="Score ceiling")
    axis.set(xlabel="Crowding score (outer / central signal)", ylabel="Images")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def validate_manifest(manifest, manifest_path):
    required_arrays = ("source_rows", "image_names", "contrast_est")
    missing = [name for name in required_arrays if name not in manifest]
    if missing:
        raise ValueError(f"denoising manifest is missing fields: {missing}")
    expected_flags = {
        "ctf_corrected": True,
        "noise_denoised": True,
        "background_mean_removed": True,
        "contrast_removed": False,
        "whitened_output": False,
        "reference_used": False,
        "relion_poses_used": False,
    }
    for name, expected in expected_flags.items():
        actual = scalar(manifest, name)
        if bool(actual) != expected:
            raise ValueError(
                f"denoising manifest requires {name}={expected}, got {actual!r}")

    count = int(scalar(manifest, "n_images"))
    native_res = int(scalar(manifest, "native_res"))
    denoise_res = int(scalar(manifest, "denoise_res"))
    vox_native = float(scalar(manifest, "vox_native"))
    if count < 1 or native_res < 1 or not 1 <= denoise_res <= native_res:
        raise ValueError("manifest contains invalid image dimensions or count")
    if not np.isfinite(vox_native) or vox_native <= 0:
        raise ValueError("manifest native pixel size must be finite and positive")

    source_rows = np.asarray(manifest["source_rows"], dtype=np.int64)
    image_names = np.asarray(manifest["image_names"], dtype=str)
    contrast = np.asarray(manifest["contrast_est"], dtype=np.float64)
    if source_rows.shape != (count,) or np.any(source_rows < 0):
        raise ValueError("manifest source_rows are invalid")
    if len(np.unique(source_rows)) != count:
        raise ValueError("manifest source_rows must be unique")
    if image_names.shape != (count,):
        raise ValueError("manifest image_names are invalid")
    if contrast.shape != (count,) or not np.all(np.isfinite(contrast)):
        raise ValueError("manifest contrast estimates are invalid")

    declared_stack = optional_text(manifest, "denoised_stack")
    fallback = manifest_path.parent / "denoised_images.mrcs"
    stack_path = (resolve_manifest_path(declared_stack, manifest_path.parent)
                  if declared_stack is not None else fallback.resolve(strict=True))
    stat = stack_path.stat()
    if "stack_size" in manifest and stat.st_size != int(scalar(manifest, "stack_size")):
        raise ValueError("denoised stack size differs from its manifest")
    if ("stack_mtime_ns" in manifest
            and stat.st_mtime_ns != int(scalar(manifest, "stack_mtime_ns"))):
        raise ValueError("denoised stack modification time differs from its manifest")
    with mrcfile.mmap(stack_path, mode="r", permissive=False) as stack:
        if stack.data.shape != (count, denoise_res, denoise_res):
            raise ValueError(
                f"stack shape {stack.data.shape} differs from the manifest")
        denoise_pixel = vox_native * native_res / denoise_res
        header_pixel = float(stack.voxel_size.x)
        if not np.isclose(header_pixel, denoise_pixel, rtol=1e-5, atol=1e-5):
            raise ValueError("denoised stack pixel size differs from the manifest")
    return stack_path, count, native_res, denoise_res, vox_native


def _selection_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--denoising-dir", type=Path, required=True)
    parser.add_argument("--n-select", type=int, default=10_000)
    parser.add_argument("--n-preview", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    distance = parser.add_mutually_exclusive_group()
    distance.add_argument(
        "--neighbor-distance-fraction", type=float,
        default=DEFAULT_NEIGHBOR_FRACTION,
        help="minimum pick separation as a fraction of the native box side")
    distance.add_argument(
        "--neighbor-distance-coordinate-pixels",
        "--neighbor-distance-native-pixels",
        dest="neighbor_distance_coordinate_pixels", type=float,
        help=("explicit minimum pick separation in the units of the RELION "
              "coordinate columns, overriding the fractional default"))
    parser.add_argument("--contrast-bins", type=int, default=4)
    parser.add_argument("--max-score-quantile", type=float, default=0.95)
    parser.add_argument("--smooth-sigma", type=float, default=1.1)
    parser.add_argument("--inner-radius", type=float, default=0.50)
    parser.add_argument("--outer-radius-min", type=float, default=0.62)
    parser.add_argument("--outer-radius-max", type=float, default=0.94)
    parser.add_argument("--inner-top-fraction", type=float, default=0.20)
    parser.add_argument("--outer-top-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--outdir", "-o", "--output-dir", dest="output_dir",
                        type=Path, required=True)
    return parser


def select_main(argv=None):
    started = time.monotonic()
    parser = _selection_parser()
    args = parser.parse_args(argv)
    if args.n_select < 1 or args.n_preview < 1:
        parser.error("selection and preview counts must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not (0 < args.inner_radius < args.outer_radius_min
            < args.outer_radius_max <= 1):
        parser.error("radial masks must be ordered inside the unit disk")
    if (args.neighbor_distance_fraction <= 0 or args.contrast_bins < 1
            or not 0 < args.max_score_quantile <= 1):
        parser.error("neighbor fraction and contrast-bin count must be positive")
    if (args.neighbor_distance_coordinate_pixels is not None
            and args.neighbor_distance_coordinate_pixels <= 0):
        parser.error("explicit neighbor distance must be positive")
    if not (0 < args.inner_top_fraction <= 1
            and 0 < args.outer_top_fraction <= 1):
        parser.error("top fractions must lie in (0,1]")

    output = args.output_dir.resolve()
    denoising = args.denoising_dir.resolve(strict=True)
    if output != denoising and output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    _require_selection_outputs_absent(output)
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = (denoising / "manifest.npz").resolve(strict=True)
    with np.load(manifest_path, allow_pickle=False) as saved:
        manifest = {name: saved[name] for name in saved.files}
    stack_path, count, native_res, side, vox_native = validate_manifest(
        manifest, manifest_path)
    if args.n_select > count:
        parser.error("--n-select exceeds the stack size")
    if args.n_preview > count:
        parser.error("--n-preview exceeds the stack size")
    neighbor_distance = (
        args.neighbor_distance_coordinate_pixels
        if args.neighbor_distance_coordinate_pixels is not None
        else args.neighbor_distance_fraction * native_res)
    input_validation_done = time.monotonic()

    source_rows = np.asarray(manifest["source_rows"], dtype=np.int64)
    image_names = np.asarray(manifest["image_names"], dtype=str)
    from .preprocessing import RelionParticles

    star_path = resolve_manifest_path(
        str(scalar(manifest, "star_path")), manifest_path.parent)
    data_folder_text = optional_text(manifest, "data_folder")
    data_folder = (resolve_manifest_path(data_folder_text, manifest_path.parent)
                   if data_folder_text is not None else None)
    stack_prefix = optional_text(manifest, "stack_prefix")
    particles = RelionParticles(
        star_path, data_folder=data_folder, stack_prefix=stack_prefix,
        pixel_size=vox_native)
    if np.any(source_rows >= particles.n):
        raise ValueError("manifest source_rows contains an out-of-range STAR row")
    np.testing.assert_array_equal(particles.image_names[source_rows], image_names)
    required = ("_rlnMicrographName", "_rlnCoordinateX", "_rlnCoordinateY")
    missing = [name for name in required if name not in particles.metadata]
    if missing:
        raise ValueError(
            "low-crowding selection requires micrograph pick metadata: "
            + ", ".join(missing))
    micrograph_names = np.asarray(
        particles.metadata["_rlnMicrographName"], dtype=str)
    all_neighbor_distances = nearest_particle_distances(
        micrograph_names, particles.column("_rlnCoordinateX"),
        particles.column("_rlnCoordinateY"))
    neighbor_distances = all_neighbor_distances[source_rows]
    contrast = np.asarray(manifest["contrast_est"], dtype=np.float64)
    metadata_setup_done = time.monotonic()
    with mrcfile.mmap(stack_path, mode="r", permissive=False) as stack:
        selection = select_low_crowding_particles(
            stack.data, micrograph_names[source_rows], neighbor_distances,
            contrast, args.n_select,
            minimum_neighbor_distance=neighbor_distance,
            contrast_bins=args.contrast_bins,
            max_score_quantile=args.max_score_quantile,
            smooth_sigma=args.smooth_sigma,
            inner_radius=args.inner_radius,
            outer_radius_min=args.outer_radius_min,
            outer_radius_max=args.outer_radius_max,
            inner_top_fraction=args.inner_top_fraction,
            outer_top_fraction=args.outer_top_fraction,
            batch_size=args.batch_size,
            progress=lambda done, total: print(
                f"  scored {done:,}/{total:,}", flush=True))

    selected_positions = selection.selected_positions
    rejected_positions = selection.rejected_positions
    geometry_pass = selection.geometry_pass
    scores = selection.crowding_scores
    inner_strength = selection.inner_strength
    outer_strength = selection.outer_strength
    score_ceiling = selection.score_ceiling
    boundaries = selection.contrast_bin_boundaries
    contrast_ids = selection.contrast_bin_ids
    micrograph_ids = selection.micrograph_ids
    stratum_codes = selection.stratum_codes
    unique_strata = selection.unique_stratum_codes
    population = selection.stratum_population
    capacity = selection.stratum_geometry_capacity
    quotas = selection.stratum_selection_quota
    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[selected_positions] = True
    selection_done = time.monotonic()

    diagnostic_rng = np.random.default_rng(
        np.random.SeedSequence([args.seed, 0x53454C]))
    diagnostic_positions = diagnostic_rng.choice(
        count, args.n_preview, replace=False)
    diagnostic_selected = selected_mask[diagnostic_positions]
    with mrcfile.mmap(stack_path, mode="r", permissive=False) as stack:
        diagnostic_images = np.array(
            stack.data[diagnostic_positions], dtype=np.float32, copy=True)
    save_selection_diagnostic(
        diagnostic_images, diagnostic_selected,
        output / "selection_diagnostic", scores=scores[diagnostic_positions])
    save_selection_scores(
        scores, selected_mask, score_ceiling, output / "selection_scores")
    diagnostic_pdf = output / "selection_diagnostic.pdf"

    dataset = optional_text(manifest, "dataset")
    source_fingerprint = optional_text(manifest, "source_fingerprint")
    manifest_hash = sha256_file(manifest_path)
    np.savez_compressed(
        output / "selection.npz",
        selected_positions=selected_positions,
        selected_source_rows=source_rows[selected_positions],
        selected_image_names=image_names[selected_positions],
        rejected_positions=rejected_positions,
        geometry_pass_positions=np.flatnonzero(geometry_pass),
        diagnostic_positions=diagnostic_positions,
        diagnostic_selected=diagnostic_selected,
        crowding_scores=scores,
        inner_strength=inner_strength,
        outer_strength=outer_strength,
        nearest_neighbor_distance_coordinate_pixels=neighbor_distances,
        contrast_bin_boundaries=boundaries,
        contrast_bin_ids=contrast_ids,
        stratum_codes=stratum_codes,
        unique_stratum_codes=unique_strata,
        stratum_population=population,
        stratum_geometry_capacity=capacity,
        stratum_selection_quota=quotas,
        effective_contrast_bins=len(np.unique(contrast_ids)),
        dataset="" if dataset is None else dataset,
        source_fingerprint="" if source_fingerprint is None else source_fingerprint,
        input_manifest=str(manifest_path),
        input_manifest_sha256=manifest_hash,
        input_stack=str(stack_path),
        native_res=native_res,
        native_pixel_size_angstrom=vox_native,
        minimum_neighbor_distance_fraction=neighbor_distance / native_res,
        minimum_neighbor_distance_coordinate_pixels=neighbor_distance,
    )
    with (output / "selected_particles.tsv").open("w") as stream:
        stream.write(
            "denoised_position\tstar_row_zero_based\timage_name\t"
            "crowding_score\tnearest_neighbor_coordinate_pixels\n")
        for position in selected_positions:
            stream.write(
                f"{position}\t{source_rows[position]}\t{image_names[position]}\t"
                f"{scores[position]:.9g}\t{neighbor_distances[position]:.9g}\n")

    selection_path = output / "selection.npz"
    posthoc_viewing_tv = viewing_tv(particles, source_rows, selected_mask)
    before_report_write = time.monotonic()
    timings = {
        "input_and_manifest_validation": input_validation_done - started,
        "metadata_and_neighbor_setup": (
            metadata_setup_done - input_validation_done),
        "scoring_and_selection": selection_done - metadata_setup_done,
        "diagnostic_and_output_writing": before_report_write - selection_done,
    }
    report = {
        "dataset": dataset,
        "input_stack": str(stack_path),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": manifest_hash,
        "selection_sha256": sha256_file(selection_path),
        "source_fingerprint": source_fingerprint,
        "source_star": str(star_path),
        "source_data_folder": None if data_folder is None else str(data_folder),
        "source_stack_prefix": stack_prefix,
        "n_input": count,
        "n_selected": args.n_select,
        "selected_fraction": args.n_select / count,
        "native_box_pixels": native_res,
        "native_pixel_size_angstrom": vox_native,
        "native_box_size_angstrom": native_res * vox_native,
        "selection_rule": (
            "nearest picked neighbor at least the configured distance; capped "
            "proportional micrograph-by-contrast-bin quotas; lowest image "
            "crowding score within each stratum"),
        "score_definition": (
            "top-RMS positive smoothed signal in the outer annulus divided "
            "by top-RMS positive smoothed signal in the central disk"),
        "smooth_sigma_pixels_at_denoise_resolution": args.smooth_sigma,
        "minimum_neighbor_distance_fraction_of_native_box": (
            neighbor_distance / native_res),
        "minimum_neighbor_distance_coordinate_pixels": neighbor_distance,
        "coordinate_units": (
            "numeric units stored in RELION _rlnCoordinateX/_rlnCoordinateY; "
            "no Angstrom conversion is assumed"),
        "maximum_crowding_score_quantile": args.max_score_quantile,
        "maximum_crowding_score": score_ceiling,
        "n_passing_geometry_and_score_gates": int(geometry_pass.sum()),
        "n_micrographs": int(len(np.unique(micrograph_ids))),
        "n_populated_micrograph_contrast_strata": int(len(unique_strata)),
        "contrast_bins": args.contrast_bins,
        "effective_contrast_bins": int(len(np.unique(contrast_ids))),
        "contrast_bin_boundaries": boundaries.tolist(),
        "inner_radius_fraction": args.inner_radius,
        "outer_radius_fraction": [args.outer_radius_min, args.outer_radius_max],
        "inner_top_fraction": args.inner_top_fraction,
        "outer_top_fraction": args.outer_top_fraction,
        "selected_score_quantiles": {
            str(q): float(np.quantile(scores[selected_positions], q))
            for q in (0, 0.1, 0.5, 0.9, 0.99, 1)},
        "score_quantiles": {
            str(q): float(np.quantile(scores, q))
            for q in (0, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 1)},
        "preview_seed": args.seed,
        "n_preview": args.n_preview,
        "selection_diagnostic": {
            "sampling": "uniform_without_replacement_from_all_denoised_images",
            "seed_sequence": [args.seed, 0x53454C],
            "n_images": int(len(diagnostic_positions)),
            "n_selected": int(np.sum(diagnostic_selected)),
            "n_not_selected": int(np.sum(~diagnostic_selected)),
            "pdf": str(diagnostic_pdf),
            "pdf_sha256": sha256_file(diagnostic_pdf),
            "annotation": "crowding score; green check selected, red cross not selected",
        },
        "score_plot": {
            "pdf": str(output / "selection_scores.pdf"),
        },
        "uses_reference_map": False,
        "uses_relion_poses_for_selection": False,
        "posthoc_relion_viewing_tv_6x12": posthoc_viewing_tv,
        "uses_reconstruction": False,
        "elapsed_seconds": before_report_write - started,
        "timings_seconds": timings,
        "provenance": {
            "selector": str(Path(__file__).resolve()),
            "selector_sha256": sha256_file(Path(__file__).resolve()),
            "selection_module": str(PARTICLE_SELECTION_SOURCE.resolve()),
            "selection_module_sha256": sha256_file(PARTICLE_SELECTION_SOURCE),
            "kam_cryoem_commit": git_commit(ROOT),
            "python": sys.version,
            "numpy": np.__version__,
        },
    }
    report_path = output / "selection_report.json"
    # Write once so the measured output stage includes a real JSON
    # serialization and filesystem write, then patch in the final timings.
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report_checkpoint = time.monotonic()
    timings["diagnostic_and_output_writing"] = (
        report_checkpoint - selection_done)
    report["elapsed_seconds"] = report_checkpoint - started
    timing_values = [report["elapsed_seconds"], *timings.values()]
    if (not np.all(np.isfinite(timing_values))
            or any(value < 0.0 for value in timing_values)):
        raise RuntimeError("selection timing measurements are not finite")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Selection timing: "
        f"input={timings['input_and_manifest_validation']:.1f}s, "
        f"metadata={timings['metadata_and_neighbor_setup']:.1f}s, "
        f"scoring={timings['scoring_and_selection']:.1f}s, "
        f"outputs={timings['diagnostic_and_output_writing']:.1f}s, "
        f"total={report['elapsed_seconds']:.1f}s",
        flush=True)
    print(json.dumps(report, indent=2), flush=True)



# ===========================================================================
# Standalone map alignment and FSC
# ===========================================================================

def _align_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Align an existing reconstruction to an external "
                     "reference and compute its FSC. A coarse reconstruction "
                     "is Fourier-upsampled at fixed physical field of view to "
                     "the unchanged native reference grid."),
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--map", dest="map_path", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument(
        "--outdir", "-o", "--output-dir", dest="output_dir", type=Path,
        required=True)
    parser.add_argument(
        "--voxel-size", "--pixel-size", dest="pixel_size", type=float,
        help="reconstruction voxel size in Angstrom; defaults to its MRC header",
    )
    parser.add_argument(
        "--reference-voxel-size", "--reference-pixel-size",
        dest="reference_pixel_size", type=float,
        help="reference voxel size in Angstrom; defaults to its MRC header",
    )
    parser.add_argument(
        "--search-lowpass", "--lowpass", dest="search_lowpass", type=float,
        default=postprocessing.DEFAULT_ALIGNMENT_LOWPASS,
        help=("low-pass resolution used for the global orientation search; "
              "a low/intermediate-frequency refinement then estimates scale "
              "and translation; the evaluation map receives no low-pass or "
              "support mask"),
    )
    parser.add_argument(
        "--allow-sign", action=argparse.BooleanOptionalAction, default=False,
        help="also test the globally sign-flipped reconstruction",
    )
    parser.add_argument("--label", default="reconstruction")
    return parser


def align_main(argv: list[str] | None = None) -> int:
    args = _align_parser().parse_args(argv)
    if not np.isfinite(args.search_lowpass) or args.search_lowpass <= 0:
        raise ValueError("--search-lowpass must be finite and positive")
    for value, name in ((args.pixel_size, "--voxel-size"),
                        (args.reference_pixel_size, "--reference-voxel-size")):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    report = postprocessing.align_map_to_reference(
        args.map_path,
        args.reference_map,
        args.output_dir,
        args.search_lowpass,
        allow_sign=args.allow_sign,
        label=args.label,
        pixel_size=args.pixel_size,
        reference_pixel_size=args.reference_pixel_size,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


def _latest_completed_relion_job(relion_dir: Path) -> tuple[Path, Path, Path]:
    """Return the latest job's original MoM initializer and final map."""
    completed = []
    for job in relion_dir.glob("job_*"):
        match = re.fullmatch(r"job_(\d+)", job.name)
        if match is None or not job.is_dir():
            continue
        # initial_model.mrc is the Fourier-upsampled MoM map supplied to RELION.
        # Prefer the original coarse MoM source so alignment can preserve its
        # bandlimit explicitly. Fall back to the upsampled copy only when the
        # original source is no longer accessible.
        initial = job / "inputs/initial_model.mrc"
        provenance_path = job / "inputs/provenance.json"
        try:
            provenance = _read_json(provenance_path)
            source = provenance.get("source", {})
            source_map = Path(source.get("map", "")).expanduser()
            if source_map and not source_map.is_absolute():
                source_map = provenance_path.parent / source_map
            expected_hash = source.get("map_sha256")
            if (source_map.is_file() and isinstance(expected_hash, str)
                    and _sha256(source_map) == expected_hash):
                initial = source_map
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        final = job / "run_class001.mrc"
        if initial.is_file() and final.is_file():
            completed.append((int(match.group(1)), job, initial, final))
    if not completed:
        raise FileNotFoundError(
            f"no completed job in {relion_dir}: expected both "
            "inputs/initial_model.mrc and run_class001.mrc")
    _, job, initial, final = max(completed, key=lambda item: item[0])
    return job.resolve(), initial.resolve(strict=True), final.resolve(strict=True)


def _aligned_viewing_inputs(job, refined_map, mom_fit=None, relion_star=None):
    """Find density inputs tied to the selected refinement's initializer."""
    inputs = {"mom_fit": None, "relion_star": None}
    reasons = []
    for name, override in (("mom_fit", mom_fit), ("relion_star", relion_star)):
        if override is not None:
            path = Path(override).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"--{name.replace('_', '-')} must be a file: {path}")
            inputs[name] = path

    provenance_path = job / "inputs/provenance.json"
    if inputs["mom_fit"] is None:
        try:
            provenance = _read_json(provenance_path)
            source = provenance.get("source", {})
            expected_hash = source.get("map_sha256")
            if not isinstance(expected_hash, str) or not expected_hash:
                reasons.append("initializer provenance lacks the source-map checksum")
            else:
                candidates = []
                if source.get("map"):
                    source_map = Path(source["map"]).expanduser()
                    if not source_map.is_absolute():
                        source_map = provenance_path.parent / source_map
                    candidates.append(source_map)
                pipeline_dir = provenance.get("preparation_request", {}).get(
                    "pipeline_dir")
                if pipeline_dir:
                    pipeline = Path(pipeline_dir).expanduser()
                    if not pipeline.is_absolute():
                        pipeline = provenance_path.parent / pipeline
                    candidates.append(pipeline / "MoM/M3/recon.mrc")
                for source_map in candidates:
                    fit = source_map.with_name("fit.npz")
                    if (source_map.is_file() and fit.is_file()
                            and _sha256(source_map) == expected_hash):
                        inputs["mom_fit"] = fit.resolve(strict=True)
                        inputs["initializer_source_map"] = str(
                            source_map.resolve(strict=True))
                        inputs["initializer_source_map_sha256"] = expected_hash
                        break
                if inputs["mom_fit"] is None:
                    reasons.append(
                        "no MoM fit accompanies a source map matching the "
                        "initializer provenance checksum")
        except (OSError, ValueError, TypeError, AttributeError) as error:
            reasons.append(f"cannot read initializer provenance: {error}")

    if inputs["relion_star"] is None:
        suffix = "_class001.mrc"
        name = refined_map.name
        if name.endswith(suffix):
            candidate = refined_map.with_name(name[:-len(suffix)] + "_data.star")
            if candidate.is_file():
                inputs["relion_star"] = candidate.resolve(strict=True)
        if inputs["relion_star"] is None:
            reasons.append("final RELION particle STAR is missing beside the refined map")
    inputs["provenance"] = str(provenance_path)
    inputs["overrides"] = {
        "mom_fit": mom_fit is not None, "relion_star": relion_star is not None,
    }
    inputs["skip_reason"] = "; ".join(reasons) if reasons else None
    return inputs


def _align_reference_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kam align_reference",
        description=("Align the initial and refined maps from the latest "
                     "completed RELION job, compute FSCs, and "
                     "compare viewing distributions in reference coordinates "
                     "when the MoM fit and final particle poses are available. "
                     "The initial comparison uses the original coarse MoM map "
                     "when provenance permits, otherwise the unmasked, "
                     "non-low-pass Fourier-upsampled initializer."),
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--relion-dir", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument(
        "--mom-fit", type=Path,
        help="override the MoM fit.npz inferred from initializer provenance for viewing-density comparison",
    )
    parser.add_argument(
        "--relion-star", type=Path,
        help="override the final particle STAR used for the aligned viewing-density comparison",
    )
    parser.add_argument(
        "--outdir", "-o", "--output-dir", dest="output_dir", type=Path,
        help=("comparison output directory; defaults to the dataset's "
              "align_reference directory"),
    )
    parser.add_argument(
        "--voxel-size", "--pixel-size", dest="pixel_size", type=float,
        help=("voxel size in Angstrom for both initial and refined maps; "
              "defaults to each map's MRC header"),
    )
    parser.add_argument(
        "--reference-voxel-size", "--reference-pixel-size",
        dest="reference_pixel_size", type=float,
        help="reference voxel size in Angstrom; defaults to its MRC header",
    )
    parser.add_argument(
        "--search-lowpass", "--lowpass", dest="search_lowpass", type=float,
        default=postprocessing.DEFAULT_ALIGNMENT_LOWPASS,
        help=("low-pass resolution used for the initial map's global "
              "orientation search; a low/intermediate-frequency refinement "
              "then estimates scale and translation; the evaluation maps "
              "receive no low-pass or support mask"),
    )
    parser.add_argument(
        "--refined-search-lowpass", dest="refined_search_lowpass",
        type=float, nargs="+", default=(80.0, 60.0, 50.0, 40.0, 30.0),
        help=("candidate search low-pass resolutions for the refined map; "
              "each candidate is aligned independently and the one with the "
              "highest full-band correlation is kept. A single low-resolution "
              "search can land in a wrong, pseudo-degenerate rotation basin "
              "(near-cylindrical or blob-like shapes score alike at 80 A) "
              "that the later scale/translation refinement cannot escape"),
    )
    parser.add_argument(
        "--allow-sign", action=argparse.BooleanOptionalAction, default=False,
        help="also test globally sign-flipped maps",
    )
    return parser


def align_reference_main(argv: list[str] | None = None) -> int:
    """Align maps, compute FSCs, and optionally compare aligned viewing densities."""
    started = time.monotonic()
    args = _align_reference_parser().parse_args(argv)
    if not np.isfinite(args.search_lowpass) or args.search_lowpass <= 0:
        raise ValueError("--search-lowpass must be finite and positive")
    refined_lowpasses = tuple(dict.fromkeys(
        float(value) for value in args.refined_search_lowpass))
    if not refined_lowpasses or any(
            not np.isfinite(value) or value <= 0
            for value in refined_lowpasses):
        raise ValueError(
            "--refined-search-lowpass values must be finite and positive")
    for value, name in ((args.pixel_size, "--voxel-size"),
                        (args.reference_pixel_size, "--reference-voxel-size")):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    relion_dir = args.relion_dir.expanduser().resolve(strict=True)
    if not relion_dir.is_dir():
        raise NotADirectoryError(relion_dir)
    reference_map = args.reference_map.expanduser().resolve(strict=True)
    job, initial_map, refined_map = _latest_completed_relion_job(relion_dir)
    viewing_inputs = _aligned_viewing_inputs(
        job, refined_map, args.mom_fit, args.relion_star)
    if args.output_dir is not None:
        output = args.output_dir.expanduser().resolve()
    elif relion_dir.name in {"relion_refine", "Relion"}:
        output = relion_dir.parent / "align_reference"
    else:
        legacy = re.fullmatch(r"Relion_(\d+)", relion_dir.name)
        output = (
            relion_dir.parent / legacy.group(1) / "align_reference"
            if legacy is not None else relion_dir.parent / "align_reference"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Reference comparison directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    print(f"Aligning RELION initial map: {initial_map}", flush=True)
    initial_started = time.monotonic()
    initial = postprocessing.align_map_to_reference(
        initial_map, reference_map, output / "initial", args.search_lowpass,
        allow_sign=bool(args.allow_sign), label="initial",
        pixel_size=args.pixel_size,
        reference_pixel_size=args.reference_pixel_size,
    )
    initial_seconds = time.monotonic() - initial_started
    print(f"Aligning RELION refined map: {refined_map}", flush=True)
    refined_started = time.monotonic()
    search_root = output / "refined_search"
    refined_candidates = []
    for lowpass in refined_lowpasses:
        print(f"  refined candidate: search lowpass {lowpass:g} A", flush=True)
        candidate = postprocessing.align_map_to_reference(
            refined_map, reference_map, search_root / f"lp{lowpass:g}",
            lowpass, allow_sign=bool(args.allow_sign), label="refined",
            pixel_size=args.pixel_size,
            reference_pixel_size=args.reference_pixel_size,
        )
        evaluation = candidate["evaluation"]
        refined_candidates.append({
            "search_lowpass_angstrom": lowpass,
            "correlation": float(evaluation["correlation"]),
            "fsc_05_angstrom": evaluation["fsc_05_angstrom"],
            "reflected": bool(evaluation["reflected"]),
        })
        print(f"    correlation={evaluation['correlation']:.4f} "
              f"fsc05={evaluation['fsc_05_angstrom']}", flush=True)
    best_refined = max(refined_candidates, key=lambda item: item["correlation"])
    refined_lowpass = best_refined["search_lowpass_angstrom"]
    shutil.rmtree(search_root)
    print(f"  selected refined search lowpass {refined_lowpass:g} A "
          "by full-band correlation", flush=True)
    refined = postprocessing.align_map_to_reference(
        refined_map, reference_map, output / "refined", refined_lowpass,
        allow_sign=bool(args.allow_sign), label="refined",
        pixel_size=args.pixel_size,
        reference_pixel_size=args.reference_pixel_size,
    )
    refined_seconds = time.monotonic() - refined_started
    with np.load(output / "initial" / "fsc_full.npz") as saved:
        frequency = np.asarray(saved["spatial_frequency"], dtype=float)
        initial_curve = np.asarray(saved["fsc"], dtype=float)
    with np.load(output / "refined" / "fsc_full.npz") as saved:
        refined_frequency = np.asarray(saved["spatial_frequency"], dtype=float)
        refined_curve = np.asarray(saved["fsc"], dtype=float)
    common_frequency_grid = (
        refined_frequency.shape == frequency.shape
        and np.allclose(refined_frequency, frequency, rtol=1e-12, atol=0.0))

    def curve_label(name, evaluation):
        resolution = evaluation["evaluation"]["fsc_05_angstrom"]
        return (f"{name} ({resolution:.1f} Å)" if resolution is not None
                else f"{name} (no 0.5 crossing)")

    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    axis.plot(
        frequency, initial_curve, label=curve_label("initial", initial))
    axis.plot(
        refined_frequency, refined_curve, label=curve_label("refined", refined))
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Spatial frequency (Å⁻¹)", ylabel="FSC",
        ylim=(-1.05, 1.05),
    )
    axis.legend(frameon=False, loc="upper right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "fsc_initial_refined.pdf")
    plt.close(figure)
    viewing_started = time.monotonic()
    if viewing_inputs["skip_reason"] is not None:
        viewing = {"status": "skipped", "reason": viewing_inputs["skip_reason"]}
        print(f"Skipping aligned viewing-density comparison: {viewing['reason']}",
              flush=True)
    else:
        print("Comparing MoM and RELION viewing distributions in reference coordinates",
              flush=True)
        viewing = postprocessing.plot_aligned_viewing_distributions(
            viewing_inputs["mom_fit"], viewing_inputs["relion_star"],
            initial, refined, output)
    viewing["input_discovery"] = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in viewing_inputs.items()
    }
    viewing_seconds = time.monotonic() - viewing_started
    postprocess_report = relion_dir.parent / "postprocess" / \
        "postprocess_report.json"
    postprocess = (_read_json(postprocess_report)
                   if postprocess_report.is_file() else None)
    if postprocess is not None:
        print("masked postprocess resolution: "
              f"{postprocess['masked_fsc_resolution_angstrom']:.2f} A",
              flush=True)
    report = {
        "complete": True,
        "relion_dir": str(relion_dir),
        "selected_job": str(job),
        "postprocess": postprocess,
        "reference_map": str(reference_map),
        "voxel_size_override_angstrom": args.pixel_size,
        "reference_voxel_size_override_angstrom": args.reference_pixel_size,
        "alignment_search_lowpass_angstrom": float(args.search_lowpass),
        "refined_search_lowpass_candidates": refined_candidates,
        "refined_search_lowpass_selected_angstrom": float(refined_lowpass),
        "full_frequency_comparison": bool(
            initial["evaluation"]["full_frequency_comparison"]
            and refined["evaluation"]["full_frequency_comparison"]),
        "allow_sign": bool(args.allow_sign),
        "initial": initial,
        "refined": refined,
        "viewing_distributions": viewing,
        "combined_fsc": {
            "common_frequency_grid": bool(common_frequency_grid),
            "pdf": str((output / "fsc_initial_refined.pdf").resolve()),
            "title": None,
        },
        "elapsed_seconds": time.monotonic() - started,
        "timings_seconds": {
            "initial_alignment": initial_seconds,
            "refined_alignment": refined_seconds,
            "viewing_distributions": viewing_seconds,
        },
    }
    _write_json_atomic(output / "alignment_reference.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


# ===========================================================================
# Prepared-moments M2+M3 reconstruction (was reconstruct_prepared.py)
# ===========================================================================

REPORT_NAME = "reconstruction_report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_nonnegative(value: object, context: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < 0.0):
        raise ValueError(f"{context} must be finite and nonnegative")
    return float(value)


def _stage_report(result, wall_seconds: float) -> dict:
    summary_path = result.summary_path.resolve(strict=True)
    summary = _read_json(summary_path)
    if not summary.get("complete") or not summary.get("fit_complete"):
        raise RuntimeError(f"incomplete reconstruction summary: {summary_path}")
    child_elapsed = _finite_nonnegative(
        summary["elapsed_seconds"], f"{summary_path}: elapsed_seconds")
    child_timings = summary.get("timings_seconds")
    if not isinstance(child_timings, dict):
        raise ValueError(f"{summary_path} lacks timings_seconds")
    for name, value in child_timings.items():
        _finite_nonnegative(value, f"{summary_path}: timings_seconds.{name}")
    return {
        "order": int(result.order),
        "best_init": int(result.best_init),
        "objective": float(result.objective),
        "relative_moment_error": float(result.relative_moment_error),
        "summary_path": str(summary_path),
        "timing_path": str(summary_path),
        "timing_json_key": "timings_seconds",
        "fit_path": str(result.fit_path.resolve(strict=True)),
        "reconstruction_path": str(
            result.reconstruction_path.resolve(strict=True)),
        "wall_seconds_this_invocation": _finite_nonnegative(
            wall_seconds, "stage wall time"),
        "child_elapsed_seconds_this_invocation": child_elapsed,
        "child_timings_seconds": child_timings,
    }


def _reconstruct_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--outdir", "-o", "--output-dir", dest="output_dir",
                        type=Path, required=True)
    parser.add_argument("--vol-degree", dest="ell_max", metavar="VOL_DEGREE",
                        type=int, default=postprocessing.DEFAULT_ELL_MAX)
    parser.add_argument(
        "--viewing-degree", type=int, choices=(0, 2, 4),
        default=postprocessing.DEFAULT_VIEWING_DEGREE,
    )
    parser.add_argument(
        "--sphere-nodes", type=int, default=postprocessing.DEFAULT_SPHERE_NODES)
    parser.add_argument(
        "--sphere-degree", default="auto",
        help=("select the smallest positive-weight quadrature of this degree; "
              "'auto' (the default) uses 3*vol_degree + viewing_degree, the "
              "exactness requirement of the M3 model, and 'none' selects by "
              "--sphere-nodes instead"))
    parser.add_argument("--n-init", type=int, default=postprocessing.DEFAULT_N_INIT)
    parser.add_argument("--maxiter", type=int, default=postprocessing.DEFAULT_MAXITER)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--resume", action="store_true",
        help="resume compatible partial outputs; otherwise output must be empty",
    )
    return parser


def reconstruct_prepared_main(argv: list[str] | None = None) -> int:
    parser = _reconstruct_parser()
    args = parser.parse_args(argv)
    _resolve_sphere_degree(parser, args)
    prepared_dir = args.prepared_dir.expanduser().resolve(strict=True)
    if not (prepared_dir / "moments.npz").is_file():
        raise FileNotFoundError(f"missing {prepared_dir / 'moments.npz'}")
    output_dir = args.output_dir.expanduser().resolve()
    nonempty = output_dir.exists() and any(output_dir.iterdir())
    if nonempty and not args.resume:
        raise FileExistsError(
            f"{output_dir} is nonempty; choose a new directory or use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    m1_dir = output_dir / "M1"
    m2_dir = output_dir / "M2"
    m3_dir = output_dir / "M3"

    common = dict(
        prepared_dir=prepared_dir,
        ell_max=args.ell_max,
        viewing_degree=args.viewing_degree,
        sphere_nodes=args.sphere_nodes,
        sphere_degree=args.sphere_degree,
        n_init=args.n_init,
        maxiter=args.maxiter,
        device=args.device,
        print_every=args.print_every,
        seed=args.seed,
        resume=args.resume,
    )
    # Sequential matching as in the paper: M1 first, each stage
    # warm-starting the next.
    m1_config = ReconstructionConfig(
        output_dir=m1_dir, order=1, initial_fit=None, **common)
    m2_config = ReconstructionConfig(
        output_dir=m2_dir, order=2, initial_fit=m1_dir, **common)
    m3_config = ReconstructionConfig(
        output_dir=m3_dir, order=3, initial_fit=m2_dir, **common)

    started_utc = _utc_now()
    started = time.monotonic()
    m1_started = time.monotonic()
    m1_result = reconstruct_single_moment(m1_config)
    m1_wall = time.monotonic() - m1_started
    m2_started = time.monotonic()
    m2_result = reconstruct_single_moment(m2_config)
    m2_wall = time.monotonic() - m2_started
    m3_started = time.monotonic()
    m3_result = reconstruct_single_moment(m3_config)
    m3_wall = time.monotonic() - m3_started
    total = time.monotonic() - started

    stages = {
        "m1": _stage_report(m1_result, m1_wall),
        "m2": _stage_report(m2_result, m2_wall),
        "m3": _stage_report(m3_result, m3_wall),
    }
    child_wall_sum = m1_wall + m2_wall + m3_wall
    report = {
        "schema_version": 1,
        "workflow": "prepared_moments_m2_then_m3",
        "complete": True,
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "configuration": {
            "prepared_dir": str(prepared_dir),
            "output_dir": str(output_dir),
            "ell_max": args.ell_max,
            "viewing_degree": args.viewing_degree,
            "sphere_nodes": args.sphere_nodes,
            "sphere_degree": args.sphere_degree,
            "n_inplane": 1,
            "n_init": args.n_init,
            "maxiter": args.maxiter,
            "device": args.device,
            "seed": args.seed,
            "print_every": args.print_every,
            "resume": args.resume,
            "m3_initialization": (
                "M1 warm-starts M2; M2 init_i warm-starts M3 init_i"),
        },
        "stages": stages,
        "timings_seconds": {
            "m1_wall_this_invocation": m1_wall,
            "m2_wall_this_invocation": m2_wall,
            "m3_wall_this_invocation": m3_wall,
            "child_stage_wall_sum": child_wall_sum,
            "orchestration_overhead": max(0.0, total - child_wall_sum),
            "total_this_invocation": total,
        },
    }
    _write_json_atomic(output_dir / REPORT_NAME, report)
    print(f"Wrote {output_dir / REPORT_NAME}", flush=True)
    return 0



def relion_prepare_main(argv=None):
    """Delegate to the RELION preparation module."""
    from . import relion_prepare

    return relion_prepare.relion_prepare_main(argv)


def postprocess_main(argv=None):
    """Delegate to the RELION post-processing step."""
    from . import relion_prepare

    return relion_prepare.postprocess_main(argv)


COMMANDS = {
    "pipeline": pipeline_main,
    "align_reference": align_reference_main,
    "denoise": denoise_main,
    "select": select_main,
    "align": align_main,
    "align3d": align_reference_main,
    "reconstruct-prepared": reconstruct_prepared_main,
    "relion_prepare": relion_prepare_main,
    "postprocess": postprocess_main,
}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    result = COMMANDS[argv[0]](argv[1:])
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
