#!/usr/bin/env python3
"""The ``MoM`` command.

    MoM pipeline particles.star --outdir results ...    denoise, select and reconstruct from a RELION particle STAR file
    MoM reconstruct denoised.mrcs --outdir out ...      reconstruct from a stack of denoised images
    MoM fit moments.npz --outdir out ...                reconstruct from precomputed moments (examples/moments)

A pipeline run writes to ``<results>/<EMPIAR number>_<Slurm job id>/kam_pipeline`` (a local identifier outside Slurm), with its settings in
``hyperparameters.json`` and its log in ``run.log``; a resumed run reuses the stages that finished with the same settings. ``MoM denoise``
and ``MoM select`` are the stages the pipeline runs as separate processes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import mrcfile
import numpy as np

from .particles import (DEFAULT_DENOISE_MAX_ORDER, DEFAULT_DENOISE_RES, DEFAULT_N_SELECT, DEFAULT_NEIGHBOR_FRACTION,
                        SELECTION_OUTPUT_NAMES, FastCWFDenoiseConfig, denoise_relion_particles, select_particles)
from .moments import (DEFAULT_MOMENT_RES, MomentPreparationConfig, prepare_moments, prepare_moments_from_file,
                      prepare_moments_from_images)
from .optimization import (DEFAULT_ELL_MAX, DEFAULT_VIEWING_DEGREE, M3_MAXITER, MAXITER, N_INIT, ReconstructionConfig,
                           reconstruct_order, select_m2_start)
from .shift import DEFAULT_SHIFT_STEP_PX
from .utils import ROOT, capture_run_log, git_commit, use_device

# ===========================================================================
# MoM pipeline orchestration
# ===========================================================================

LOCAL_RUN_ID = (
    "local_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    + f"_{os.getpid()}"
)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write_json_exclusive(path: Path, value: dict) -> None:
    """Create a JSON file without replacing a concurrent pipeline manifest."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _write_json_atomic(path: Path, value: dict) -> None:
    """Create or replace a JSON report in one step (written next to it, then renamed), so an interrupted run never leaves half a report."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _optional_path(value: Path | None) -> str | None:
    return None if value is None else str(value.expanduser().resolve())


def _dataset_label(args: argparse.Namespace) -> str:
    return args.star_path.stem if args.dataset_label is None else args.dataset_label


def _dataset_number(args: argparse.Namespace) -> str:
    """The name of the run folder: the EMPIAR number of an ``EMPIAR-<number>`` label, otherwise the label."""
    label = _dataset_label(args)
    match = re.fullmatch(r"EMPIAR-(\d+)", label, re.IGNORECASE)
    return match.group(1) if match else re.sub(r"[^A-Za-z0-9_.-]", "_", label)


def _execution_id() -> str:
    """Return the Slurm job ID, or a collision-resistant local-run ID."""
    return os.environ.get("SLURM_JOB_ID", LOCAL_RUN_ID)


def _configuration(args: argparse.Namespace) -> dict:
    """Every setting that determines the output of a stage.

    It is recorded once in pipeline_config.json and in each stage that finishes; a finished stage is reused only when
    its recorded configuration equals the current one."""
    return {
        "dataset_label": _dataset_label(args),
        "inputs": {
            "star_path": str(args.star_path.expanduser().resolve()),
            "data_folder": _optional_path(args.data_folder),
            "stack_prefix": args.stack_prefix,
            "native_pixel_size": args.native_pixel_size,
        },
        "denoising": {
            "resolution": args.denoise_res,
            "fle_max_order": args.denoise_max_order,
            "batch_size": args.batch_size,
            "device": args.device,
            "ctf_mean_defocus_bin_angstrom": args.ctf_mean_defocus_bin,
            "background_radius": args.background_radius,
            "flip_sign": args.flip_sign,
            "whiten_per_micrograph": args.whiten_per_micrograph,
        },
        "selection": {
            "n_images": args.n_select,
            "neighbor_distance_fraction": args.neighbor_distance_fraction,
            "preview_images": args.n_preview,
        },
        "moments": {
            "resolution": args.moment_res,
        },
        "model": {
            "ell_max": args.ell_max,
            "viewing_degree": args.viewing_degree,
            "symmetry": args.symmetry,
        },
        "optimization": {
            "n_initializations": N_INIT,
            "maxiter": MAXITER,
            "m3_maxiter": M3_MAXITER,
            "seed": args.seed,
            "shift_distribution": _shift_settings(args, args.moment_res),
        },
    }


def _denoise_command(args, paths):
    command = [
        sys.executable, "-u", "-m", "kam.command_line", "denoise",
        str(args.star_path.expanduser().resolve()),
        "--dataset-label", _dataset_label(args),
        "--denoise-res", str(args.denoise_res),
        "--batch-size", str(args.batch_size), "--workers", str(args.workers),
        "--device", args.device,
        "--background-radius", str(args.background_radius),
        "--denoise-max-order", str(args.denoise_max_order),
        "--ctf-mean-defocus-bin", str(args.ctf_mean_defocus_bin),
        "--flip-sign" if args.flip_sign else "--no-flip-sign",
        "--whiten-per-micrograph" if args.whiten_per_micrograph else "--no-whiten-per-micrograph",
        "--outdir", str(paths["denoising"]),
    ]
    if args.micrograph_noise_psd is not None:
        command.extend(["--micrograph-noise-psd", str(args.micrograph_noise_psd)])
    if args.data_folder is not None:
        command.extend(["--datadir", str(args.data_folder.expanduser().resolve())])
    if args.stack_prefix is not None:
        command.extend(["--strip-prefix", args.stack_prefix])
    if args.native_pixel_size is not None:
        command.extend(["--native-pixel-size", str(args.native_pixel_size)])
    return command


def _selection_command(args, paths, auto=True):
    """The select command: the first pass takes the tightest score ceiling that leaves ``n_select`` particles, the second keeps them all."""
    command = [
        sys.executable, "-u", "-m", "kam.command_line", "select",
        "--denoising-dir", str(paths["denoising"]),
        "--n-select", str(args.n_select), "--n-preview", str(args.n_preview),
        "--seed", str(args.seed), "--batch-size", str(args.batch_size),
        *(["--auto-score-quantile"] if auto else ["--max-score-quantile", "1.0", "--no-auto-score-quantile"]),
        "--outdir", str(paths["selection"]),
        "--neighbor-distance-fraction", str(args.neighbor_distance_fraction),
    ]
    return command


def _report_value(value):
    """The resolved settings as JSON values (paths as absolute strings)."""
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {str(key): _report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(item) for item in value]
    return value


def _resolved_stage_settings(args, paths, auto=True):
    """The settings the denoise and select stages run with, the defaults of their own parsers included."""
    command = _denoise_command(args, paths)
    denoise = FastCWFDenoiseConfig(**vars(_denoise_parse_args(command[command.index("denoise") + 1:])))
    command = _selection_command(args, paths, auto)
    selection = _selection_parser().parse_args(command[command.index("select") + 1:])
    return asdict(denoise), vars(selection)


def _hyperparameters(args: argparse.Namespace, paths: dict[str, Path]) -> dict:
    """Record CLI values and resolved stage defaults before processing data."""
    denoise, selection = _resolved_stage_settings(args, paths)
    stages = {"denoise": denoise, "selection": selection}
    stages["redenoise"], stages["reselect"] = _resolved_stage_settings(*_second_pass(args, paths), auto=False)
    final = paths["redenoised"]
    stages["moment_preparation"] = asdict(MomentPreparationConfig(
        denoising_dir=final,
        selection_file=final / "selection.npz",
        output_dir=paths["prepared"],
        n_images=args.n_select,
        denoise_res=args.denoise_res, moment_res=args.moment_res,
        batch_size=args.batch_size, device=args.device, fb_ell_max=args.ell_max,
    ))
    for order in (1, 2, 3):
        stages[f"M{order}"] = asdict(ReconstructionConfig(
            prepared_dir=paths["prepared"], output_dir=paths[f"m{order}"],
            order=order, ell_max=args.ell_max,
            viewing_degree=args.viewing_degree,
            symmetry=args.symmetry,
            n_init=1 if order == 3 else N_INIT,
            maxiter=_maxiter(order),
            device=args.device, seed=args.seed, resume=args.resume,
            initial_fit=_initial_fit(paths, order),
        ))
        if order == 2:
            stages["M2_selection"] = {"m2_dir": str(paths["m2"]), "output_dir": str(paths["m2_selected"]),
                                      "rule": "minimum M3 residual of the M2 solutions, before M3 fitting"}
    stages["shift_distribution"] = _shift_settings(args, args.moment_res)
    environment_names = (
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    return _report_value({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            {"star_path": "particles", "data_folder": "datadir",
             "results_dir": "outdir", "stack_prefix": "strip_prefix"}.get(key, key): value
            for key, value in vars(args).items()
        },
        "configuration": _configuration(args),
        "resolved_stage_configs": stages,
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
        "denoising": root / "denoise",
        "selection": root / "denoise",
        "redenoised": root / "MoM" / "denoised_selected" / "denoise",
        "prepared": root / "MoM" / "prepared",
        "m1": root / "MoM" / "M1",
        "m2": root / "MoM" / "M2",
        "m2_selected": root / "MoM" / "M2_selected",
        "m3": root / "MoM" / "M3",
    }


def _ensure_manifest(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    """Create pipeline_config.json in a fresh output directory; an existing one is left as it is."""
    manifest_path = paths["manifest"]
    paths["root"].mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        return
    if any(path != paths["log"] for path in paths["root"].iterdir()):      # run.log is opened before the manifest is written
        raise FileExistsError(
            f"output directory is nonempty but has no pipeline manifest: "
            f"{paths['root']}")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"], capture_output=True, text=True)
    _write_json_exclusive(manifest_path, {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code": {"commit": git_commit(ROOT), "uncommitted_changes": bool(dirty.stdout.strip())},
        "configuration": _configuration(args),
    })


def _run(command: list[str], *, dry_run: bool) -> None:
    print("[pipeline] " + shlex.join(command), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, check=True, cwd=ROOT, env=environment)


def _nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


# A stage is complete when the pipeline has marked its report: ``pipeline_stage`` holds the configuration the stage was made with.
_STAGE_REPORTS = {
    "denoise": ("denoising", "report.json"),
    "select": ("selection", "selection_report.json"),
    "prepare": ("prepared", "report.json"),
    "m1": ("m1", "summary.json"),
    "m2": ("m2", "summary.json"),
    "m2_select": ("m2_selected", "selection.json"),
    "m3": ("m3", "summary.json"),
}


def _stage_report_path(paths: dict[str, Path], name: str) -> Path:
    directory, filename = _STAGE_REPORTS[name]
    return paths[directory] / filename


def _stage_complete(report_path: Path, configuration: dict) -> bool:
    """True if the stage finished; a stage that finished with another configuration is an error."""
    if not report_path.is_file():
        return False
    stage = _read_json(report_path).get("pipeline_stage")
    if stage is None:
        return False
    if stage["configuration"] != configuration:
        raise ValueError(
            f"{report_path} was made with a different configuration; "
            "use the original arguments or a new output directory")
    return True


def _finish_stage(report_path: Path, configuration: dict) -> None:
    """Mark the stage complete in its report and record the configuration it was made with."""
    record = _read_json(report_path)
    record["pipeline_stage"] = {
        "configuration": configuration,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(report_path, record)


def _run_stage(name: str, report_path: Path, configuration: dict, args: argparse.Namespace, execute, *,
               existing: bool, resumable: bool = False) -> str:
    """Skip a complete stage, refuse to overwrite an incomplete one, otherwise execute and mark it complete.

    ``existing`` tells whether the stage already has output; ``resumable`` lets ``execute`` continue it."""
    if _stage_complete(report_path, configuration):
        if not args.resume:
            raise FileExistsError(f"{name} output already exists: {report_path.parent}")
        print(f"[pipeline] {name}: complete, skipping", flush=True)
        return "skipped_completed"
    if existing and not resumable:
        raise RuntimeError(
            f"{name} output is incomplete and will not be overwritten: {report_path.parent}. "
            "Inspect it, then use a new output directory or remove only that "
            "incomplete stage yourself.")
    execute()
    if not args.dry_run:
        _finish_stage(report_path, configuration)
    return "resumed_and_completed" if existing else "completed_this_invocation"


def _second_pass(args: argparse.Namespace, paths: dict[str, Path]):
    """Arguments and paths of the re-denoising pass on the selected particles only.

    Its outputs live in ``MoM/denoised_selected/denoise`` (the input of the moments), next to the STAR of
    the selected particles; the top-level ``denoise/`` keeps the first pass."""
    import copy

    folder = paths["redenoised"]
    second = copy.copy(args)
    second.star_path = folder.parent / "selected_particles.star"
    # per-micrograph whitening reuses the spectra of the first pass, which has all particles of each micrograph
    second.micrograph_noise_psd = paths["denoising"] / "manifest.npz" if args.whiten_per_micrograph else None
    if args.data_folder is None:          # relative stack paths are relative to the STAR of the first pass
        second.data_folder = args.star_path.expanduser().resolve().parent
    second_paths = dict(paths, denoising=folder, selection=folder)
    return second, second_paths


def _denoise(args: argparse.Namespace, paths: dict[str, Path], configuration: dict) -> str:
    return _run_stage(
        "denoising", _stage_report_path(paths, "denoise"), configuration, args,
        lambda: _run(_denoise_command(args, paths), dry_run=args.dry_run),
        existing=_nonempty(paths["denoising"]))


def _select(args: argparse.Namespace, paths: dict[str, Path], configuration: dict, auto: bool = True) -> str:
    return _run_stage(
        "selection", _stage_report_path(paths, "select"), configuration, args,
        lambda: _run(_selection_command(args, paths, auto), dry_run=args.dry_run),
        existing=any((paths["selection"] / name).exists() for name in SELECTION_OUTPUT_NAMES))


def _redenoise(args: argparse.Namespace, paths: dict[str, Path], configuration: dict) -> str:
    """Denoise the selected particles again: their own mean, covariance and noise estimates."""
    second, second_paths = _second_pass(args, paths)
    if not args.dry_run and not second.star_path.is_file():
        with np.load(paths["selection"] / "selection.npz", allow_pickle=False) as saved:
            rows = np.sort(np.asarray(saved["selected_source_rows"], dtype=np.int64))
        _write_selected_star(args.star_path.expanduser().resolve(), rows, second.star_path)
    return _denoise(second, second_paths, configuration)


def _reselect(args: argparse.Namespace, paths: dict[str, Path], configuration: dict) -> str:
    """Select on the re-denoised particles (all of them stay selected); their sample montage is copied to the first pass."""
    second, second_paths = _second_pass(args, paths)
    status = _select(second, second_paths, configuration, auto=False)
    if not args.dry_run:
        shutil.copy2(second_paths["selection"] / "selection_selected_sample.pdf",
                     paths["selection"] / "selection_selected_sample.pdf")
    return status


def _prepare(args: argparse.Namespace, paths: dict[str, Path], configuration: dict) -> str:
    """Form the moments from the selected particles of ``paths['denoising']``."""
    def execute():
        print("[pipeline] prepare_moments(denoising={!s}, selection={!s}, output={!s}, n_images={}, denoise_res={}, "
              "moment_res={}, fb_ell_max={})".format(
                  paths["denoising"], paths["selection"] / "selection.npz", paths["prepared"], args.n_select,
                  args.denoise_res, args.moment_res, args.ell_max), flush=True)
        if not args.dry_run:
            prepare_moments(MomentPreparationConfig(
                denoising_dir=paths["denoising"],
                selection_file=paths["selection"] / "selection.npz",
                output_dir=paths["prepared"],
                n_images=args.n_select,
                denoise_res=args.denoise_res, moment_res=args.moment_res,
                batch_size=args.batch_size, device=args.device, fb_ell_max=args.ell_max,
            ))

    return _run_stage("moment preparation", _stage_report_path(paths, "prepare"), configuration, args, execute,
                      existing=_nonempty(paths["prepared"]))


def _shift_settings(args: argparse.Namespace, side: int) -> dict:
    """Ring grid of the shift distribution (pixels of the ``side``-pixel moment grid)."""
    return dict(step_px=args.shift_step_px,
                max_px=side / 8 if args.shift_max_px is None else args.shift_max_px)


def _maxiter(order: int) -> int:
    """L-BFGS iteration limit of moment order ``order``: M3_MAXITER for M3, MAXITER for M1 and M2."""
    return M3_MAXITER if order == 3 else MAXITER


def _initial_fit(paths: dict[str, Path], order: int):
    """The warm start of moment order ``order``: none for M1, M1 for M2, the selected M2 solution for M3."""
    return {1: None, 2: paths["m1"], 3: paths["m2_selected"]}[order]


def _reconstruct(args: argparse.Namespace, paths: dict[str, Path], configuration: dict, order: int) -> str:
    output = paths[f"m{order}"]
    initial_fit = _initial_fit(paths, order)
    n_init = 1 if order == 3 else N_INIT

    def execute():
        print("[pipeline] reconstruct_order(prepared={!s}, output={!s}, order={}, L={}, viewing_degree={}, "
              "n_init={}, maxiter={}, initial_fit={!s})".format(
                  paths["prepared"], output, order, args.ell_max, args.viewing_degree,
                  n_init, _maxiter(order), initial_fit), flush=True)
        print(f"[pipeline] shift distribution: rings every {args.shift_step_px} px up to "
              f"{'side/8' if args.shift_max_px is None else args.shift_max_px} px (M1, M2, M3)", flush=True)
        if not args.dry_run:
            reconstruct_order(ReconstructionConfig(
                prepared_dir=paths["prepared"], output_dir=output, order=order,
                ell_max=args.ell_max, viewing_degree=args.viewing_degree, symmetry=args.symmetry,
                n_init=n_init, maxiter=_maxiter(order), device=args.device,
                seed=args.seed, resume=args.resume, initial_fit=initial_fit,
            ), _shift_settings(args, args.moment_res))

    return _run_stage(f"M{order} reconstruction", _stage_report_path(paths, f"m{order}"), configuration, args, execute,
                      existing=_nonempty(output), resumable=args.resume)


def _select_m2(args: argparse.Namespace, paths: dict[str, Path], configuration: dict) -> str:
    """Pick the M2 initialization whose solution best explains M3 (``optimization.select_m2_start``)."""

    def execute():
        print(f"[pipeline] select_m2_start(prepared={paths['prepared']!s}, m2={paths['m2']!s}, output={paths['m2_selected']!s})", flush=True)
        if not args.dry_run:
            select_m2_start(paths["prepared"], paths["m2"], paths["m2_selected"], ell_max=args.ell_max,
                            viewing_degree=args.viewing_degree, symmetry=args.symmetry, device=args.device)

    return _run_stage("M2 selection", _stage_report_path(paths, "m2_select"), configuration, args, execute,
                      existing=_nonempty(paths["m2_selected"]))


# The sections of ``_configuration`` that determine each stage (the stage and everything upstream of it); a changed setting only
# invalidates the stages it can affect, so M-stage settings can be varied on finished moments. The second pass depends on the selection.
_STAGE_SECTIONS = {
    "denoise": ("dataset_label", "inputs", "denoising"),
    "select": ("dataset_label", "inputs", "denoising", "selection"),
    "redenoise": ("dataset_label", "inputs", "denoising", "selection"),
    "reselect": ("dataset_label", "inputs", "denoising", "selection"),
    "prepare": ("dataset_label", "inputs", "denoising", "selection", "moments", "model"),
}


def _stages(args: argparse.Namespace, paths: dict[str, Path]) -> list:
    """(label, report, run) of every pipeline stage in execution order; ``run`` returns how the stage ended.

    Every stage is marked with the settings that determine it (``_STAGE_SECTIONS``; the M stages with all of them), so a finished
    stage is reused only by a run configured alike."""
    full = _configuration(args)

    def configuration(label):
        return {key: full[key] for key in _STAGE_SECTIONS.get(label, full)}

    stages = [("denoise", _stage_report_path(paths, "denoise"), lambda: _denoise(args, paths, configuration("denoise"))),
              ("select", _stage_report_path(paths, "select"), lambda: _select(args, paths, configuration("select")))]
    final_args, final_paths = _second_pass(args, paths)
    stages += [("redenoise", _stage_report_path(final_paths, "denoise"), lambda: _redenoise(args, paths, configuration("redenoise"))),
               ("reselect", _stage_report_path(final_paths, "select"), lambda: _reselect(args, paths, configuration("reselect")))]
    stages.append(("prepare", _stage_report_path(paths, "prepare"), lambda: _prepare(final_args, final_paths, configuration("prepare"))))
    for order in (1, 2, 3):
        if order == 3:
            stages.append(("m2_select", _stage_report_path(paths, "m2_select"),
                           lambda: _select_m2(args, paths, configuration("m2_select"))))
        stages.append((f"m{order}", _stage_report_path(paths, f"m{order}"),
                       lambda order=order: _reconstruct(args, paths, configuration(f"m{order}"), order)))
    return stages


def _write_timing_report(
        args: argparse.Namespace, paths: dict[str, Path], stages: list,
        statuses: dict[str, str], observed: dict[str, float],
        started_utc: str, finished_utc: str,
        invocation_elapsed: float) -> None:
    """Aggregate the stage reports into one root timing report; every invocation appends to its history."""
    stage_reports = {}
    for label, report_path, _ in stages:
        record = _read_json(report_path)
        stage_reports[label] = {
            "status": statuses[label],
            "report": str(report_path),
            "elapsed_seconds": float(record["elapsed_seconds"]),
            "timings_seconds": record["timings_seconds"],
            "pipeline_observed_elapsed_seconds": observed[label],
        }
    stage_elapsed = {label: stage["elapsed_seconds"] for label, stage in stage_reports.items()}
    invocation = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "full_invocation_elapsed_seconds": float(invocation_elapsed),
        "resume": bool(args.resume),
        "stage_status": dict(statuses),
        "pipeline_observed_stage_elapsed_seconds": dict(observed),
    }
    timing_path = paths["timing_report"]
    history = _read_json(timing_path)["invocations"] if timing_path.is_file() else []
    report = {
        "dataset": _dataset_label(args),
        "pipeline_output": str(paths["root"]),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "full_invocation_elapsed_seconds": float(invocation_elapsed),
        "stage_elapsed_seconds": stage_elapsed,
        "sum_stage_elapsed_seconds": float(sum(stage_elapsed.values())),
        "stages": stage_reports,
        "invocations": [*history, invocation],
    }
    _write_json_atomic(timing_path, report)
    compact = ", ".join(f"{label}={elapsed:.1f}s" for label, elapsed in stage_elapsed.items())
    print(
        f"[timing] {compact}; stage-sum={report['sum_stage_elapsed_seconds']:.1f}s; "
        f"this invocation={invocation_elapsed:.1f}s",
        flush=True,
    )
    print(f"[timing] report={timing_path}", flush=True)


def _pipeline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="MoM pipeline", allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=("Denoise the particles, select the least crowded ones and reconstruct the map from their M1, M2 and M3 "
                     "Fourier-Bessel moments. A run writes to <results>/<EMPIAR-number>_<SLURM-job-id>/kam_pipeline (a local "
                     "identifier outside Slurm); resolved settings go to hyperparameters.json and live progress to run.log."))
    parser.add_argument("star_path", metavar="particles", type=Path,
                        help="input particles STAR file (including CTF/optics metadata)")
    parser.add_argument(
        "--outdir", dest="results_dir", type=Path,
        help=("parent results directory; writes "
              "<EMPIAR-number>_<SLURM-job-id>/kam_pipeline inside it"))
    parser.add_argument("--dataset-label",
                        help="provenance label only; never changes the algorithm")
    parser.add_argument("--datadir", dest="data_folder", type=Path)
    parser.add_argument("--strip-prefix", dest="stack_prefix")
    parser.add_argument("--native-pixel-size", type=float)
    parser.add_argument("--n-select", type=int, default=DEFAULT_N_SELECT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--denoise-res", type=int, default=DEFAULT_DENOISE_RES,
        help="side length used by FastPCA denoising")
    parser.add_argument(
        "--denoise-max-order", type=int, default=DEFAULT_DENOISE_MAX_ORDER,
        help="largest angular order |nu| of the Fourier-Bessel basis in FastPCA/CWF denoising (the 64 px basis supports 88)")
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
    parser.add_argument(
        "--symmetry", default="C1",
        help="point group of the molecule (C1, Cn, Dn, T, O or I; axes as in kam.volume): the volume coefficients are restricted to it")
    parser.add_argument(
        "--shift-step-px", type=float, default=DEFAULT_SHIFT_STEP_PX,
        help=("spacing of the rings of the shift distribution in moment-grid pixels (sub-pixel allowed); the per-image shifts "
              "are modeled by an isotropic radial distribution on these rings, "
              "fitted jointly with the volume and the viewing density in each of the M1, M2 and M3 stages (see kam.shift)"))
    parser.add_argument(
        "--shift-max-px", type=float, default=None,
        help=("largest ring radius in moment-grid pixels (default: a quarter of the radius of the "
              "downsampled moment images, i.e. side / 8)"))
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
    parser.add_argument(
        "--whiten-per-micrograph", action=argparse.BooleanOptionalAction, default=False,
        help="whiten every image with the noise spectrum of its micrograph instead of one spectrum for the whole dataset")
    parser.set_defaults(micrograph_noise_psd=None)
    parser.add_argument(
        "--neighbor-distance-fraction", type=float, default=DEFAULT_NEIGHBOR_FRACTION,
        help="minimum pick separation as a fraction of the box (field of view) of the particle images")
    parser.add_argument("--n-preview", type=int, default=100)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_selected_star(star_path: Path, rows, output: Path) -> None:
    """Write a STAR holding only ``rows`` of the particle loop (the last loop in the file)."""
    lines = Path(star_path).read_text().split("\n")
    start = max(i for i, line in enumerate(lines) if line.startswith("loop_")) + 1
    while start < len(lines) and lines[start].startswith("_rln"):
        start += 1
    body = [line for line in lines[start:] if line.strip()]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines[:start] + [body[k] for k in rows]) + "\n")


def pipeline_main(argv=None) -> None:
    parser = _pipeline_parser()
    args = parser.parse_args(argv)
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
        _ensure_manifest(args, paths)
        parameters = _hyperparameters(args, paths)
        if not paths["hyperparameters"].exists():
            _write_json_exclusive(paths["hyperparameters"], {**parameters, "started_utc": invocation_started_utc})
        print(f"[pipeline] hyperparameters={paths['hyperparameters']}", flush=True)
        print(json.dumps(parameters["arguments"], indent=2), flush=True)
        _execute_pipeline(args, paths, invocation_started_utc, invocation_started)
        print(f"[{datetime.now(timezone.utc).isoformat()}] PIPELINE COMPLETE "
              f"elapsed={time.monotonic() - invocation_started:.1f}s",
              flush=True)


def _execute_pipeline(args, paths, invocation_started_utc, invocation_started):
    stages = _stages(args, paths)
    statuses = {}
    observed = {}

    def run_stage(label, function) -> None:
        stage_started = time.monotonic()
        print(f"[{datetime.now(timezone.utc).isoformat()}] STEP {len(statuses) + 1}/{len(stages)} START {label}", flush=True)
        status = function()
        observed[label] = float(time.monotonic() - stage_started)
        statuses[label] = "dry_run" if args.dry_run else status
        print(f"[{datetime.now(timezone.utc).isoformat()}] STEP END {label} "
              f"status={statuses[label]} elapsed={observed[label]:.1f}s", flush=True)

    for label, _, function in stages:
        run_stage(label, function)
    if not args.dry_run:
        invocation_finished_utc = datetime.now(timezone.utc).isoformat()
        _write_timing_report(
            args, paths, stages, statuses, observed,
            invocation_started_utc, invocation_finished_utc,
            time.monotonic() - invocation_started)
        _write_json_atomic(paths["root"] / "pipeline_report.json", {
            "complete": True,
            "dataset": _dataset_label(args),
            "initializer": str(paths["m3"] / "recon.mrc"),
            "reconstructions": {f"M{order}": str(paths[f"m{order}"] / "recon.mrc") for order in (1, 2, 3)},
            "diagnostics": {
                "denoising_preview": str(paths["denoising"] / "denoising_preview.pdf"),
                "selection_diagnostic": str(paths["selection"] / "selection_diagnostic.pdf"),
                "selection_scores": str(paths["selection"] / "selection_scores.pdf"),
                "selection_selected_sample": str(
                    paths["selection"] / "selection_selected_sample.pdf"),
                "redenoising_preview": str(paths["redenoised"] / "denoising_preview.pdf"),
            },
            "timing_report": str(paths["timing_report"]),
            "hyperparameters": str(paths["hyperparameters"]),
            "log": str(paths["log"]),
        })
    print(f"[pipeline] finished M1, M2 and M3; output={paths['root']}",
          flush=True)


# ===========================================================================
# denoise and select subcommands (run by the pipeline as separate processes)
# ===========================================================================

def _denoise_parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("star_path", metavar="particles", type=Path,
                        help="input particles STAR file")
    parser.add_argument(
        "--datadir", dest="data_folder",
        type=Path,
        help="Root for relative stack paths; default is the STAR directory",
    )
    parser.add_argument(
        "--strip-prefix", dest="stack_prefix",
        help="Optional leading relative directory to strip from STAR stack paths",
    )
    parser.add_argument(
        "--native-pixel-size",
        type=float,
        help="Optional A/pixel override; default uses RELION metadata",
    )
    parser.add_argument("--denoise-res", type=int, default=DEFAULT_DENOISE_RES)
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
    parser.add_argument("--outdir", dest="output_dir",
                        type=Path, required=True)
    parser.add_argument("--background-radius", type=float, default=0.9)
    parser.add_argument(
        "--denoise-max-order", type=int, default=DEFAULT_DENOISE_MAX_ORDER,
        help="largest angular order |nu| of the Fourier-Bessel basis in FastPCA/CWF denoising (the 64 px basis supports 88)")
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
    parser.add_argument(
        "--whiten-per-micrograph", action=argparse.BooleanOptionalAction, default=False,
        help="whiten every image with the noise spectrum of its micrograph instead of one spectrum for the whole dataset")
    parser.add_argument(
        "--micrograph-noise-psd", type=Path,
        help="manifest.npz of an earlier denoising whose per-micrograph noise spectra are used (they are estimated otherwise)")
    return parser.parse_args(argv)


def denoise_main(argv=None):
    denoise_relion_particles(FastCWFDenoiseConfig(**vars(_denoise_parse_args(argv))))


def _selection_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--denoising-dir", type=Path, required=True)
    parser.add_argument("--n-select", type=int, default=10_000)
    parser.add_argument("--n-preview", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--neighbor-distance-fraction", type=float, default=DEFAULT_NEIGHBOR_FRACTION,
        help="minimum pick separation as a fraction of the box (field of view) of the particle images")
    parser.add_argument("--max-score-quantile", type=float, default=0.95)
    parser.add_argument("--auto-score-quantile", action=argparse.BooleanOptionalAction, default=True,
                        help="use the tightest ceiling leaving --n-select candidates (the default); "
                             "--no-auto-score-quantile uses --max-score-quantile instead")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--outdir", dest="output_dir",
                        type=Path, required=True)
    return parser


def select_main(argv=None):
    select_particles(**vars(_selection_parser().parse_args(argv)))


def _add_fit_options(parser: argparse.ArgumentParser) -> None:
    """Options of the moment matching shared by ``MoM reconstruct`` and ``MoM fit``."""
    parser.add_argument("--outdir", type=Path, required=True, help="new output directory")
    parser.add_argument("--viewing-degree", type=int, choices=(0, 2, 4), default=DEFAULT_VIEWING_DEGREE,
                        help="degree of the positive spherical-harmonic viewing density")
    parser.add_argument("--symmetry", default="C1", help="point group of the molecule (C1, Cn, Dn, T, O or I)")
    parser.add_argument("--shift-step-px", type=float, default=DEFAULT_SHIFT_STEP_PX,
                        help="ring spacing of the fitted shift distribution, in moment-grid pixels")
    parser.add_argument("--shift-max-px", type=float, default=None,
                        help="largest ring radius in moment-grid pixels (default: side / 8)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")


def _fit_moments(root: Path, args: argparse.Namespace, side: int) -> None:
    """M1 -> M2 -> M2 selection -> M3 on the moments in ``root/prepared`` (``side``-pixel moment images); the map is ``root/M3/recon.mrc``."""
    for order in (1, 2, 3):
        if order == 3:
            select_m2_start(root / "prepared", root / "M2", root / "M2_selected", ell_max=args.ell_max,
                            viewing_degree=args.viewing_degree, symmetry=args.symmetry, device=args.device)
        print(f"[MoM] M{order}", flush=True)
        reconstruct_order(ReconstructionConfig(
            prepared_dir=root / "prepared", output_dir=root / f"M{order}", order=order,
            ell_max=args.ell_max, viewing_degree=args.viewing_degree, symmetry=args.symmetry,
            n_init=1 if order == 3 else N_INIT, maxiter=_maxiter(order), device=args.device, seed=args.seed,
            initial_fit={1: None, 2: root / "M1", 3: root / "M2_selected"}[order],
        ), _shift_settings(args, side))
    print(f"[MoM] map: {root / 'M3' / 'recon.mrc'}", flush=True)


def _reconstruct_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="MoM reconstruct", allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=("MoM ab initio map from a stack of denoised particle images: the moment and M1 -> M2 -> M3 stages of "
                     "MoM pipeline without denoising and selection. The map is OUTDIR/M3/recon.mrc."))
    parser.add_argument("stack", type=Path, help="MRC stack of square denoised images; the pixel size is read from its header")
    parser.add_argument("--moment-res", type=int, help="side of the moment images (default: the side of the stack)")
    parser.add_argument("--vol-degree", dest="ell_max", metavar="VOL_DEGREE", type=int, default=DEFAULT_ELL_MAX,
                        help="maximum SphFB angular degree of the volume")
    parser.add_argument("--batch-size", type=int, default=256)
    _add_fit_options(parser)
    return parser


def reconstruct_main(argv=None):
    args = _reconstruct_parser().parse_args(argv)
    with mrcfile.open(args.stack, permissive=True) as handle:
        images = np.asarray(handle.data, dtype=np.float32)
        pixel = float(handle.voxel_size.x)
    args.moment_res = images.shape[-1] if args.moment_res is None else args.moment_res
    root = args.outdir.resolve()
    use_device(args.device)
    print(f"[MoM] {len(images)} images of {images.shape[-1]} px at {pixel:.4g} A -> moments on {args.moment_res} px", flush=True)
    prepare_moments_from_images(
        images, pixel, root / "prepared", moment_res=args.moment_res, ell_max=args.ell_max, batch_size=args.batch_size,
        dataset=args.stack.stem, source=args.stack.resolve())
    del images
    _fit_moments(root, args, args.moment_res)
    return 0


def _fit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="MoM fit", allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=("MoM ab initio map from precomputed moments (e.g. examples/moments): the M1 -> M2 -> M3 stages of MoM "
                     "reconstruct. The volume degree is that of the moments. The map is OUTDIR/M3/recon.mrc."))
    parser.add_argument("moments", type=Path, help="moments.npz of an earlier run (OUTDIR/prepared/moments.npz)")
    _add_fit_options(parser)
    return parser


def fit_main(argv=None):
    args = _fit_parser().parse_args(argv)
    root = args.outdir.resolve()
    use_device(args.device)
    prepare_moments_from_file(args.moments, root / "prepared")
    with np.load(root / "prepared" / "moments.npz", allow_pickle=False) as saved:
        args.ell_max, side, n_images = int(saved["fb_ell_max"]), int(saved["ds_res"]), int(saved["n_images"])
    print(f"[MoM] moments of {n_images} images on {side} px, volume degree {args.ell_max}: {args.moments}", flush=True)
    _fit_moments(root, args, side)
    return 0


COMMANDS = {
    "pipeline": pipeline_main,
    "reconstruct": reconstruct_main,
    "fit": fit_main,
    "denoise": denoise_main,
    "select": select_main,
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
