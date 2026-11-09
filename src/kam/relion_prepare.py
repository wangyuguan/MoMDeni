"""RELION preparation (``kam relion_prepare``) and post-processing (``kam postprocess``) of a finished pipeline run.

The refinement launcher contains site-specific Slurm, module, MPI, and GPU settings (Della).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import textwrap
import time

import mrcfile
import numpy as np

from .preprocessing import RelionParticles
from .utils import fourier_resample_real, sha256_file

_OPTICS_COLUMNS = (
    "_rlnVoltage",
    "_rlnSphericalAberration",
    "_rlnAmplitudeContrast",
    "_rlnMtfFileName",
    "_rlnMicrographOriginalPixelSize",
    "_rlnCtfDataAreCtfPremultiplied",
    "_rlnBeamTiltX",
    "_rlnBeamTiltY",
    "_rlnOddZernike",
    "_rlnEvenZernike",
)
_PARTICLE_IDENTITY_COLUMNS = {
    "_rlnMicrographName",
    "_rlnMicrographMovieName",
    "_rlnParticleName",
    "_rlnImageOriginalName",
    "_rlnGroupName",
    "_rlnGroupNumber",
}
_POSE_COLUMNS = {
    "_rlnAngleRot",
    "_rlnAngleTilt",
    "_rlnAnglePsi",
    "_rlnAngleRotPrior",
    "_rlnAngleTiltPrior",
    "_rlnAnglePsiPrior",
}


def _canonical_dataset(value):
    value = str(value).strip().upper()
    if value.isdigit():
        value = f"EMPIAR-{value}"
    return value


def _load_source(summary_path, map_path):
    """The M3 summary, its map and the voxel size of the map."""
    config = json.loads(summary_path.read_text())
    with mrcfile.open(map_path, mode="r", permissive=False) as handle:
        volume = np.array(handle.data, dtype=np.float64, copy=True)
        voxel = np.array(
            [float(handle.voxel_size[axis]) for axis in ("x", "y", "z")]
        )
    return config, volume, voxel


def _common_value(particles, name):
    values = np.asarray(particles.metadata[name])
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        strings = np.asarray(values, dtype=str)
        if np.any(strings != strings[0]):
            raise ValueError(f"one optics group cannot represent varying {name}")
        return strings[0]
    if not np.allclose(numeric, numeric[0], rtol=1e-7, atol=1e-10):
        raise ValueError(f"one optics group cannot represent varying {name}")
    return float(numeric[0])


def _particle_columns(particles):
    names = []
    for name in particles.metadata:
        if name in ("_rlnImageName", "_rlnOpticsGroup", "_rlnRandomSubset"):
            continue
        if name in _POSE_COLUMNS or name.startswith("_rlnOrigin"):
            continue
        if name in _OPTICS_COLUMNS or name in {
            "_rlnImagePixelSize",
            "_rlnImageSize",
            "_rlnImageDimensionality",
        }:
            continue
        if (
            name in _PARTICLE_IDENTITY_COLUMNS
            or name.startswith("_rlnCoordinate")
            or name.startswith("_rlnDefocus")
            or name.startswith("_rlnCtf")
            or name.startswith("_rlnHelical")
            or name == "_rlnPhaseShift"
        ):
            names.append(name)
    return names


def _format_star_value(value):
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return format(value, ".17g")
    value = str(value)
    if not value or any(character.isspace() for character in value) or value[0] in "#_;":
        if "'" not in value:
            return f"'{value}'"
        if '"' not in value:
            return f'"{value}"'
        raise ValueError("STAR value contains whitespace and both quote characters")
    return value


def _write_loop(stream, block_name, labels, rows):
    stream.write(f"data_{block_name}\n\nloop_\n")
    for index, label in enumerate(labels, 1):
        stream.write(f"{label} #{index}\n")
    for row in rows:
        stream.write("\t".join(_format_star_value(value) for value in row) + "\n")
    stream.write("\n")


def _write_relion_star(particles, destination, native_side, native_pixel):
    optics_labels = [
        "_rlnOpticsGroupName",
        "_rlnOpticsGroup",
        "_rlnImagePixelSize",
        "_rlnImageSize",
        "_rlnImageDimensionality",
    ]
    optics_row = ["opticsGroup1", 1, native_pixel, native_side, 2]
    for name in _OPTICS_COLUMNS:
        if name in particles.metadata:
            optics_labels.append(name)
            optics_row.append(_common_value(particles, name))

    retained = _particle_columns(particles)
    particle_labels = ["_rlnImageName", "_rlnOpticsGroup", *retained]
    columns = {name: np.asarray(particles.metadata[name]) for name in retained}
    stack_paths = [path.resolve(strict=True) for path in particles.stack_paths]

    def particle_rows():
        for row in range(particles.n):
            stack = stack_paths[int(particles.stack_ids[row])]
            image_name = f"{int(particles.stack_indices[row]) + 1}@{stack}"
            yield [image_name, 1, *(columns[name][row] for name in retained)]

    destination = Path(destination)
    with destination.open("x") as stream:
        stream.write(
            "# RELION 5 input: absolute raw-particle paths; poses, origins and random subsets deliberately omitted.\n\n"
        )
        _write_loop(stream, "optics", optics_labels, [optics_row])
        _write_loop(stream, "particles", particle_labels, particle_rows())
    dropped = sorted(
        name
        for name in particles.metadata
        if name == "_rlnRandomSubset"
        or name in _POSE_COLUMNS
        or name.startswith("_rlnOrigin")
    )
    return {
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "n_particles": particles.n,
        "n_stacks": len(stack_paths),
        "optics_columns": optics_labels,
        "particle_columns": particle_labels,
        "dropped_pose_origin_subset_columns": dropped,
    }


def _write_mrc(path, volume, pixel):
    with mrcfile.new(path, overwrite=False) as handle:
        handle.set_data(np.ascontiguousarray(volume, dtype=np.float32))
        handle.voxel_size = pixel
        handle.header.mapc, handle.header.mapr, handle.header.maps = 1, 2, 3
        handle.header.nxstart = handle.header.nystart = handle.header.nzstart = 0
        handle.header.origin.x = handle.header.origin.y = handle.header.origin.z = 0
        handle.update_header_stats()


def prepare_refinement(source_summary, source_map, particles, output, dataset, native_side, native_pixel, data_folder, stack_prefix):
    """Write the RELION inputs of a finished pipeline run to the new ``output`` directory: the M3 map resampled to the native
    grid, the particle STAR without poses or origins, and
    ``provenance.json`` (returned)."""
    config, source, source_voxel = _load_source(source_summary, source_map)
    source_side = source.shape[0]
    native, imaginary_error = fourier_resample_real(source, native_side)
    restored, _ = fourier_resample_real(native, source_side)
    roundtrip_error = float(
        np.linalg.norm(restored - source)
        / max(float(np.linalg.norm(source)), np.finfo(float).tiny)
    )
    mean_error = float(
        abs(native.mean() - source.mean())
        / max(abs(float(source.mean())), float(np.linalg.norm(source)) / source.size,
              np.finfo(float).tiny)
    )
    # The initializer handed to RELION is the Fourier-resampled MoM map itself:
    # no support mask and no pre-filter.

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-relion-", dir=output) as temporary:
        staging = Path(temporary)
        star_report = _write_relion_star(
            particles,
            staging / "particles.star",
            native_side,
            native_pixel,
        )
        star_report["path"] = str((output / "particles.star").resolve())
        _write_mrc(staging / "initial_model.mrc", native, native_pixel)
        provenance = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "source": {
                "summary": str(source_summary),
                "summary_sha256": sha256_file(source_summary),
                "map": str(source_map),
                "map_sha256": sha256_file(source_map),
                "n_initializations": config["n_initializations"],
                "side": source_side,
                "pixel_size_angstrom": source_voxel.tolist(),
                "field_of_view_angstrom": source_side * float(source_voxel[0]),
            },
            "native_geometry": {
                "side": native_side,
                "pixel_size_angstrom": native_pixel,
                "field_of_view_angstrom": native_side * native_pixel,
            },
            "fourier_resampling": {
                "relative_roundtrip_error": roundtrip_error,
                "relative_mean_error": mean_error,
                "relative_imaginary_norm": imaginary_error,
            },
            "particles": {
                "source_star": str(particles.star_path),
                "source_star_sha256": sha256_file(particles.star_path),
                "source_fingerprint": particles.fingerprint(),
                "data_folder": str(particles.star_path.parent if data_folder is None else data_folder),
                "stack_prefix": stack_prefix,
                "n_particles": particles.n,
                "n_stacks": len(particles.stack_paths),
                "native_side": particles.res,
                "native_pixel_size_angstrom": particles.pixel_size,
                "metadata_pixel_size_angstrom": particles.metadata_pixel_size,
            },
            "output_star": star_report,
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        for name in ("particles.star", "initial_model.mrc", "provenance.json"):
            os.link(staging / name, output / name)
    print(json.dumps(provenance, indent=2), flush=True)
    print(f"Prepared RELION input in {output}", flush=True)
    return provenance


def _safe_symmetry(value: str) -> str:
    symmetry = str(value).strip().upper()
    return symmetry


def _shell_literal(value) -> str:
    text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    return shlex.quote(text)


def _write_atomic_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o755)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _gpu_preflight_program():
    """Check CUDA runtime devices inside each MPI task before starting RELION."""
    return textwrap.dedent('''\
        import ctypes
        import json
        import os
        from pathlib import Path
        import socket
        import sys

        rank = int(os.environ.get("SLURM_PROCID", "-1"))
        report = {
            "rank": rank,
            "hostname": socket.gethostname(),
            "environment": {name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "SLURM_JOB_GPUS",
                "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_LOCALID")},
            "devices": [],
            "complete": False,
        }
        try:
            runtime = ctypes.CDLL("libcudart.so.12")
            runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
            runtime.cudaGetErrorString.restype = ctypes.c_char_p
            runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
            runtime.cudaGetDeviceCount.restype = ctypes.c_int
            runtime.cudaSetDevice.argtypes = [ctypes.c_int]
            runtime.cudaSetDevice.restype = ctypes.c_int
            runtime.cudaDeviceGetPCIBusId.argtypes = [
                ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            runtime.cudaDeviceGetPCIBusId.restype = ctypes.c_int
            runtime.cudaFree.argtypes = [ctypes.c_void_p]
            runtime.cudaFree.restype = ctypes.c_int

            def check(status, operation):
                if status:
                    message = runtime.cudaGetErrorString(status).decode()
                    raise RuntimeError(f"{operation}: CUDA {status}: {message}")

            count = ctypes.c_int()
            check(runtime.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
            report["visible_device_count"] = count.value
            if count.value < 1:
                raise RuntimeError("no CUDA device is visible to this MPI rank")
            for ordinal in range(count.value):
                check(runtime.cudaSetDevice(ordinal), f"cudaSetDevice({ordinal})")
                check(runtime.cudaFree(None), f"CUDA context initialization ({ordinal})")
                bus_id = ctypes.create_string_buffer(32)
                check(runtime.cudaDeviceGetPCIBusId(bus_id, len(bus_id), ordinal),
                      f"cudaDeviceGetPCIBusId({ordinal})")
                report["devices"].append({
                    "local_ordinal": ordinal, "pci_bus_id": bus_id.value.decode(),
                    "context_initialized": True,
                })
            if count.value < 2:
                report["warning"] = (
                    "fewer than two devices visible; RELION automatic mapping uses "
                    "rank-local CUDA device counts")
            report["complete"] = True
        except Exception as error:
            report["error"] = str(error)
        Path(f"gpu_rank_{rank}.json").write_text(json.dumps(report, indent=2) + "\\n")
        print("RELION CUDA preflight " + json.dumps(report), flush=True)
        sys.exit(0 if report["complete"] else 2)
        ''')


_SBATCH_OVERRIDDEN_OPTIONS = frozenset((
    "--job-name", "-J", "--chdir", "-D", "--output", "-o", "--error", "-e"))


def _current_batch_script_text():
    """Return the submitted batch script of the running SLURM job."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError(
            "relion_prepare must run inside the pipeline's SLURM job so the "
            "refinement launcher can copy that job's #SBATCH header")
    return subprocess.run(
        ["scontrol", "write", "batch_script", job_id, "-"],
        capture_output=True, text=True, check=True).stdout


def _copied_sbatch_header(script_text, overrides):
    """Copy the submitting script's #SBATCH header, replacing per-job options.

    The pipeline job's own header carries every site-specific setting
    (account, partition, GPUs, memory, time, mail), so the refinement job
    inherits them instead of hard-coding one cluster's values here.
    """
    header = []
    for line in script_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            if stripped and not stripped.startswith("#"):
                break
            continue
        option = stripped[len("#SBATCH"):].strip()
        option = option.split("=", 1)[0].split(None, 1)[0]
        if option not in _SBATCH_OVERRIDDEN_OPTIONS:
            header.append(stripped)
    header.extend(f"#SBATCH {name}={value}"
                  for name, value in overrides.items())
    return header


def _write_refinement_script(preparation, dataset, parameters):
    """Write a launcher that consumes completed preparation and runs RELION."""
    root = Path(__file__).resolve().parents[2]
    preparation = Path(preparation).resolve()
    result = preparation.parent / "relion_refine"
    script = preparation / "run_refinement.sh"
    job_suffix = dataset.split("-", 1)[-1]
    diameter = parameters["particle_diameter_angstrom"]
    offset_range = parameters["offset_range_pixels"]
    offset_step = parameters["offset_step_pixels"]
    low_resol_join_halves = parameters["low_resol_join_halves_angstrom"]
    symmetry = parameters["symmetry"]
    rank_launcher = (
        "set -euo pipefail\npython3 -c "
        + shlex.quote(_gpu_preflight_program())
        + '\nexec relion_refine_mpi "$@"')
    header = _copied_sbatch_header(_current_batch_script_text(), {
        "--job-name": f"relion{job_suffix}",
        "--chdir": str(root),
        "--output": f"{preparation}/slurm_%j.out",
        "--error": f"{preparation}/slurm_%j.err",
    })
    gres = next(
        (line.split("=", 1)[1].strip() for line in header
         if line.startswith("#SBATCH --gres=")), None)
    text = "\n".join([
        "#!/usr/bin/env bash",
        *header,
        "",
        "set -euo pipefail",
        "",
        f"ROOT={_shell_literal(root)}",
        f"PREPARED={_shell_literal(preparation)}",
        f"SCRIPT={_shell_literal(script)}",
        f"RESULT={_shell_literal(result)}",
        "",
        'if [[ -z "${SLURM_JOB_ID:-}" ]]; then',
        '    exec sbatch --wait "$SCRIPT"',
        "fi",
        "",
        'VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"',
        'GPU_COUNT="${SLURM_GPUS_ON_NODE:-0}"',
        'if [[ "${CUDA_VISIBLE_DEVICES+x}" ]]; then',
        '    VISIBLE_GPUS="$CUDA_VISIBLE_DEVICES"',
        "    GPU_COUNT=0",
        "fi",
        'if [[ -n "$VISIBLE_GPUS" && "$VISIBLE_GPUS" != "-1" && "$VISIBLE_GPUS" != "NoDevFiles" ]]; then',
        '    IFS=, read -r -a GPU_IDS <<< "$VISIBLE_GPUS"',
        '    GPU_COUNT="${#GPU_IDS[@]}"',
        "fi",
        'CPU_COUNT="${SLURM_CPUS_ON_NODE:-0}"',
        'if [[ ! "$GPU_COUNT" =~ ^[0-9]+$ ]] || (( GPU_COUNT < 2 )); then',
        '    echo "RELION needs two visible GPUs; run this script outside the current allocation to submit it." >&2',
        "    exit 2",
        "fi",
        'if [[ ! "$CPU_COUNT" =~ ^[0-9]+$ ]] || (( CPU_COUNT < 24 )); then',
        '    echo "RELION needs at least 24 allocated CPUs (3 MPI ranks x 8); current allocation is too small." >&2',
        "    exit 2",
        "fi",
        "",
        'JOB="$RESULT/job_${SLURM_JOB_ID}"',
        'mkdir -p "$RESULT"',
        'mkdir "$JOB"',
        'exec > >(tee "$JOB/refinement.out") 2> >(tee "$JOB/refinement.err" >&2)',
        'cp "$SCRIPT" "$JOB/"',
        'mkdir "$JOB/inputs"',
        'for name in particles.star initial_model.mrc provenance.json; do',
        '    cp "$PREPARED/$name" "$JOB/inputs/$name"',
        'done',
        'sha256sum "$JOB/inputs/"* > "$JOB/input_sha256.txt"',
        'START_SECONDS=$(date +%s)',
        'trap \'status=$?; elapsed=$(( $(date +%s) - START_SECONDS )); printf "{\\"elapsed_seconds\\": %s, \\"exit_code\\": %s}\\n" "$elapsed" "$status" > "$JOB/timing.json"\' EXIT',
        "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1",
        'export PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT/src"',
        "",
        "set +u",
        "source /etc/profile.d/modules.sh",
        "module purge",
        "module use /projects/CRYOEM/Modules/modulefiles-rhel9",
        "module load relion/5.0.1/gcc-11.5.0-gpu",
        "set -u",
        'export OMP_NUM_THREADS=8 SLURM_EXPORT_ENV=ALL',
        'cd "$JOB"',
        "relion_refine --version > relion_version.txt 2>&1",
        "command -v relion_refine_mpi | tee relion_executable.txt",
        "nvidia-smi --query-gpu=name,memory.total --format=csv",
        "",
        "# The initializer is the unmasked MoM map; RELION low-passes it to 60 A (--ini_high 60, the GUI default).",
        "command=(",
        '    srun --export=ALL --mpi=pmix_v3 --kill-on-bad-exit=1 -n 3 -c "$OMP_NUM_THREADS"',
        (f"    --gres={gres} --cpu-bind=cores --gpu-bind=none"
         if gres else "    --cpu-bind=cores --gpu-bind=none"),
        f"    bash -c {shlex.quote(rank_launcher)} relion_rank",
        '    --i "$JOB/inputs/particles.star" --ref "$JOB/inputs/initial_model.mrc"',
        '    --o "$JOB/run" --auto_refine --split_random_halves',
        "    --firstiter_cc --trust_ref_size --ini_high 60",
        f"    --ctf --sym {_shell_literal(symmetry)} --particle_diameter {diameter:.17g}",
        "    --flatten_solvent --zero_mask --norm --scale",
        "    --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4",
        f"    --offset_range {offset_range:.17g} --offset_step {offset_step:.17g}",
        f"    --low_resol_join_halves {low_resol_join_halves:.17g}",
        "    --dont_combine_weights_via_disc --pool 3 --pad 2",
        f"    --random_seed {parameters['seed']} --j \"$OMP_NUM_THREADS\" --gpu ''",
        ")",
        "printf '%q ' \"${command[@]}\" > command.txt",
        "printf '\\n' >> command.txt",
        'echo "Starting RELION refinement: $(date --iso-8601=seconds)"',
        '"${command[@]}"',
        'echo "Finished RELION refinement: $(date --iso-8601=seconds)"',
        "",
    ])
    _write_atomic_executable(script, text)
    return {
        "script": str(script),
        "result_dir": str(result),
        "dataset": dataset,
        "mpi_ranks": 3,
        "requested_gpus": gres,
        **parameters,
    }


# RELION's mask (particle) diameter is MASK_DIAMETER_FACTOR times the diameter of the mean particle image that encloses
# MASK_MASS_FRACTION of its positive mass; the mean image is the radial profile given by the first moment (no reference).
MASK_DIAMETER_FACTOR = 1.25
MASK_MASS_FRACTION = 0.99


def particle_diameter_from_m1(prepared_dir):
    """Diameter (Angstrom) of the mean denoised particle image from the first moment of ``prepared_dir/moments.npz``.

    M1 = E c_0 holds the rotationally averaged Fourier transform of the mean image on the FB radial nodes; its Hankel transform is the
    real-space radial profile f(r).  Returns 2 R with ``int_0^R max(f, 0) r dr`` equal to MASK_MASS_FRACTION of the total."""
    from scipy.special import jv

    with np.load(Path(prepared_dir) / "moments.npz", allow_pickle=False) as saved:
        nodes, weights = saved["fb_radial_nodes"], saved["fb_radial_weights"]
        radial0 = saved["fb_radial_basis"][:, : int(saved["fb_radial_counts"][0])]
        m1 = np.real(saved["mbar1"])
        side, pixel = int(saved["ds_res"]), float(saved["pixel_size"])
    # kappa w F(kappa) on the radial nodes (the radial basis carries sqrt(kappa w)), up to a constant
    hankel = np.sqrt(nodes * weights) * (radial0 @ m1)
    r = np.linspace(0.0, side / 2, 4001)
    profile = jv(0, 2 * np.pi * np.outer(r, nodes)) @ hankel
    mass = np.cumsum(np.maximum(profile * np.sign(profile[0]), 0.0) * r)
    return float(2 * r[np.searchsorted(mass / mass[-1], MASK_MASS_FRACTION)] * pixel)


def relion_prepare_main(argv=None):
    """Prepare a completed MoM initializer and an independent RELION launcher."""
    parser = argparse.ArgumentParser(
        prog="kam relion_prepare",
        description=(
            "Prepare RELION input from kam_pipeline/MoM/M3 and generate the "
            "Della-specific run_refinement.sh used by these experiments."),
        allow_abbrev=False)
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--particles", dest="particle_star", type=Path,
                        help="override the raw STAR used by pipeline")
    parser.add_argument("--datadir", dest="data_folder", type=Path)
    parser.add_argument("--native-pixel-size", type=float)
    parser.add_argument("--symmetry", help="RELION --sym; default: the symmetry of the pipeline's volume")
    parser.add_argument("--low-resol-join-halves", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="reuse preparation only if inputs, parameters, and files match")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate inputs and print paths without writing files")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    pipeline = args.pipeline_dir.expanduser().resolve(strict=True)
    manifest_path = pipeline / "pipeline_config.json"
    manifest = json.loads(manifest_path.read_text())
    configuration = manifest["configuration"]
    dataset = _canonical_dataset(configuration["dataset_label"])
    inputs = configuration["inputs"]
    source_summary = pipeline / "MoM/M3/summary.json"
    source_map = pipeline / "MoM/M3/recon.mrc"
    particle_star = (args.particle_star if args.particle_star is not None else Path(inputs["star_path"]))
    particle_star = particle_star.expanduser().resolve(strict=True)
    data_folder = args.data_folder
    stack_prefix = None
    if args.particle_star is None:
        if data_folder is None and inputs.get("data_folder") is not None:
            data_folder = Path(inputs["data_folder"])
        stack_prefix = inputs.get("stack_prefix")
    if data_folder is not None:
        data_folder = data_folder.expanduser().resolve(strict=True)
    particles = RelionParticles(
        particle_star, data_folder=data_folder, stack_prefix=stack_prefix, pixel_size=args.native_pixel_size)
    # RELION refines the images of particle_star (the raw images, also when the moments used Fourier-cropped copies of them):
    # their box and pixel size are the native grid the MoM map is resampled to (same field of view).
    native_side, native_pixel = particles.res, particles.pixel_size
    parameters = {
        "particle_diameter_angstrom": MASK_DIAMETER_FACTOR * particle_diameter_from_m1(pipeline / "MoM" / "prepared"),
        "symmetry": _safe_symmetry(args.symmetry or configuration["model"]["symmetry"]),
        # The same translational search for every dataset: up to 1/8 of the box (the support of the MoM shift rings), 11 steps per axis.
        "offset_range_pixels": native_side / 8,
        "offset_step_pixels": native_side / 40,
        "low_resol_join_halves_angstrom": args.low_resol_join_halves,
        "seed": args.seed,
    }
    preparation = pipeline.parent / "relion_prepare"
    request = {
        "pipeline_dir": str(pipeline),
        "pipeline_manifest_sha256": sha256_file(manifest_path),
        "source_map_sha256": sha256_file(source_map),
        "source_summary_sha256": sha256_file(source_summary),
        "dataset": dataset,
        "particle_star": str(particle_star),
        "particle_source_fingerprint": particles.fingerprint(),
        "data_folder": None if data_folder is None else str(data_folder),
        "stack_prefix": stack_prefix,
        "native_side": native_side,
        "native_pixel_size_angstrom": native_pixel,
        "n_particles": particles.n,
        "parameters": parameters,
    }
    inputs_loaded = time.perf_counter()
    if args.dry_run:
        print(json.dumps({"request": request, "preparation_dir": str(preparation),
                          "refinement_script": str(preparation / "run_refinement.sh"),
                          "refinement_dir": str(pipeline.parent / "relion_refine")},
                         indent=2), flush=True)
        return 0

    provenance_path = preparation / "provenance.json"
    if preparation.exists() and any(preparation.iterdir()):
        if not args.resume:
            raise FileExistsError(f"preparation exists; use --resume: {preparation}")
        previous = json.loads(provenance_path.read_text())
        if previous.get("preparation_request") != request or not previous.get("complete"):
            raise ValueError("existing preparation is incomplete or uses different inputs/settings")
        print(f"Reusing RELION preparation: {preparation}", flush=True)
        print(f"Run: bash {shlex.quote(str(preparation / 'run_refinement.sh'))}", flush=True)
        return 0

    provenance = prepare_refinement(
        source_summary, source_map, particles, preparation, dataset, native_side, native_pixel, data_folder, stack_prefix)
    inputs_done = time.perf_counter()
    refinement = _write_refinement_script(preparation, dataset, parameters)
    provenance.update(
        preparation_request=request,
        refinement=refinement,
        complete=True,
    )
    completed = time.perf_counter()
    provenance["timings_seconds"] = {
        "input_loading": inputs_loaded - started,
        "native_map_and_star_preparation": inputs_done - inputs_loaded,
        "launcher_and_output_checksums": completed - inputs_done,
        "total": completed - started,
    }
    with tempfile.NamedTemporaryFile(
            mode="w", dir=preparation, prefix=".provenance-", suffix=".json",
            delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(provenance, indent=2) + "\n")
    os.replace(temporary, provenance_path)
    print(f"RELION preparation complete: {preparation}", flush=True)
    print(f"Run: bash {shlex.quote(refinement['script'])}", flush=True)
    return 0


# ===========================================================================
# Post-processing of a finished refinement (masked, sharpened resolution)
# ===========================================================================

_POSTPROCESS_PROTEIN_VOLUME_PER_DALTON = 1.21
_POSTPROCESS_MASK_SAFETY = 1.3


def _largest_component(mask):
    """Keep the largest connected component of a Boolean mask."""
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == (1 + int(np.argmax(sizes)))


def solvent_mask_from_map(volume, pixel_size, molecular_weight, lowpass=15.0, extend=3,
                          edge_angstrom=6.0):
    """Build a reference-free solvent mask from a reconstruction.

    The threshold is set so the enclosed volume matches the expected
    molecular volume rather than by eye, on a low-passed copy so that noise
    cannot define the support; the mask is then restricted to its largest
    component, extended, and given a cosine edge, because a hard edge
    convolves the Fourier shells and inflates the masked FSC.
    """
    from scipy import ndimage

    volume = np.asarray(volume, dtype=np.float64)
    smoothed = ndimage.gaussian_filter(
        volume, lowpass / (2.355 * pixel_size))
    target_voxels = (float(molecular_weight)
                     * _POSTPROCESS_PROTEIN_VOLUME_PER_DALTON
                     * _POSTPROCESS_MASK_SAFETY / pixel_size ** 3)
    fraction = float(np.clip(1.0 - target_voxels / smoothed.size, 0.5,
                             1.0 - 1e-9))
    threshold = float(np.quantile(smoothed, fraction))
    core = _largest_component(smoothed > threshold)
    grown = ndimage.binary_dilation(core, iterations=int(extend))
    distance = ndimage.distance_transform_edt(~grown) * pixel_size
    mask = np.clip(0.5 * (1.0 + np.cos(np.pi * distance / edge_angstrom)),
                   0.0, 1.0)
    mask[grown] = 1.0
    mask[distance >= edge_angstrom] = 0.0
    return mask, dict(
        lowpass_angstrom=float(lowpass),
        molecular_weight_daltons=float(molecular_weight),
        volume_per_dalton_cubic_angstrom=_POSTPROCESS_PROTEIN_VOLUME_PER_DALTON,
        volume_safety_factor=_POSTPROCESS_MASK_SAFETY,
        threshold=threshold,
        core_voxel_fraction=float(core.mean()),
        mask_weight_fraction=float(mask.mean()),
        extend_voxels=int(extend),
        cosine_edge_angstrom=float(edge_angstrom),
    )


def _parse_star_values(path, keys):
    text = Path(path).read_text()
    values = {}
    for key in keys:
        match = re.search(rf"{key}\s+(\S+)", text)
        values[key] = match.group(1) if match else None
    return values


def postprocess_main(argv=None):
    """Mask and sharpen a finished refinement, and record its resolution."""
    parser = argparse.ArgumentParser(
        prog="kam postprocess", allow_abbrev=False,
        description=(
            "Build a reference-free solvent mask from the refined map and run "
            "relion_postprocess on the unfiltered half maps. The masked, "
            "phase-randomization-corrected FSC is the number published maps "
            "quote; the refinement's own running estimate is unmasked and "
            "systematically worse."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--relion-dir", type=Path, required=True,
                        help="relion_refine directory holding job_<id>")
    parser.add_argument("--outdir", type=Path,
                        help="defaults to <relion-dir>/../postprocess")
    parser.add_argument("--molecular-weight", type=float, required=True,
                        help="complex mass in daltons; sets the mask threshold")
    parser.add_argument("--mask-lowpass", type=float, default=15.0)
    parser.add_argument("--mask-extend", type=int, default=3)
    parser.add_argument("--mask-edge", type=float, default=6.0)
    parser.add_argument("--autob-lowres", type=float, default=10.0)
    args = parser.parse_args(argv)

    relion_dir = args.relion_dir.expanduser().resolve(strict=True)
    jobs = sorted(relion_dir.glob("job_*"),
                  key=lambda path: int(path.name.split("_")[1]))
    jobs = [job for job in jobs
            if (job / "run_half1_class001_unfil.mrc").is_file()]
    if not jobs:
        raise FileNotFoundError(
            f"no finished refinement with half maps in {relion_dir}")
    job = jobs[-1]
    output = (args.outdir.expanduser().resolve() if args.outdir
              else relion_dir.parent / "postprocess")
    output.mkdir(parents=True, exist_ok=True)

    with mrcfile.open(job / "run_class001.mrc") as handle:
        refined = handle.data.astype(np.float64)
        pixel_size = float(handle.voxel_size.x)
    mask, mask_report = solvent_mask_from_map(
        refined, pixel_size, args.molecular_weight, lowpass=args.mask_lowpass,
        extend=args.mask_extend, edge_angstrom=args.mask_edge)
    _write_mrc(output / "mask.mrc", mask, pixel_size)
    print(f"solvent mask: {json.dumps(mask_report)}", flush=True)

    command = [
        "relion_postprocess",
        "--i", str(job / "run_half1_class001_unfil.mrc"),
        "--mask", str(output / "mask.mrc"),
        "--angpix", f"{pixel_size:.6f}",
        "--auto_bfac", "--autob_lowres", f"{args.autob_lowres:g}",
        "--o", str(output / "postprocess"),
    ]
    print("[postprocess] " + " ".join(command), flush=True)
    completed = subprocess.run(command, capture_output=True, text=True)
    (output / "postprocess.log").write_text(
        completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            "relion_postprocess failed; load the RELION module before this "
            f"step. Last output:\n{completed.stderr[-2000:]}")
    values = _parse_star_values(
        output / "postprocess.star",
        ("_rlnFinalResolution", "_rlnBfactorUsedForSharpening",
         "_rlnUnfilteredMapHalf1"))
    report = dict(
        complete=True,
        relion_job=str(job),
        masked_fsc_resolution_angstrom=float(values["_rlnFinalResolution"]),
        bfactor_used_for_sharpening=float(
            values["_rlnBfactorUsedForSharpening"]),
        fsc_criterion=0.143,
        mask=mask_report,
        pixel_size_angstrom=pixel_size,
        postprocessed_map=str(output / "postprocess.mrc"),
    )
    (output / "postprocess_report.json").write_text(
        json.dumps(report, indent=2))
    print(f"masked FSC resolution: "
          f"{report['masked_fsc_resolution_angstrom']:.2f} A "
          f"(B = {report['bfactor_used_for_sharpening']:.1f} A^2)", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0
