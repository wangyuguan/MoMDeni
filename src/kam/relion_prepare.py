"""Optional RELION preparation and Della refinement-job generation.

This experiment driver is intentionally kept outside ``src`` because its
launcher contains site-specific Slurm, module, MPI, and GPU settings.  The
core MoM package does not import it.
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
import subprocess
import sys
import tempfile
import textwrap
import time

ROOT = Path(__file__).resolve().parents[2]

import mrcfile
import numpy as np
from scipy import fft, signal

from .preprocessing import RelionParticles


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



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
    if not value.startswith("EMPIAR-") or not value[7:].isdigit():
        raise ValueError("dataset must be an EMPIAR identifier, e.g. EMPIAR-10073")
    return value


def _load_source(config_path, map_path, dataset, native_side, native_pixel):
    requested_config_path = Path(config_path).resolve(strict=True)
    map_path = Path(map_path).resolve(strict=True)
    requested_config = json.loads(requested_config_path.read_text())
    authoritative_path = requested_config_path
    config = requested_config
    completion_keys = ("fit_complete", "complete")
    if (requested_config_path.name == "config.json"
            and all(key not in requested_config for key in completion_keys)):
        authoritative_path = requested_config_path.with_name(
            "summary.json").resolve(strict=True)
        config = json.loads(authoritative_path.read_text())
        identity_keys = (
            "dataset", "order", "moment_res", "native_side",
            "native_pixel_size_angstrom", "pixel_size_angstrom",
            "n_initializations",
        )
        missing_identity = [
            key for key in identity_keys
            if key not in requested_config or key not in config
        ]
        if missing_identity:
            raise ValueError(
                "config.json and summary.json lack shared source identity "
                f"fields: {missing_identity}")
        mismatched = {
            key: (requested_config[key], config[key])
            for key in requested_config.keys() & config.keys()
            if requested_config[key] != config[key]
        }
        if mismatched:
            raise ValueError(
                "config.json disagrees with its completed summary.json: "
                f"{mismatched}")
    expected = {"order": 3, "native_side": native_side}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"source config has {key}={config.get(key)!r}; expected {value!r}"
            )
    if _canonical_dataset(config.get("dataset", "")) != dataset:
        raise ValueError("source config belongs to a different dataset")
    for complete_key in completion_keys:
        if config.get(complete_key) is not True:
            raise ValueError(
                f"source summary must report {complete_key}=true")
    source_side = config.get("moment_res")
    if (isinstance(source_side, bool) or not isinstance(source_side, int)
            or source_side < 4 or source_side % 2):
        raise ValueError(
            "source summary moment_res must be an even integer of at least 4")
    n_initializations = config.get("n_initializations")
    if (isinstance(n_initializations, bool)
            or not isinstance(n_initializations, int)
            or n_initializations < 1):
        raise ValueError(
            "source summary n_initializations must be a positive integer")

    config_native_pixel = float(config.get("native_pixel_size_angstrom", math.nan))
    source_pixel = float(config.get("pixel_size_angstrom", math.nan))
    if not math.isclose(config_native_pixel, native_pixel, rel_tol=2e-7, abs_tol=0.0):
        raise ValueError(
            "source config native pixel size disagrees with --native-pixel-size"
        )
    expected_source_pixel = native_side * native_pixel / source_side
    if not math.isclose(source_pixel, expected_source_pixel, rel_tol=2e-7, abs_tol=0.0):
        raise ValueError("source and native fields of view do not agree")

    with mrcfile.open(map_path, mode="r", permissive=False) as handle:
        volume = np.array(handle.data, dtype=np.float64, copy=True)
        voxel = np.array(
            [float(handle.voxel_size[axis]) for axis in ("x", "y", "z")]
        )
        axes = (int(handle.header.mapc), int(handle.header.mapr), int(handle.header.maps))
        starts = tuple(
            int(handle.header[key]) for key in ("nxstart", "nystart", "nzstart")
        )
        origin = tuple(float(handle.header.origin[key]) for key in ("x", "y", "z"))
    if volume.shape != (source_side,) * 3 or not np.all(np.isfinite(volume)):
        raise ValueError(
            f"source map must be a finite {source_side} x {source_side} x "
            f"{source_side} real volume")
    if not np.allclose(voxel, source_pixel, rtol=2e-6, atol=0.0):
        raise ValueError("source map voxel size disagrees with its config")
    if axes != (1, 2, 3) or starts != (0, 0, 0) or origin != (0.0, 0.0, 0.0):
        raise ValueError("source map must use canonical MRC axes and a centered origin")
    return (
        requested_config_path, authoritative_path, map_path, config, volume,
        voxel, source_side,
    )


def _common_value(particles, name):
    values = np.asarray(particles.metadata[name])
    if values.shape != (particles.n,):
        raise ValueError(f"invalid STAR column shape for {name}")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        strings = np.asarray(values, dtype=str)
        if np.any(strings != strings[0]):
            raise ValueError(f"one optics group cannot represent varying {name}")
        return strings[0]
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"nonfinite STAR values in {name}")
    if not np.allclose(numeric, numeric[0], rtol=1e-7, atol=1e-10):
        raise ValueError(f"one optics group cannot represent varying {name}")
    return float(numeric[0])


def _validate_particles(
    star_path,
    data_folder,
    stack_prefix,
    native_side,
    native_pixel,
    expected_particles,
):
    particles = RelionParticles(
        star_path,
        data_folder=data_folder,
        stack_prefix=stack_prefix,
        pixel_size=native_pixel,
    )
    if particles.res != native_side:
        raise ValueError(
            f"raw particles are {particles.res} square, not {native_side} square"
        )
    if expected_particles is not None and particles.n != expected_particles:
        raise ValueError(
            f"raw-particle STAR has {particles.n} rows, not {expected_particles} rows"
        )
    if not math.isclose(
        particles.pixel_size, native_pixel, rel_tol=2e-7, abs_tol=0.0
    ):
        raise ValueError(
            "effective particle pixel size disagrees with --native-pixel-size"
        )

    voltage = particles.column("_rlnVoltage")
    spherical_aberration = particles.column("_rlnSphericalAberration")
    amplitude_contrast = particles.column("_rlnAmplitudeContrast")
    defocus_u = particles.column("_rlnDefocusU")
    defocus_v = particles.column("_rlnDefocusV")
    particles.column("_rlnDefocusAngle")
    if np.any(voltage <= 0.0) or np.any(spherical_aberration < 0.0):
        raise ValueError("invalid voltage or spherical-aberration metadata")
    if np.any((amplitude_contrast < 0.0) | (amplitude_contrast >= 1.0)):
        raise ValueError("amplitude contrast must lie in [0, 1)")
    if np.any(defocus_u <= 0.0) or np.any(defocus_v <= 0.0):
        raise ValueError("defocus values must be positive")
    if "_rlnPhaseShift" in particles.metadata:
        particles.column("_rlnPhaseShift")

    for name in ("_rlnVoltage", "_rlnSphericalAberration", "_rlnAmplitudeContrast"):
        _common_value(particles, name)
    # RelionParticles has now checked every referenced stack header and index.
    return particles


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
            values = np.asarray(particles.metadata[name])
            if values.shape != (particles.n,):
                raise ValueError(f"invalid STAR column shape for {name}")
            try:
                numeric = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError):
                strings = np.asarray(values, dtype=str)
                if any("\n" in value or "\r" in value for value in strings):
                    raise ValueError(f"multiline STAR value in {name}")
            else:
                if not np.all(np.isfinite(numeric)):
                    raise ValueError(f"nonfinite STAR value in {name}")
            names.append(name)
    required = {"_rlnDefocusU", "_rlnDefocusV", "_rlnDefocusAngle"}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"particle STAR lacks required CTF columns: {sorted(missing)}")
    return names


def _format_star_value(value):
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("cannot write a nonfinite STAR value")
        return format(value, ".17g")
    value = str(value)
    if "\n" in value or "\r" in value:
        raise ValueError("multiline STAR values are not supported")
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
            "# RELION 5 input: absolute raw-particle paths; poses, origins, and "
            "random subsets deliberately omitted.\n\n"
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
        "sha256": _sha256(destination),
        "format": "RELION 5 optics-plus-particles STAR",
        "n_particles": particles.n,
        "n_stacks": len(stack_paths),
        "optics_groups": 1,
        "optics_columns": optics_labels,
        "particle_columns": particle_labels,
        "dropped_pose_origin_subset_columns": dropped,
        "absolute_stack_paths": True,
        "random_subsets_present": False,
    }


def _fourier_resample_real(volume, side):
    volume = np.asarray(volume, dtype=np.float64)
    if (
        volume.ndim != 3
        or len(set(volume.shape)) != 1
        or volume.shape[0] % 2
        or side % 2
    ):
        raise ValueError("source and native map sides must be even cubes")
    result = np.fft.ifftshift(volume)
    for axis in range(3):
        result = signal.resample(result, side, axis=axis, window=None)
    result = np.fft.fftshift(result)
    imaginary = float(np.linalg.norm(np.imag(result)))
    result = np.asarray(np.real(result), dtype=np.float64)
    relative_imaginary = imaginary / max(
        float(np.linalg.norm(result)), np.finfo(float).tiny
    )
    if relative_imaginary > 1e-12 or not np.all(np.isfinite(result)):
        raise FloatingPointError("Fourier interpolation did not produce a finite real map")
    return result, relative_imaginary


def _soft_spherical_support(side, pixel, diameter, edge):
    inner = diameter / 2.0
    outer = inner + edge
    if not 0.0 < inner < outer < side * pixel / 2.0:
        raise ValueError("particle support must fit strictly inside the native map")
    coordinate = (np.arange(side, dtype=np.float64) - side // 2) * pixel
    radius = np.sqrt(
        coordinate[:, None, None] ** 2
        + coordinate[None, :, None] ** 2
        + coordinate[None, None, :] ** 2
    )
    mask = np.ones_like(radius)
    mask[radius >= outer] = 0.0
    transition = (radius > inner) & (radius < outer)
    mask[transition] = 0.5 * (
        1.0 + np.cos(np.pi * (radius[transition] - inner) / edge)
    )
    return mask


def _write_mrc(path, volume, pixel):
    with mrcfile.new(path, overwrite=False) as handle:
        handle.set_data(np.ascontiguousarray(volume, dtype=np.float32))
        handle.voxel_size = pixel
        handle.header.mapc, handle.header.mapr, handle.header.maps = 1, 2, 3
        handle.header.nxstart = handle.header.nystart = handle.header.nzstart = 0
        handle.header.origin.x = handle.header.origin.y = handle.header.origin.z = 0
        handle.update_header_stats()


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def prepare_refinement_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-map", required=True, type=Path)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--particles", "--particle-star", dest="particle_star",
                        required=True, type=Path)
    parser.add_argument("--datadir", "--data-folder", dest="data_folder", type=Path)
    parser.add_argument("--strip-prefix", "--stack-prefix", dest="stack_prefix")
    parser.add_argument("--outdir", "-o", "--output-dir", dest="output_dir",
                        required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--native-side", required=True, type=_positive_int)
    parser.add_argument(
        "--native-pixel-size", dest="native_pixel", required=True,
        type=_positive_float,
    )
    parser.add_argument(
        "--expected-particles", type=_positive_int,
        help="expected STAR row count; inferred when omitted",
    )
    parser.add_argument("--particle-diameter", required=True, type=_positive_float)
    args = parser.parse_args(argv)

    dataset = _canonical_dataset(args.dataset)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("--outdir must be new or empty")
    if args.native_side % 2:
        parser.error("--native-side must be even")

    (requested_config_path, authoritative_config_path, map_path, config,
     source, source_voxel, source_side) = _load_source(
         args.source_config,
         args.source_map,
         dataset,
         args.native_side,
         args.native_pixel,
     )
    particles = _validate_particles(
        args.particle_star,
        args.data_folder,
        args.stack_prefix,
        args.native_side,
        args.native_pixel,
        args.expected_particles,
    )
    native, imaginary_error = _fourier_resample_real(source, args.native_side)
    restored, _ = _fourier_resample_real(native, source_side)
    roundtrip_error = float(
        np.linalg.norm(restored - source)
        / max(float(np.linalg.norm(source)), np.finfo(float).tiny)
    )
    mean_error = float(
        abs(native.mean() - source.mean())
        / max(abs(float(source.mean())), float(np.linalg.norm(source)) / source.size,
              np.finfo(float).tiny)
    )
    if roundtrip_error > 2e-11 or mean_error > 2e-11:
        raise FloatingPointError(
            "Fourier-resampling validation failed: "
            f"roundtrip={roundtrip_error:.3e}, mean={mean_error:.3e}"
        )
    # The initializer handed to RELION is the Fourier-resampled MoM map itself:
    # no support mask and no pre-filter. RELION's --ini_high low-passes it.

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-relion-", dir=output) as temporary:
        staging = Path(temporary)
        star_report = _write_relion_star(
            particles,
            staging / "particles.star",
            args.native_side,
            args.native_pixel,
        )
        star_report["path"] = str((output / "particles.star").resolve())
        _write_mrc(staging / "initial_model.mrc", native, args.native_pixel)
        provenance = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "source": {
                "config_argument": str(requested_config_path),
                "config_argument_sha256": _sha256(requested_config_path),
                "authoritative_summary": str(authoritative_config_path),
                "authoritative_summary_sha256": _sha256(
                    authoritative_config_path),
                "map": str(map_path),
                "map_sha256": _sha256(map_path),
                "order": 3,
                "n_initializations": config["n_initializations"],
                "side": source_side,
                "pixel_size_angstrom": source_voxel.tolist(),
                "field_of_view_angstrom": source_side * float(source_voxel[0]),
                "fit_complete": config.get("fit_complete"),
                "complete": config.get("complete"),
            },
            "native_geometry": {
                "side": args.native_side,
                "pixel_size_angstrom": args.native_pixel,
                "field_of_view_angstrom": args.native_side * args.native_pixel,
            },
            "fourier_resampling": {
                "method": "centered separable scipy.signal.resample",
                "lowpass_applied": False,
                "amplitude_rescaling": False,
                "relative_roundtrip_error": roundtrip_error,
                "relative_mean_error": mean_error,
                "relative_imaginary_norm": imaginary_error,
            },
            "initializer": {
                "support_mask_applied": False,
                "lowpass_applied": False,
                "note": "RELION --ini_high, when given, low-passes the reference at start-up; none by default",
                "particle_diameter_angstrom": args.particle_diameter,
            },
            "particles": {
                "source_star": str(particles.star_path),
                "source_star_sha256": _sha256(particles.star_path),
                "source_fingerprint": particles.fingerprint(),
                "data_folder": str(
                    particles.star_path.parent
                    if args.data_folder is None
                    else Path(args.data_folder).resolve(strict=True)
                ),
                "stack_prefix": args.stack_prefix,
                "n_particles": particles.n,
                "n_stacks": len(particles.stack_paths),
                "native_side": particles.res,
                "native_pixel_size_angstrom": particles.pixel_size,
                "metadata_pixel_size_angstrom": particles.metadata_pixel_size,
                "all_stack_headers_and_indices_validated": True,
                "ctf_metadata_validated": True,
            },
            "output_star": star_report,
            "output_files": {
                "initial_model": "initial_model.mrc",
                "particles": "particles.star",
            },
            "relion_initialization": {
                "poses_inherited": False,
                "origins_inherited": False,
                "random_subsets_inherited": False,
                "ctf_metadata_preserved": True,
                "coordinate_metadata_preserved_when_present": True,
            },
            "implementation": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": _sha256(__file__),
            },
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
    if not re.fullmatch(r"[A-Z][A-Z0-9]*", symmetry):
        raise ValueError("RELION symmetry must look like C1, D2, O, or I1")
    return symmetry


def _shell_literal(value) -> str:
    text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    if "\n" in text or "\r" in text:
        raise ValueError("shell-script values cannot contain newlines")
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
    if not header:
        raise RuntimeError(
            "the submitting SLURM script has no #SBATCH header to copy")
    header.extend(f"#SBATCH {name}={value}"
                  for name, value in overrides.items())
    return header


def _write_refinement_script(preparation, dataset, parameters):
    """Write a launcher that consumes completed preparation and runs RELION."""
    root = Path(__file__).resolve().parents[2]
    preparation = Path(preparation).resolve()
    result = preparation.parent / "relion_refine"
    script = preparation / "run_refinement.sh"
    job_suffix = dataset.split("-", 1)[1]
    diameter = parameters["particle_diameter_angstrom"]
    ini_high = parameters["ini_high_angstrom"]
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
        f'sha256sum "$ROOT/src/kam/command_line.py" '
        f'{_shell_literal(Path(__file__).resolve())} > "$JOB/code_sha256.txt"',
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
        ("# The initializer is the unmasked, unfiltered MoM map; RELION applies no initial low-pass."
         if ini_high is None else
         f"# The initializer is the unmasked, unfiltered MoM map; ini_high {ini_high:g} A low-passes it."),
        ("# Without --ini_high RELION starts from the resolution of the reference itself."
         if ini_high is None else
         "# ini_high also controls RELION's initial working resolution and CC iteration."),
        "command=(",
        '    srun --export=ALL --mpi=pmix_v3 --kill-on-bad-exit=1 -n 3 -c "$OMP_NUM_THREADS"',
        (f"    --gres={gres} --cpu-bind=cores --gpu-bind=none"
         if gres else "    --cpu-bind=cores --gpu-bind=none"),
        f"    bash -c {shlex.quote(rank_launcher)} relion_rank",
        '    --i "$JOB/inputs/particles.star" --ref "$JOB/inputs/initial_model.mrc"',
        '    --o "$JOB/run" --auto_refine --split_random_halves',
        ("    --firstiter_cc --trust_ref_size" if ini_high is None
         else f"    --firstiter_cc --trust_ref_size --ini_high {ini_high:.17g}"),
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
        "gpu_mapping": "automatic_per_rank_visible_devices",
        "gpu_preflight": "per-rank CUDA runtime enumeration and context initialization",
        "requested_gpus": gres,
        "slurm_submission_waits": True,
        "inputs_copied_to_job": True,
        **parameters,
    }


def relion_prepare_main(argv=None):
    """Prepare a completed MoM initializer and an independent RELION launcher."""
    parser = argparse.ArgumentParser(
        prog="kam relion_prepare",
        description=(
            "Prepare RELION input from kam_pipeline/MoM/M3 and generate the "
            "Della-specific run_refinement.sh used by these experiments."),
        allow_abbrev=False)
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--particles", "--particle-star", dest="particle_star", type=Path,
                        help="override the raw STAR used by pipeline")
    parser.add_argument("--datadir", "--data-folder", dest="data_folder", type=Path)
    parser.add_argument("--strip-prefix", "--stack-prefix", dest="stack_prefix")
    parser.add_argument("--native-pixel-size", type=_positive_float)
    parser.add_argument("--expected-particles", type=_positive_int)
    parser.add_argument("--particle-diameter", required=True, type=_positive_float)
    parser.add_argument("--symmetry", default="C1")
    parser.add_argument("--ini-high", type=_positive_float, default=None,
                        help="optional RELION --ini_high initial low-pass in A (default: none, no low-pass)")
    parser.add_argument("--offset-range", type=_positive_float, default=40.0)
    parser.add_argument("--offset-step", type=_positive_float, default=4.0)
    parser.add_argument("--low-resol-join-halves", type=_positive_float, default=40.0)
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
    configuration = manifest.get("configuration", manifest)
    dataset = _canonical_dataset(configuration["dataset_label"])
    dataset_number = dataset.split("-", 1)[1]
    run_name = pipeline.parent.name
    slurm_run = re.fullmatch(
        rf"{re.escape(dataset_number)}_[0-9]+", run_name)
    local_run = re.fullmatch(
        rf"{re.escape(dataset_number)}_local_[0-9]{{8}}T[0-9]{{12}}Z_[0-9]+",
        run_name,
    )
    if pipeline.name != "kam_pipeline" or not (slurm_run or local_run):
        raise ValueError(
            "--pipeline-dir must have layout "
            f"{dataset_number}_<SLURM_JOB_ID>/kam_pipeline")
    inputs = configuration["inputs"]
    source_config = pipeline / "MoM/M3/config.json"
    source_map = pipeline / "MoM/M3/recon.mrc"
    source = json.loads(source_config.read_text())
    native_side = int(source["native_side"])
    native_pixel = float(source["native_pixel_size_angstrom"])
    if args.native_pixel_size is not None:
        native_pixel = args.native_pixel_size
    denoise_manifest = pipeline / "denoise/manifest.npz"
    with np.load(denoise_manifest, allow_pickle=False) as saved:
        denoise_side = int(np.asarray(saved["native_res"]).item())
        denoise_pixel = float(np.asarray(saved["vox_native"]).item())
    if native_side != denoise_side or not math.isclose(
            native_pixel, denoise_pixel, rel_tol=2e-7, abs_tol=0.0):
        raise ValueError("native particle geometry differs from the denoising manifest")
    source_details = _load_source(
        source_config, source_map, dataset, native_side, native_pixel)
    del source_details
    particle_star = (args.particle_star if args.particle_star is not None
                     else Path(inputs["star_path"]))
    particle_star = particle_star.expanduser().resolve(strict=True)
    data_folder = args.data_folder
    stack_prefix = args.stack_prefix
    if args.particle_star is None:
        if data_folder is None and inputs.get("data_folder") is not None:
            data_folder = Path(inputs["data_folder"])
        if stack_prefix is None:
            stack_prefix = inputs.get("stack_prefix")
    if data_folder is not None:
        data_folder = data_folder.expanduser().resolve(strict=True)
    particles = _validate_particles(
        particle_star, data_folder, stack_prefix, native_side,
        native_pixel, args.expected_particles)
    if args.particle_diameter >= native_side * native_pixel:
        raise ValueError("particle diameter does not fit inside the box")
    if args.ini_high is not None and args.ini_high < 2.0 * native_pixel:
        raise ValueError("--ini-high must be at least the native Nyquist resolution")
    parameters = {
        "particle_diameter_angstrom": args.particle_diameter,
        "symmetry": _safe_symmetry(args.symmetry),
        "ini_high_angstrom": args.ini_high,
        "offset_range_pixels": args.offset_range,
        "offset_step_pixels": args.offset_step,
        "low_resol_join_halves_angstrom": args.low_resol_join_halves,
        "seed": args.seed,
    }
    preparation = pipeline.parent / "relion_prepare"
    request = {
        "schema_version": 1,
        "relion_driver_sha256": _sha256(__file__),
        "pipeline_dir": str(pipeline),
        "pipeline_manifest_sha256": _sha256(manifest_path),
        "source_map_sha256": _sha256(source_map),
        "source_config_sha256": _sha256(source_config),
        "source_summary_sha256": _sha256(source_config.with_name("summary.json")),
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
    validation_done = time.perf_counter()
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
        hashes = previous.get("prepared_file_sha256", {})
        required = {"particles.star", "initial_model.mrc", "run_refinement.sh"}
        if set(hashes) != required:
            raise ValueError("existing preparation lacks complete output checksums")
        for name, expected_hash in hashes.items():
            if _sha256(preparation / name) != expected_hash:
                raise ValueError(f"prepared file changed: {preparation / name}")
        print(f"Reusing validated RELION preparation: {preparation}", flush=True)
        print(f"Run: bash {shlex.quote(str(preparation / 'run_refinement.sh'))}", flush=True)
        return 0

    command = [
        "--source-map", str(source_map), "--source-config", str(source_config),
        "--particles", str(particle_star), "--outdir", str(preparation),
        "--dataset", dataset, "--native-side", str(native_side),
        "--native-pixel-size", str(native_pixel),
        "--expected-particles", str(particles.n),
        "--particle-diameter", str(args.particle_diameter),
    ]
    if data_folder is not None:
        command.extend(["--datadir", str(data_folder)])
    if stack_prefix is not None:
        command.extend(["--strip-prefix", stack_prefix])
    provenance = prepare_refinement_main(command)
    inputs_done = time.perf_counter()
    refinement = _write_refinement_script(preparation, dataset, parameters)
    provenance.update(
        preparation_request=request,
        refinement=refinement,
        complete=True,
        prepared_file_sha256={
            name: _sha256(preparation / name)
            for name in ("particles.star", "initial_model.mrc", "run_refinement.sh")},
    )
    completed = time.perf_counter()
    provenance["timings_seconds"] = {
        "input_validation": validation_done - started,
        "native_map_and_star_preparation": inputs_done - validation_done,
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


if __name__ == "__main__":
    raise SystemExit(relion_prepare_main())


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


def solvent_mask_from_map(volume, pixel_size, molecular_weight=None,
                          diameter=None, lowpass=15.0, extend=3,
                          edge_angstrom=6.0):
    """Build a reference-free solvent mask from a reconstruction.

    The threshold is set so the enclosed volume matches the expected
    molecular volume rather than by eye, on a low-passed copy so that noise
    cannot define the support; the mask is then restricted to its largest
    component, extended, and given a cosine edge, because a hard edge
    convolves the Fourier shells and inflates the masked FSC. Without a
    molecular weight the mask falls back to the soft sphere already used for
    the refinement input.
    """
    from scipy import ndimage

    volume = np.asarray(volume, dtype=np.float64)
    side = volume.shape[0]
    if molecular_weight is None:
        if diameter is None:
            raise ValueError("give either molecular_weight or diameter")
        return _soft_spherical_support(
            side, pixel_size, float(diameter), edge_angstrom), dict(
                kind="soft sphere", diameter_angstrom=float(diameter),
                cosine_edge_angstrom=float(edge_angstrom))
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
        kind="molecular-volume threshold on a low-passed map",
        lowpass_angstrom=float(lowpass),
        molecular_weight_daltons=float(molecular_weight),
        volume_per_dalton_cubic_angstrom=_POSTPROCESS_PROTEIN_VOLUME_PER_DALTON,
        volume_safety_factor=_POSTPROCESS_MASK_SAFETY,
        threshold=threshold,
        core_voxel_fraction=float(core.mean()),
        mask_weight_fraction=float(mask.mean()),
        extend_voxels=int(extend),
        cosine_edge_angstrom=float(edge_angstrom),
        largest_component_only=True,
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
    parser.add_argument("--molecular-weight", type=float,
                        help="complex mass in daltons; sets the mask threshold")
    parser.add_argument("--particle-diameter", type=float,
                        help="fallback soft-sphere diameter in Angstrom")
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
        refined, pixel_size, molecular_weight=args.molecular_weight,
        diameter=args.particle_diameter, lowpass=args.mask_lowpass,
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
        mask_reference_free=True,
        mask=mask_report,
        pixel_size_angstrom=pixel_size,
        postprocessed_map=str(output / "postprocess.mrc"),
        phase_randomization_correction="applied by relion_postprocess",
        note=("masked, phase-randomization-corrected gold-standard FSC; "
              "comparable with published resolutions"),
    )
    (output / "postprocess_report.json").write_text(
        json.dumps(report, indent=2))
    print(f"masked FSC resolution: "
          f"{report['masked_fsc_resolution_angstrom']:.2f} A "
          f"(B = {report['bfactor_used_for_sharpening']:.1f} A^2)", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0
