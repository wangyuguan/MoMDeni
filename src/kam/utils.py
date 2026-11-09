"""Run-log capture, particle display, MRC volume I/O and the CUDA device helpers."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import select
import subprocess
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class _WriteThroughStream:
    """Flush Python writes immediately while preserving stream attributes."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        result = self._stream.write(text)
        self._stream.flush()
        return result

    def __getattr__(self, name):
        return getattr(self._stream, name)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path):
    """HEAD commit of the git checkout at ``path``; "unknown" outside a checkout (the value is stored in npz archives, which cannot hold None)."""
    done = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else "unknown"


@contextmanager
def capture_run_log(path):
    """Append stdout/stderr to ``path`` while retaining live terminal output.

    Capture operates on file descriptors, including inherited subprocess and
    native writes. Python streams flush each write. Use this process-wide
    context from the main thread, and wait for child processes before leaving
    it. Separate stdout/stderr streams retain their own ordering; simultaneous
    writes from different streams have no defined ordering in the merged log.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    streams = (sys.stdout, sys.stderr)
    saved_fds = {}
    redirected = []
    threads = []
    stop = threading.Event()
    log_lock = threading.Lock()
    errors = []

    def write_all(fd, data):
        remaining = memoryview(data)
        while remaining:
            written = os.write(fd, remaining)
            remaining = remaining[written:]

    def drain(read_fd, terminal_fd):
        terminal_active = True
        log_active = True
        pending = bytearray()

        def append_log(data):
            nonlocal log_active
            if log_active:
                try:
                    with log_lock:
                        write_all(log_fd, data)
                except OSError as error:
                    errors.append(error)
                    log_active = False

        try:
            while True:
                ready, _, _ = select.select([read_fd], [], [], 0.1)
                if not ready:
                    if pending:
                        append_log(pending)
                        pending.clear()
                    if stop.is_set():
                        break
                    continue
                data = os.read(read_fd, 65536)
                if not data:
                    break
                if log_active:
                    pending.extend(data)
                    line_end = max(pending.rfind(b"\n"), pending.rfind(b"\r")) + 1
                    if line_end:
                        append_log(pending[:line_end])
                        del pending[:line_end]
                    elif len(pending) >= 65536:
                        append_log(pending)
                        pending.clear()
                if terminal_active:
                    try:
                        write_all(terminal_fd, data)
                    except OSError as error:
                        errors.append(error)
                        terminal_active = False
        except OSError as error:
            errors.append(error)
        finally:
            if pending:
                append_log(pending)
            os.close(read_fd)

    try:
        for stream in streams:
            stream.flush()
        for fd in (1, 2):
            saved_fds[fd] = os.dup(fd)
            read_fd, write_fd = os.pipe()
            worker = threading.Thread(
                target=drain, args=(read_fd, saved_fds[fd]), daemon=True,
                name=f"run-log-{fd}")
            try:
                worker.start()
            except BaseException:
                os.close(read_fd)
                os.close(write_fd)
                raise
            threads.append(worker)
            try:
                os.dup2(write_fd, fd)
                redirected.append(fd)
            finally:
                os.close(write_fd)
        sys.stdout, sys.stderr = (_WriteThroughStream(s) for s in streams)
        yield path
    finally:
        active_exception = sys.exc_info()[0] is not None
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError) as error:
                errors.append(error)
        sys.stdout, sys.stderr = streams
        for fd in reversed(redirected):
            os.dup2(saved_fds[fd], fd)
        stop.set()
        for worker in threads:
            worker.join()
        for fd in saved_fds.values():
            os.close(fd)
        os.close(log_fd)
        if errors:
            message = f"run log encountered an output error: {errors[0]}"
            if active_exception:
                try:
                    os.write(2, (message + "\n").encode())
                except OSError:
                    pass
            else:
                raise OSError(message) from errors[0]


def show_particle_image(axis, image):
    """Display a particle with per-image 1--99% grayscale limits."""
    image = np.asarray(image)
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        width = max(1.0, abs(lo)) * 0.5
        lo, hi = lo - width, hi + width
    return axis.imshow(image, cmap="gray", vmin=lo, vmax=hi,
                       interpolation="nearest", origin="upper")


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
    # Convert MRC (section, row, column) storage to canonical (z, y, x) order.
    stored_physical_axes = (axis_mapping[2], axis_mapping[1], axis_mapping[0])
    permutation = tuple(stored_physical_axes.index(axis) for axis in (3, 2, 1))
    if permutation != (0, 1, 2):
        vol = np.ascontiguousarray(np.transpose(vol, permutation))
    if not np.allclose(voxel_xyz, voxel_xyz[0], rtol=1e-5, atol=1e-7):
        raise ValueError(
            "alignment requires isotropic MRC voxels, but the header for "
            f"{path} gives {voxel_xyz.tolist()} Angstrom")
    return vol, float(np.mean(voxel_xyz))


def save_mrc(vol, path, voxel_size):
    """Write a real volume to an MRC file (float32, isotropic voxel size)."""
    import mrcfile
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(vol, dtype=np.float32))
        m.voxel_size = voxel_size


def fourier_resample_real(volume, target_side):
    """Resample a centered real cube by separable Fourier interpolation; also returns the relative imaginary part that was dropped."""
    from scipy import signal
    volume = np.asarray(volume, dtype=np.float64)
    if volume.shape[0] == target_side:
        return np.array(volume, copy=True), 0.0
    result = np.fft.ifftshift(volume)
    for axis in range(3):
        result = signal.resample(result, target_side, axis=axis, window=None)
    result = np.fft.fftshift(result)
    imaginary_norm = float(np.linalg.norm(np.imag(result)))
    result = np.asarray(np.real(result), dtype=np.float64)
    return result, imaginary_norm / max(float(np.linalg.norm(result)), np.finfo(float).tiny)


# ===========================================================================
# CUDA device helpers
# ===========================================================================

_TORCH = None


def _gpu_torch():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


def _gpu_dev():
    """Return the first CUDA device visible to the process."""
    torch = _gpu_torch()
    torch.cuda.set_device(0)
    return torch.device("cuda")
