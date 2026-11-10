"""Helpers shared by the modules: run log, hashes and provenance, JSON and MRC output, image display, centered FFTs and the compute device."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


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
    """Copy everything written to stdout and stderr, subprocess output included, to the file ``path`` while it still reaches the terminal."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    log = open(path, "ab", buffering=0)
    lock = threading.Lock()
    streams = sys.stdout, sys.stderr
    for stream in streams:
        stream.flush()

    def tee(read_fd, terminal_fd):
        while data := os.read(read_fd, 65536):
            os.write(terminal_fd, data)
            with lock:
                log.write(data)
        os.close(read_fd)

    saved, threads = {}, []
    for fd in (1, 2):
        saved[fd] = os.dup(fd)
        read_fd, write_fd = os.pipe()
        threads.append(threading.Thread(target=tee, args=(read_fd, saved[fd]), daemon=True))
        threads[-1].start()
        os.dup2(write_fd, fd)
        os.close(write_fd)
    sys.stdout = open(1, "w", buffering=1, closefd=False)
    sys.stderr = open(2, "w", buffering=1, closefd=False)
    try:
        yield path
    finally:      # restore the terminal; the pipes then reach end-of-file and the copying threads finish
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = streams
        for fd in (1, 2):
            os.dup2(saved[fd], fd)
        for thread in threads:
            thread.join()
        for fd in saved.values():
            os.close(fd)
        log.close()


def show_particle_image(axis, image):
    """Display a particle with per-image 1--99% grayscale limits."""
    lo, hi = np.percentile(image, (1, 99))
    return axis.imshow(image, cmap="gray", vmin=lo, vmax=hi,
                       interpolation="nearest", origin="upper")


def save_mrc(vol, path, voxel_size):
    """Write a real volume to an MRC file (float32, isotropic voxel size)."""
    import mrcfile
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(vol, dtype=np.float32))
        m.voxel_size = voxel_size


def centered_fft2(img):
    """Centered two-dimensional forward FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))


def centered_ifft2(fimg):
    """Centered 2-D inverse FFT (inverse of :func:`centered_fft2`)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fimg, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))


def centered_ifftn(fvol):
    """Centered n-D inverse FFT."""
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fvol)))


def centered_coords_1d(L):
    """Integer frequency coordinate at each array index of a centered grid: index ``a`` holds ``a - L//2``."""
    return np.arange(L) - (L // 2)


# ===========================================================================
# Compute device
# ===========================================================================

_DEVICE = "cpu"
_TORCH = None


def use_device(name):
    """Select the device ('cpu' or 'gpu') of the moment and projection kernels; 'gpu' needs PyTorch with CUDA."""
    global _DEVICE
    _DEVICE = name
    return _DEVICE


def on_gpu():
    return _DEVICE == "gpu"


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
