"""Logging/plotting utilities and the CUDA moment kernels.

The gpu_* kernels accept and return NumPy arrays. Torch computes the
moment matrix products, and the volume projections (SphFB matrices or
voxel cuFINUFFT plans) use the same CUDA device.
"""

import atexit
from contextlib import contextmanager
import os
from pathlib import Path
import select
import sys
import threading
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class _Tee:
    """Write-through stream that also records into a file handle."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, text):
        self._stream.write(text)
        keep = text.split("\r")[-1]
        if keep:
            self._fh.write(keep)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def log_to(path, argv=None):
    """Mirror stdout/stderr into an appended, timestamped log file."""
    fh = open(path, "a", buffering=1)
    fh.write("\n" + "=" * 70 + "\n")
    fh.write(f"run {datetime.now():%Y-%m-%d %H:%M:%S}"
             + (f"  |  {' '.join(argv)}" if argv else "") + "\n")
    fh.write("=" * 70 + "\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)

    def _close():
        sys.stdout = getattr(sys.stdout, "_stream", sys.stdout)
        sys.stderr = getattr(sys.stderr, "_stream", sys.stderr)
        fh.close()

    atexit.register(_close)
    return path


class _WriteThroughStream:
    """Flush Python writes immediately while preserving stream attributes."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        result = self._stream.write(text)
        self._stream.flush()
        return result

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self._stream, name)


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
            if written <= 0:
                raise OSError("log output write made no progress")
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
    if (image.ndim != 2 or image.size == 0 or not np.isrealobj(image)
            or not np.all(np.isfinite(image))):
        raise ValueError("particle image must be a finite real 2-D array")
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        width = max(1.0, abs(lo)) * 0.5
        lo, hi = lo - width, hi + width
    return axis.imshow(image, cmap="gray", vmin=lo, vmax=hi,
                       interpolation="nearest", origin="upper")


def plot_contrast_histogram(contrasts, path, bins=80):
    """Save the positive per-image contrast histogram as a PDF."""
    path = Path(path).with_suffix(".pdf")
    contrasts = np.asarray(contrasts, dtype=float)
    if (contrasts.ndim != 1 or contrasts.size == 0
            or not np.all(np.isfinite(contrasts))
            or np.any(contrasts <= 0)):
        raise ValueError("contrasts must be a nonempty vector of positive values")
    mean = float(np.mean(contrasts))
    std = float(np.std(contrasts))
    second = float(np.mean(contrasts**2))
    third = float(np.mean(contrasts**3))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(contrasts, bins=bins, density=True, color="#4c78a8",
            alpha=0.85, edgecolor="white", linewidth=0.25)
    ax.axvline(1.0, color="#e45756", ls="--", lw=1.5,
               label="target mean = 1")
    ax.set_xlabel("per-image contrast")
    ax.set_ylabel("density")
    ax.set_title("Simulated image contrast")
    ax.text(0.98, 0.96,
            f"mean={mean:.4f}\nstd={std:.4f}\nE[c²]={second:.4f}\nE[c³]={third:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                      edgecolor="0.8"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_fsc(k, curve, path, voxel_size, L, title=None):
    """Save an FSC curve as a PDF, marking the cutoff crossings.

    ``k`` is the shell radius in cycles per box, so frequency is ``k / (L * voxel)``
    in 1/Angstrom and the corresponding resolution is its reciprocal.
    """
    path = Path(path).with_suffix(".pdf")
    from .postprocessing import fsc_resolution           # local: avoids an import cycle

    freq = k / (L * voxel_size)                    # 1/Angstrom
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(freq, curve, lw=2, color="#1f77b4")
    ax.axhline(0.0, color="0.7", lw=0.8)
    res = fsc_resolution(k, curve, voxel_size, L)
    ax.axhline(0.5, color="0.5", ls="--", lw=1)
    if res:
        ax.axvline(1.0 / res, color="0.5", ls="--", lw=1)
    ax.annotate("FSC=0.5: " + (f"{res:.1f} $\\AA$" if res else "no measured crossing"),
                (freq[-1], 0.5), ha="right", va="bottom", fontsize=9, color="0.3")
    ax.set_xlim(0, freq[-1])
    ax.set_ylim(min(-0.05, float(np.nanmin(curve)) - 0.05), 1.02)
    ax.set_xlabel(r"spatial frequency (1/$\AA$)")
    ax.set_ylabel("FSC")
    ax.set_title(title or "FSC: reconstruction (aligned) vs. downsampled ground truth",
                 fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fsc_stages(k, curves, labels, path, voxel_size, L):
    """Save overlaid FSC curves from successive fitting stages as a PDF."""
    path = Path(path).with_suffix(".pdf")
    from .postprocessing import fsc_resolution           # local: avoids an import cycle

    freq = k / (L * voxel_size)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for curve, lab, col in zip(curves, labels,
                               ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")):
        res = fsc_resolution(k, curve, voxel_size, L)
        tag = f"{lab}: " + (f"{res:.1f} $\\AA$" if res else "no measured crossing")
        ax.plot(freq, curve, lw=2, color=col, label=tag)
    ax.axhline(0.5, color="0.5", ls="--", lw=1)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlim(0, freq[-1])
    ax.set_ylim(min(-0.05, float(min(np.nanmin(c) for c in curves)) - 0.05), 1.02)
    ax.set_xlabel(r"spatial frequency (1/$\AA$)")
    ax.set_ylabel("FSC")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_projections(raw, ds, path, raw_res, ds_res):
    """Save matched detector-grid and downsampled projections as a PDF."""
    path = Path(path).with_suffix(".pdf")
    n = raw.shape[0]
    fig, ax = plt.subplots(2, n, figsize=(2.2 * n, 5.0), squeeze=False)
    for j in range(n):
        show_particle_image(ax[0, j], raw[j])
        show_particle_image(ax[1, j], ds[j])
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(
        f"detector images ({raw_res}x{raw_res}, top) and moment input "
        f"({ds_res}x{ds_res}, bottom)", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# CUDA backend shared by projection and alignment operators
# ===========================================================================

_TORCH = None


def gpu_available():
    """True if PyTorch can use a CUDA device."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
