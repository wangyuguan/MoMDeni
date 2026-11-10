"""Steerable Fourier--Bessel (FB) moments of particle images under in-plane-uniform viewing, and the moment archive of a denoised stack.

An image is expanded as ``sum_{q,t} c[q,t] Phi_q,t(kappa) e^{i q phi}`` (``kappa`` in cycles per pixel, ``|q| <= L``), with radial functions
``Phi_q`` that span the central slices of the SphFB volume basis (:class:`FBMomentBasis`). Only the entries invariant under in-plane rotation
are kept:

    M1:  E c_0
    M2:  E c_q c_q^H                         q = 0..L
    M3:  E c_a (x) c_b (x) conj(c_{a+b})      0 <= a <= b, a + b <= L

Each block is weighted so that norms of the flat vectors equal Frobenius norms of the full tensors. :func:`prepare_moments` forms the
moments of the selected denoised images and saves them with the radial basis in ``moments.npz``, so that the fits use the same functions.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mrcfile
import numpy as np
from scipy.special import jv

from . import utils
from .utils import ROOT, git_commit, on_gpu, sha256_array, sha256_file, write_json
from .volume import SphFBParam

DEFAULT_MOMENT_RES = 64
# Relative singular-value cut for the per-q radial basis of the slice span.
RADIAL_SINGULAR_THRESHOLD = 1e-2


def _radial_quadrature(side):
    """Gauss--Legendre nodes and ``dkappa`` weights on ``[0, 1/2]``."""
    nodes, weights = np.polynomial.legendre.leggauss(2 * int(side) + 16)
    return 0.25 * (nodes + 1.0), 0.25 * weights


def _ray_columns(param, q):
    """SphFB coefficient indices with angular order ``m = q``."""
    return [param.indices[(ell, k, q)]
            for ell in range(abs(q), param.ell_max + 1)
            for k in range(param.k_max[ell])]


class FBMomentBasis:
    """Per-q radial basis, data/model coefficient maps, and moment layout.

    ``radial`` optionally supplies a stored ``(nodes, weights, basis, counts)``
    tuple from :meth:`radial_arrays` so that prepared moments and later
    reconstructions use bit-identical radial functions.
    """

    def __init__(self, side, ell_max, radial=None):
        self.side = int(side)
        self.ell_max = int(ell_max)
        if radial is None:
            param = SphFBParam(self.side, self.ell_max, np.eye(3)[None])
            self.nodes, self.weights = _radial_quadrature(self.side)
            sw = np.sqrt(self.nodes * self.weights)
            ray = self._ray(param)
            self.radial = []
            for q in range(self.ell_max + 1):
                A = sw[:, None] * ray[:, _ray_columns(param, q)]
                U, s, _ = np.linalg.svd(A, full_matrices=False)
                U = U[:, s > RADIAL_SINGULAR_THRESHOLD * s[0]]
                # Fix the SVD sign ambiguity deterministically.
                pivot = np.argmax(np.abs(U), axis=0)
                U *= np.sign(U[pivot, np.arange(U.shape[1])])
                self.radial.append(U)
        else:
            nodes, weights, packed, counts = (np.asarray(a) for a in radial)
            counts = counts.astype(int)
            self.nodes = nodes.astype(float)
            self.weights = weights.astype(float)
            edges = np.r_[0, np.cumsum(counts)]
            self.radial = [packed[:, edges[q]:edges[q + 1]].astype(float)
                           for q in range(self.ell_max + 1)]
        self.counts = np.array([U.shape[1] for U in self.radial], dtype=int)
        edges = np.r_[0, np.cumsum(self.counts)]
        self.slices = [slice(edges[q], edges[q + 1])
                       for q in range(self.ell_max + 1)]
        self.n_coefficients = int(edges[-1])
        self._layout()

    # ------------------------------------------------------------------ #
    # Radial basis and coefficient maps
    # ------------------------------------------------------------------ #
    def _ray(self, param):
        """SphFB basis on the ``phi = 0`` ray of the image plane (real)."""
        k = self.nodes
        ray = param._basis(k, np.zeros_like(k), np.zeros_like(k))
        return ray.real

    def radial_arrays(self):
        """Arrays that reproduce this radial basis via ``radial=``."""
        return (self.nodes.copy(), self.weights.copy(),
                np.concatenate(self.radial, axis=1), self.counts.copy())

    def fingerprint(self):
        """SHA-256 of the radial nodes, weights, and basis values."""
        h = hashlib.sha256()
        for array in self.radial_arrays():
            h.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        h.update(np.int64([self.side, self.ell_max]).tobytes())
        return h.hexdigest()

    def slice_projection(self, param):
        """Matrix ``G`` with model coefficients ``c = u @ G.T``.

        ``u`` holds the complex SphFB coefficients of a rotated volume, as in
        :meth:`volume.SphFBParam._rotated_coefficients`.
        """
        sw = np.sqrt(self.nodes * self.weights)
        ray = self._ray(param)
        G = np.zeros((self.n_coefficients, param.n_params), dtype=np.complex128)
        for q in range(self.ell_max + 1):
            cols = _ray_columns(param, q)
            G[self.slices[q], cols] = self.radial[q].T @ (sw[:, None] * ray[:, cols])
        return G

    def image_operator(self, image_side):
        """Complex ``(image_side**2, D)`` map from real images to coefficients.

        Images are flattened in C order over ``(row, column)`` and use the
        centered-FFT pixel origin ``(image_side//2, image_side//2)``. The
        image transform matches the central ``side x side`` crop of
        :func:`utils.centered_fft2`: model coordinate ``x`` is the column
        frequency and ``y`` the row frequency. The angular Fourier integral of
        the image DTFT is evaluated exactly with Bessel functions.
        """
        N = int(image_side)
        c = np.arange(N) - N // 2
        y, x = np.meshgrid(c, c, indexing="ij")
        rho, alpha = np.hypot(x, y).ravel(), np.arctan2(y, x).ravel()
        sw = np.sqrt(self.nodes * self.weights)
        z = 2 * np.pi * self.side / N * self.nodes[:, None] * rho[None, :]
        scale = (self.side / N) ** 2
        op = np.empty((N * N, self.n_coefficients), dtype=np.complex128)
        for q in range(self.ell_max + 1):
            radial = jv(q, z).T @ (sw[:, None] * self.radial[q])
            phase = scale * (-1j) ** q * np.exp(-1j * q * alpha)
            op[:, self.slices[q]] = phase[:, None] * radial
        return op

    # ------------------------------------------------------------------ #
    # Moment layout
    # ------------------------------------------------------------------ #
    def _layout(self):
        T = self.counts
        L = self.ell_max
        self.m2_blocks = []
        offset = 0
        for q in range(L + 1):
            size = int(T[q] * T[q])
            self.m2_blocks.append(dict(
                q=q, weight=1.0 if q == 0 else 2.0,
                slice=slice(offset, offset + size), shape=(T[q], T[q])))
            offset += size
        self.m2_size = offset
        self.m3_blocks = []
        offset = 0
        for a in range(L + 1):
            for b in range(a, L + 1 - a):
                c = a + b
                if a == 0 and b == 0:
                    weight = 1.0
                elif a == 0 or a == b:
                    weight = 6.0
                else:
                    weight = 12.0
                shape = (int(T[a]), int(T[b]), int(T[c]))
                size = int(np.prod(shape))
                self.m3_blocks.append(dict(
                    q=(a, b, c), weight=weight,
                    slice=slice(offset, offset + size), shape=shape))
                offset += size
        self.m3_size = offset
        self.shapes = ((int(T[0]),), (self.m2_size,), (self.m3_size,))

    def unique_m3_entries(self):
        """Entries of the complete M3 up to permutation and conjugation."""
        n = 0
        for block in self.m3_blocks:
            a, b, c = block["q"]
            Ta, Tb, Tc = block["shape"]
            if a == b == c == 0:
                n += Ta * (Ta + 1) * (Ta + 2) // 6
            elif a == b:
                n += Ta * (Ta + 1) // 2 * Tc
            elif a == 0:
                n += Ta * Tb * (Tb + 1) // 2
            else:
                n += Ta * Tb * Tc
        return n

    def summary(self):
        return dict(
            radial_singular_threshold=RADIAL_SINGULAR_THRESHOLD,
            side=self.side, ell_max=self.ell_max,
            radial_counts=self.counts.tolist(),
            n_coefficients=self.n_coefficients,
            n_radial_nodes=len(self.nodes),
            m1_entries=int(self.counts[0]),
            m2_entries=self.m2_size,
            m3_entries=self.m3_size,
            m3_blocks=[block["q"] for block in self.m3_blocks],
            m3_unique_entries=self.unique_m3_entries(),
            radial_basis_sha256=self.fingerprint(),
        )

    # ------------------------------------------------------------------ #
    # Moments and their gradients
    # ------------------------------------------------------------------ #
    def _check(self, C, w):
        C = np.asarray(C, dtype=np.complex128)
        w = np.asarray(w, dtype=float)
        return C, w

    def moments(self, C, w, orders=(1, 2, 3), batch_size=4096):
        """Weighted flat M1--M3 of coefficient rows ``C`` with masses ``w``.

        Orders not listed are returned as ``None``.
        """
        C, w = self._check(C, w)
        if on_gpu():
            return self._moments_gpu(C, w, orders, batch_size)
        m = [np.zeros(s, dtype=np.complex128) if k + 1 in orders else None
             for k, s in enumerate(self.shapes)]
        for start in range(0, len(C), batch_size):
            X = [C[start:start + batch_size, s] for s in self.slices]
            ww = w[start:start + batch_size]
            if m[0] is not None:
                m[0] += ww @ X[0]
            if m[1] is not None:
                for block in self.m2_blocks:
                    Y = X[block["q"]]
                    m[1][block["slice"]] += np.sqrt(block["weight"]) * (
                        (ww[:, None] * Y).T @ np.conj(Y)).ravel()
            if m[2] is not None:
                for block in self.m3_blocks:
                    a, b, c = block["q"]
                    m[2][block["slice"]] += np.sqrt(block["weight"]) * np.einsum(
                        "n,ni,nj,nk->ijk", ww, X[a], X[b], np.conj(X[c]),
                        optimize=True).ravel()
        return tuple(m)

    def moment_gradients(self, C, w, clean):
        """Coefficient and mass gradients of ``sum_k 2 Re vdot(clean_k, m_k)``.

        ``clean[k]`` is ``None`` for an inactive order, otherwise the flat
        derivative of the loss with respect to the flat model moment, halved
        (for a squared loss ``lambda |s m - target|^2`` this is
        ``s*lambda*(s m - target)``). Returns ``G_C, G_w`` with
        ``df = Re vdot(G_C, dC) + G_w @ dw``.
        """
        C, w = self._check(C, w)
        if on_gpu():
            return self._moment_gradients_gpu(C, w, clean)
        X = [C[:, s] for s in self.slices]
        G = [np.zeros_like(x) for x in X]
        gw = np.zeros(len(C))
        if clean[0] is not None:
            r = np.asarray(clean[0], dtype=np.complex128)
            G[0] += 2 * w[:, None] * r[None, :]
            gw += 2 * np.real(X[0] @ np.conj(r))
        if clean[1] is not None:
            r2 = np.asarray(clean[1], dtype=np.complex128)
            for block in self.m2_blocks:
                q = block["q"]
                R = np.sqrt(block["weight"]) * r2[block["slice"]].reshape(
                    block["shape"])
                H = R + R.conj().T
                G[q] += 2 * w[:, None] * (X[q] @ H.T)
                gw += 2 * np.real(np.einsum(
                    "ni,ij,nj->n", np.conj(X[q]), R, X[q], optimize=True))
        if clean[2] is not None:
            r3 = np.asarray(clean[2], dtype=np.complex128)
            for block in self.m3_blocks:
                a, b, c = block["q"]
                R = np.sqrt(block["weight"]) * r3[block["slice"]].reshape(
                    block["shape"])
                x, y, z = X[a], X[b], X[c]
                Rz = np.einsum("ijk,nk->nij", R, z, optimize=True)
                gx = np.einsum("nij,nj->ni", Rz, np.conj(y), optimize=True)
                gy = np.einsum("nij,ni->nj", Rz, np.conj(x), optimize=True)
                gz = np.einsum("ijk,ni,nj->nk", np.conj(R), x, y, optimize=True)
                G[a] += 2 * w[:, None] * gx
                G[b] += 2 * w[:, None] * gy
                G[c] += 2 * w[:, None] * gz
                gw += 2 * np.real(np.sum(gz * np.conj(z), axis=1))
        return np.concatenate(G, axis=1), gw

    # The same block sums and gradients as the two NumPy methods above, in complex128 on the GPU.
    def _moments_gpu(self, C, w, orders, batch_size):
        torch, dev = utils._gpu_torch(), utils._gpu_dev()
        m = [torch.zeros(s, dtype=torch.complex128, device=dev) if k + 1 in orders else None
             for k, s in enumerate(self.shapes)]
        for start in range(0, len(C), batch_size):
            Ct = torch.as_tensor(np.ascontiguousarray(C[start:start + batch_size]), device=dev)
            ww = torch.as_tensor(w[start:start + batch_size], device=dev).to(torch.complex128)
            X = [Ct[:, s] for s in self.slices]
            if m[0] is not None:
                m[0] += ww @ X[0]
            if m[1] is not None:
                for block in self.m2_blocks:
                    Y = X[block["q"]]
                    m[1][block["slice"]] += np.sqrt(block["weight"]) * (
                        (ww[:, None] * Y).T @ torch.conj(Y)).reshape(-1)
            if m[2] is not None:
                for block in self.m3_blocks:
                    a, b, c = block["q"]
                    P = (ww[:, None] * X[a])[:, :, None] * X[b][:, None, :]
                    m[2][block["slice"]] += np.sqrt(block["weight"]) * (
                        P.reshape(len(P), -1).T @ torch.conj(X[c])).reshape(-1)
        return tuple(None if x is None else x.cpu().numpy() for x in m)

    def _moment_gradients_gpu(self, C, w, clean):
        torch, dev = utils._gpu_torch(), utils._gpu_dev()
        Ct = torch.as_tensor(np.ascontiguousarray(C), device=dev)
        wc = torch.as_tensor(w, device=dev).to(torch.complex128)[:, None]
        X = [Ct[:, s] for s in self.slices]
        G = [torch.zeros_like(x) for x in X]
        gw = torch.zeros(len(C), dtype=torch.float64, device=dev)
        if clean[0] is not None:
            r = torch.as_tensor(np.asarray(clean[0], dtype=np.complex128), device=dev)
            G[0] += 2 * wc * r[None, :]
            gw += 2 * torch.real(X[0] @ torch.conj(r))
        if clean[1] is not None:
            r2 = torch.as_tensor(np.asarray(clean[1], dtype=np.complex128), device=dev)
            for block in self.m2_blocks:
                q = block["q"]
                R = np.sqrt(block["weight"]) * r2[block["slice"]].reshape(block["shape"])
                H = R + R.conj().T
                G[q] += 2 * wc * (X[q] @ H.T)
                gw += 2 * torch.real(((torch.conj(X[q]) @ R) * X[q]).sum(1))
        if clean[2] is not None:
            r3 = torch.as_tensor(np.asarray(clean[2], dtype=np.complex128), device=dev)
            for block in self.m3_blocks:
                a, b, c = block["q"]
                R = np.sqrt(block["weight"]) * r3[block["slice"]].reshape(block["shape"])
                x, y, z = X[a], X[b], X[c]
                Rz = torch.einsum("ijk,nk->nij", R, z)
                gx = torch.einsum("nij,nj->ni", Rz, torch.conj(y))
                gy = torch.einsum("nij,ni->nj", Rz, torch.conj(x))
                gz = torch.einsum("nij,ijk->nk", x[:, :, None] * y[:, None, :], torch.conj(R))
                G[a] += 2 * wc * gx
                G[b] += 2 * wc * gy
                G[c] += 2 * wc * gz
                gw += 2 * torch.real((gz * torch.conj(z)).sum(1))
        return torch.cat(G, dim=1).cpu().numpy(), gw.cpu().numpy()


def empirical_moments(images, moment_res, ell_max, batch_size=256):
    """Uniformly weighted FB M1--M3 of a real denoised image stack; returns ``(basis, (m1, m2, m3))``."""
    images = np.asarray(images)
    basis = FBMomentBasis(moment_res, ell_max)
    operator = basis.image_operator(images.shape[1])
    n = len(images)
    w = np.full(n, 1.0 / n)
    sums = [np.zeros(s, dtype=np.complex128) for s in basis.shapes]
    for start in range(0, n, batch_size):
        batch = np.asarray(images[start:start + batch_size], dtype=np.float64)
        C = batch.reshape(len(batch), -1) @ operator
        for total, part in zip(sums, basis.moments(C, w[start:start + len(batch)])):
            total += part
    return basis, tuple(sums)


# ===========================================================================
# Moment archive (prepared/moments.npz and report.json)
# ===========================================================================

@dataclass(frozen=True)
class MomentPreparationConfig:
    """Configuration for :func:`prepare_moments`."""

    denoising_dir: Path
    selection_file: Path
    output_dir: Path
    n_images: int
    denoise_res: int
    moment_res: int
    batch_size: int
    device: str
    fb_ell_max: int


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


def _scalar(saved, key):
    return np.asarray(saved[key]).reshape(()).item()


def read_denoising_output(directory):
    """The files of a denoising output and the manifest entries the moment archive records."""

    directory = Path(directory).resolve(strict=True)
    manifest_path = directory / "manifest.npz"
    with np.load(manifest_path, allow_pickle=False) as saved:
        manifest = {key: _scalar(saved, key) for key in ("star_path", "data_folder", "stack_prefix", "source_fingerprint")}
        manifest.update(
            dataset=str(_scalar(saved, "dataset")),
            native_res=int(_scalar(saved, "native_res")),
            native_pixel=float(_scalar(saved, "vox_native")),
            denoise_res=int(_scalar(saved, "denoise_res")),
            denoise_pixel=float(_scalar(saved, "vox_denoise")),
            contrast_est=np.asarray(saved["contrast_est"], dtype=np.float64),
        )
        stack_path = Path(str(_scalar(saved, "denoised_stack")))
    return directory, stack_path, manifest_path, directory / "report.json", manifest


def load_selection(selection_file):
    """The selected positions, source rows and image names of a selection file."""

    selection_file = Path(selection_file).resolve(strict=True)
    with np.load(selection_file, allow_pickle=False) as saved:
        positions = np.asarray(saved["selected_positions"], dtype=np.int64)
        rows = np.asarray(saved["selected_source_rows"], dtype=np.int64)
        names = np.asarray(saved["selected_image_names"])
    return selection_file, positions, rows, names


def _save_moments(path, images, moment_res, ell_max, batch_size, **fields):
    """Form the in-plane-invariant FB M1--M3 of ``images`` and save them with ``fields`` and the radial basis to ``path``.

    Mirroring an image conjugates its stored q>=0 FB coefficients, so averaging every image with its mirror leaves exactly the real parts of
    the invariant moments; only the real parts are kept, and no images are duplicated. Returns the basis, the moments and their metadata."""

    basis, moments = empirical_moments(images, moment_res, ell_max, batch_size=batch_size)
    m1, m2, m3 = (np.ascontiguousarray(moment.real).astype(np.complex128) for moment in moments)
    nodes, weights, radial, counts = basis.radial_arrays()
    np.savez(path, mbar1=m1, mbar2=m2, mbar3=m3, ds_res=moment_res, fb_ell_max=basis.ell_max, fb_radial_nodes=nodes,
             fb_radial_weights=weights, fb_radial_basis=radial, fb_radial_counts=counts, **fields)
    metadata = dict(n_images=len(images), denoise_res=int(np.shape(images)[1]), moment_res=moment_res, fb=basis.summary(),
                    batch_size=batch_size)
    return basis, (m1, m2, m3), metadata


def prepare_moments_from_images(images, pixel_size, output, *, moment_res, ell_max, batch_size=256, dataset="", source=""):
    """Write ``moments.npz`` and ``report.json`` (the files of :func:`prepare_moments`) for a stack of denoised images of pixel ``pixel_size``."""

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    side = int(np.shape(images)[-1])
    pixel = pixel_size * side / moment_res
    moments_path = output / "moments.npz"
    basis, (m1, m2, m3), metadata = _save_moments(
        moments_path, images, moment_res, ell_max, batch_size, n_images=len(images), dataset=dataset, pixel_size=pixel,
        native_side=side, native_pixel_size=pixel_size)
    write_json(output / "report.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), dataset=dataset, denoised_stack=str(source),
        n_images=len(images), image_side=side, image_pixel_size_angstrom=pixel_size, moment_res=moment_res,
        pixel_size_angstrom=pixel, fb_ell_max=ell_max, n_frequencies=basis.n_coefficients,
        n_m2_entries=int(m2.size), n_canonical_m3_entries=len(m3), metadata=metadata,
        prepared_file=str(moments_path), prepared_sha256=sha256_file(moments_path), kam_cryoem_commit=git_commit(ROOT),
        elapsed_seconds=time.monotonic() - started))
    return output


def prepare_moments(config: MomentPreparationConfig):
    """Prepare M1--M3 from the selected images of a denoising output."""

    output = config.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new or empty moment preparation directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timings = {}

    utils.use_device(config.device)
    denoising_dir, stack_path, manifest_path, denoising_report, manifest = read_denoising_output(config.denoising_dir)
    selection_path, positions, rows, names = load_selection(config.selection_file)
    timings["input_loading"] = time.monotonic() - started

    phase_started = time.monotonic()
    with mrcfile.mmap(stack_path, mode="r", permissive=False) as handle:
        selected = np.asarray(handle.data[positions], dtype=np.float32)
    timings["selected_image_loading"] = time.monotonic() - phase_started

    phase_started = time.monotonic()
    pixel = manifest["native_res"] * manifest["native_pixel"] / config.moment_res
    moments_path = output / "moments.npz"
    basis, moments, metadata = _save_moments(
        moments_path, selected, config.moment_res, config.fb_ell_max, config.batch_size,
        selected_positions=positions, selected_source_rows=rows, selected_image_names=names, n_images=len(positions),
        dataset=manifest["dataset"], pixel_size=pixel, native_side=manifest["native_res"], native_pixel_size=manifest["native_pixel"],
        denoise_side=manifest["denoise_res"], denoised_pixel_size=manifest["denoise_pixel"])
    del selected
    timings["moment_formation"] = time.monotonic() - phase_started
    metadata.update(moment_norms=[float(np.linalg.norm(moment)) for moment in moments])
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
        n_frequencies=basis.n_coefficients,
        n_m2_entries=int(moments[1].size),
        n_canonical_m3_entries=len(moments[2]),
        source=dict(
            star_path=str(manifest["star_path"]),
            data_folder=str(manifest["data_folder"]),
            stack_prefix=str(manifest["stack_prefix"]),
            source_fingerprint=str(manifest["source_fingerprint"]),
            contrast_estimate_sha256=sha256_array(manifest["contrast_est"]),
            contrast_estimate_range=[float(manifest["contrast_est"].min()), float(manifest["contrast_est"].max())],
        ),
        metadata=metadata,
        config={**asdict(config), "denoising_dir": str(config.denoising_dir), "selection_file": str(config.selection_file),
                "output_dir": str(config.output_dir)},
        prepared_file=str(moments_path),
        prepared_sha256=sha256_file(moments_path),
        kam_cryoem_commit=git_commit(ROOT),
    )
    timings["total"] = time.monotonic() - started
    report.update(timings_seconds=timings, elapsed_seconds=timings["total"])
    write_json(output / "report.json", report)


def load_prepared_moments(path, order):
    """Load the moment of order ``order`` (the fit target) and the geometry of a moment archive."""

    path = Path(path).resolve(strict=True)
    report = json.loads((path / "report.json").read_text())
    with np.load(path / "moments.npz", allow_pickle=False) as saved:
        values = {key: np.asarray(saved[key]).copy() for key in saved.files}
    side = int(values["ds_res"].reshape(()))
    fb_basis = FBMomentBasis(side, int(values["fb_ell_max"].reshape(())), radial=tuple(
        values[key] for key in ("fb_radial_nodes", "fb_radial_weights", "fb_radial_basis", "fb_radial_counts")))
    return PreparedMomentData(
        directory=path, report=report, values=values, target=np.asarray(values[f"mbar{order}"], dtype=np.complex128), order=order,
        side=side, pixel_size=float(values["pixel_size"].reshape(())), n_images=int(values["n_images"].reshape(())),
        fb_basis=fb_basis, shapes=fb_basis.shapes)
