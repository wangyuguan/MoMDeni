"""Steerable Fourier--Bessel image moments for in-plane-uniform viewing.

A Fourier image is expanded as ``sum_{q,t} c[q,t] Phi_q,t(kappa) e^{i q phi}``
with ``kappa`` in cycles per moment-grid pixel (``0 <= kappa <= 1/2``), angular
frequency ``|q| <= ell_max``, and a real radial basis ``Phi_q`` that is
orthonormal for ``kappa dkappa``. ``Phi_q`` spans the top singular directions of
the SphFB slice-radial functions ``j_ell(root kappa / 0.5) Y_ell^q(pi/2, 0)``
(``ell = q mod 2``, ``|q| <= ell <= ell_max``), keeping the directions whose
singular values exceed ``RADIAL_SINGULAR_THRESHOLD`` times the largest one of
that ``q`` block. Images are real, so
``c[-q] = (-1)**q conj(c[q])`` and only ``q >= 0`` is stored.

An in-plane rotation by ``alpha`` multiplies ``c[q]`` by a phase in ``q alpha``.
For an in-plane-uniform rotation distribution, every moment entry whose
angular frequencies do not sum to zero vanishes. The retained blocks are

    M1:  E c_0
    M2:  E c_q c_q^H                         q = 0..L
    M3:  E c_a (x) c_b (x) conj(c_{a+b})      0 <= a <= b, a + b <= L

Each block is flattened and multiplied by the square root of the number of
blocks of the complete ``|q| <= L`` tensor it represents (label permutations and
``q -> -q`` conjugates). Euclidean norms and inner products of the flat
vectors therefore equal Frobenius ones of the complete tensors.

The model images are needed only at viewing directions (in-plane angle zero):
averaging over the in-plane angle is exactly the block selection above.
"""

import hashlib
import os
import sys

import finufft
import numpy as np
from scipy.special import jv

DEFAULT_BATCH = 2048
UPSAMPFAC = 2.0
_DEVICE = "cpu"
FB_MOMENT_VERSION = 2
# Relative singular-value cut for the per-q radial basis of the slice span.
RADIAL_SINGULAR_THRESHOLD = 1e-2

def use_device(name):
    """Select the device for computational kernels and return its name.

    GPU execution requires PyTorch with CUDA support.
    """
    global _DEVICE
    name = str(name).lower()
    if name not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {name!r}")
    if name == "gpu":
        from . import utils
        if not utils.gpu_available():
            raise RuntimeError("gpu requested but PyTorch CUDA is unavailable")
    _DEVICE = name
    return _DEVICE


def _on_gpu():
    return _DEVICE == "gpu"


def _progress(seq, total, desc, unit="chunk"):
    """Add a stdout progress bar when ``tqdm`` is available."""
    try:
        from tqdm import tqdm
    except ImportError:
        return seq
    return tqdm(seq, total=total, desc=desc, unit=unit, leave=False, file=sys.stdout)


def _n_batches(M, batch_size):
    return 1 if not batch_size or batch_size >= M else -(-M // batch_size)


def _iter_batches(M, batch_size=DEFAULT_BATCH):
    """Yield ``slice`` objects covering ``range(M)`` in chunks of ``batch_size``."""
    if batch_size is None or batch_size >= M:
        yield slice(0, M)
        return
    for start in range(0, M, batch_size):
        yield slice(start, min(start + batch_size, M))


def centered_fftn(vol):
    """Centered n-D forward FFT:  Vhat[k] = sum_n vol[n] exp(-2πi k·n / L)."""
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol)))


def centered_ifftn(fvol):
    """Centered n-D inverse FFT (inverse of :func:`centered_fftn`)."""
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fvol)))


def centered_fft2(img):
    """Centered two-dimensional forward FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1)),
                                       axes=(-2, -1)),
                           axes=(-2, -1))


def centered_ifft2(fimg):
    """Centered 2-D inverse FFT (inverse of :func:`centered_fft2`)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fimg, axes=(-2, -1)),
                                        axes=(-2, -1)),
                           axes=(-2, -1))


def fourier_crop(vol, res):
    """Resample a real volume to ``res^3`` through its centered Fourier cube.

    Cropping band-limits and downsamples; zero-padding upsamples. Both preserve
    the physical field of view. The factor ``res^3 / L^3`` preserves density
    amplitude, where ``L`` is the input side length.
    """
    L = vol.shape[0]
    if res == L:
        return vol
    Vhat = centered_fftn(vol)
    if res < L:
        start = (L // 2) - (res // 2)
        sl = slice(start, start + res)
        Vhat = Vhat[sl, sl, sl]
    else:
        padded = np.zeros((res, res, res), dtype=Vhat.dtype)
        start = (res // 2) - (L // 2)
        sl = slice(start, start + L)
        padded[sl, sl, sl] = Vhat
        Vhat = padded
    return np.real(centered_ifftn(Vhat)) * (res ** 3 / L ** 3)


def centered_coords_1d(L):
    """Integer frequency coordinate at each array index of a centered grid.

    Array index ``a`` holds frequency ``a - L//2``. For even ``L`` this spans
    ``-L/2, ..., L/2-1``; for odd ``L`` it spans
    ``-(L-1)/2, ..., (L-1)/2``.
    """
    return np.arange(L) - (L // 2)


def _slice_freq_grid(ds_res, L):
    """Central 2-D frequencies normalized by ``L`` in Fortran order.

    Each returned vector has length ``ds_res**2``.
    """
    kc = (np.arange(ds_res) - (ds_res // 2)) / L
    kx, ky = np.meshgrid(kc, kc, indexing="xy")
    return kx.ravel(order="F"), ky.ravel(order="F")


def _rotated_slice_points(rots, ds_res, L):
    """NUFFT sample points 2π·R_i·(kx,ky,0) for every rotation, flattened."""
    kx, ky = _slice_freq_grid(ds_res, L)                                  # (d^2,)
    K = np.vstack((kx, ky))                                               # (2, d^2)
    R2 = np.asarray(rots)[:, :, :2]                                       # (M, 3, 2)
    rot = R2 @ K                                                          # (M, 3, d^2)
    rot = rot.transpose(1, 2, 0)                                          # (3, d^2, M)
    s = 2 * np.pi * rot[0].ravel(order="F")
    t = 2 * np.pi * rot[1].ravel(order="F")
    u = 2 * np.pi * rot[2].ravel(order="F")
    return s, t, u


def slice_images(vol, rots, ds_res=None, eps=1e-5, isign=1):
    """Compute central Fourier slices of a real-space volume with one NUFFT.

    The type-2 transform uses the positive-exponent convention, Fortran pixel
    ordering, and scale ``ds_res^2 / L^2``. For ``ds_res < L``, frequencies are
    sampled directly from the ``L^3`` volume to avoid aliasing rotated slice
    points outside the smaller Nyquist cube.

    Returns ``(M, ds_res, ds_res)`` complex Fourier images.
    """
    L = vol.shape[0]
    if ds_res is None:
        ds_res = L
    M = len(rots)
    s, t, u = _rotated_slice_points(rots, ds_res, L)
    volc = np.ascontiguousarray(vol, dtype=np.complex128)
    c = finufft.nufft3d2(s, t, u, volc, isign=isign, eps=eps, upsampfac=UPSAMPFAC)
    del volc
    Ihat = c.reshape(ds_res ** 2, M, order="F").T
    return Ihat.reshape(M, ds_res, ds_res, order="F") * (ds_res ** 2 / L ** 2)


def nyquist_disk_mask(ds_res, r_max=None):
    """Boolean ``(ds_res, ds_res)`` mask of the 2D frequencies inside the Nyquist disk.

    On the centered grid, index ``a`` holds frequency ``a - ds_res//2``, so the
    default radius is ``ds_res/2``.
    """
    c = centered_coords_1d(ds_res)
    cx, cy = np.meshgrid(c, c, indexing="ij")
    if r_max is None:
        r_max = ds_res / 2.0
    return np.sqrt(cx ** 2 + cy ** 2) <= r_max


def disk_support(ds_res, r_max=None):
    """Indices inside the Nyquist disk in the images' Fortran flattening."""
    return np.flatnonzero(nyquist_disk_mask(ds_res, r_max).ravel(order="F"))


def real_images(Ihat):
    """Convert Fourier images to real images.

    The Nyquist row/col of an even grid has no conjugate partner, so it is zeroed
    to enforce two-dimensional Hermitian symmetry. Moment calculations use the
    original Fourier images.
    """
    Ihat = np.array(Ihat, dtype=np.complex128, copy=True)
    if Ihat.shape[-1] % 2 == 0:
        Ihat[:, 0, :] = 0.0
        Ihat[:, :, 0] = 0.0
    return np.real(centered_ifft2(Ihat))


def embed_support(values, ds_res, support):
    """Embed support-local values in a dense Fourier grid.

    ``values`` has shape ``(n, len(support))``. The result has shape
    ``(n, ds_res, ds_res)`` in Fortran pixel order and is zero off support.
    """
    values = np.asarray(values)
    n = values.shape[0]
    flat = np.zeros((n, ds_res * ds_res), dtype=values.dtype)
    flat[:, support] = values
    return flat.reshape(n, ds_res, ds_res, order="F")


def fourier_crop_images(Ihat, res):
    """Crop Fourier images to a central ``res x res`` block.

    The factor ``res**2 / L**2`` matches :func:`slice_images`. Cropping acts on
    the final two axes and leaves the images in Fourier space.
    """
    L = Ihat.shape[-1]
    if res == L:
        return Ihat
    start = (L // 2) - (res // 2)
    sl = slice(start, start + res)
    return Ihat[..., sl, sl] * (res ** 2 / L ** 2)


class SO3Grid:
    """Product SO(3) grid: S^2 quadrature times a uniform circle.

    Attributes
    ----------
    rots : (M, 3, 3)     all atoms, sphere-major order (atom index = a*n_gamma + j)
    Ns : int             number of sphere nodes
    n_gamma : int        circle nodes
    q : (Ns,)            probability masses for uniform surface measure
    xyz : (Ns, 3)        sphere node directions (unit vectors)
    """

    def __init__(self, rots, Ns, n_gamma, q, xyz):
        self.rots = rots
        self.Ns = int(Ns)
        self.n_gamma = int(n_gamma)
        self.q = q
        self.xyz = xyz
        self.M = rots.shape[0]

    def atom_weights(self, p):
        """SO(3) atom weights for a sphere PMF ``p``:  w_{a,j} = p_a / n_gamma."""
        return np.repeat(np.asarray(p) / self.n_gamma, self.n_gamma)


SPHERE_RULE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "sphere_rules")
_RULE_CACHE = {}


def _sphere_rules():
    """Load and cache the precomputed S^2 quadrature catalogue.

    Files contain rows ``x y z w`` after two header lines. Nodes are normalized
    to the unit sphere and weights to unit sum. Each rule records whether all
    weights are positive. The returned catalogue is sorted by degree.
    """
    if _RULE_CACHE:
        return _RULE_CACHE["rules"]
    import re
    rules = []
    for path in sorted(os.listdir(SPHERE_RULE_DIR)):
        m = re.match(r"N(\d+)_M(\d+)_(\w+)\.dat$", path)
        if not m:
            continue
        d = np.genfromtxt(os.path.join(SPHERE_RULE_DIR, path), skip_header=2)
        xyz, w = np.atleast_2d(d)[:, :3], np.atleast_1d(np.atleast_2d(d)[:, 3])
        rules.append({"degree": int(m.group(1)), "n": w.size,
                      "symmetry": m.group(3), "file": path,
                      "xyz": xyz / np.linalg.norm(xyz, axis=1, keepdims=True),
                      "q": w / w.sum(), "positive": bool(np.all(w > 0))})
    rules.sort(key=lambda r: r["degree"])
    _RULE_CACHE["rules"] = rules
    return rules


def sphere_quadrature(n_points=None, degree=None):
    """Select a positive-weight Gauss quadrature on S^2.

    Specify exactly one criterion: ``n_points`` selects the highest-degree rule
    within the node budget, while ``degree`` selects the smallest rule meeting
    the requested polynomial degree. Rules with negative weights are excluded
    because the weights define a probability mass over viewing directions.

    Returns polar angles, azimuths, normalized weights, and unit directions.
    """
    if (n_points is None) == (degree is None):
        raise ValueError("give exactly one of n_points / degree")
    usable = [r for r in _sphere_rules() if r["positive"]]
    if n_points is not None:
        cand = [r for r in usable if r["n"] <= int(n_points)]
        if not cand:
            raise ValueError(f"no positive-weight rule with <= {n_points} nodes; "
                             f"the smallest has {min(r['n'] for r in usable)}")
        rule = max(cand, key=lambda r: r["degree"])
    else:
        cand = [r for r in usable if r["degree"] >= int(degree)]
        if not cand:
            raise ValueError(f"no positive-weight rule of degree >= {degree}; "
                             f"the largest is {max(r['degree'] for r in usable)}")
        rule = min(cand, key=lambda r: r["n"])
    xyz = rule["xyz"]
    theta = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return theta, phi, rule["q"].copy(), xyz.copy()


def so3_sphere_grid(n_points, n_gamma, degree=None):
    """SO(3) grid = precomputed S^2 quadrature x uniform in-plane angles.

    Each viewing direction from :func:`sphere_quadrature` is paired with
    ``n_gamma`` equispaced in-plane rotations.
    """
    th, ph, q, xyz = sphere_quadrature(n_points=None if degree else n_points,
                                       degree=degree)
    Ns = th.size
    gammas = 2 * np.pi * np.arange(n_gamma) / n_gamma
    rots = np.empty((Ns * n_gamma, 3, 3))
    for a in range(Ns):
        Rab = Rz(ph[a]) @ Ry(th[a])
        for j in range(n_gamma):
            rots[a * n_gamma + j] = Rab @ Rz(gammas[j])
    return SO3Grid(rots, Ns, n_gamma, q, xyz)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

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
        from .optimization import SphFBParam

        self.side = int(side)
        self.ell_max = int(ell_max)
        if self.side < 4 or self.side % 2:
            raise ValueError("side must be even and at least 4")
        if self.ell_max < 0:
            raise ValueError("ell_max must be nonnegative")
        if radial is None:
            param = SphFBParam(self.side, self.ell_max, np.eye(3)[None])
            if param.ell_max != self.ell_max:
                raise ValueError(
                    f"ell_max={self.ell_max} is unsupported at side "
                    f"{self.side}; at most {param.ell_max}")
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
            if (counts.shape != (self.ell_max + 1,) or np.any(counts < 1)
                    or packed.shape != (len(nodes), counts.sum())
                    or weights.shape != nodes.shape):
                raise ValueError("stored FB radial basis has inconsistent shapes")
            self.nodes = nodes.astype(float)
            self.weights = weights.astype(float)
            edges = np.r_[0, np.cumsum(counts)]
            self.radial = [packed[:, edges[q]:edges[q + 1]].astype(float)
                           for q in range(self.ell_max + 1)]
        for U in self.radial:
            if np.max(np.abs(U.T @ U - np.eye(U.shape[1]))) > 1e-10:
                raise ValueError("FB radial basis is not orthonormal")
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
        if np.max(np.abs(ray.imag)) > 1e-12 * max(np.max(np.abs(ray)), 1e-300):
            raise RuntimeError("SphFB ray basis is unexpectedly complex")
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
        :meth:`optimization.SphFBParam._rotated_coefficients`.
        """
        if param.L != self.side or param.ell_max != self.ell_max:
            raise ValueError("SphFBParam does not match the FB moment basis")
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
        image transform matches :func:`moments.fourier_crop_images` applied to
        :func:`moments.centered_fft2`: model coordinate ``x`` is the column
        frequency and ``y`` the row frequency. The angular Fourier integral of
        the image DTFT is evaluated exactly with Bessel functions.
        """
        N = int(image_side)
        if N < self.side:
            raise ValueError("images must be at least the moment side")
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

    def coefficients(self, images, batch_size=2048, operator=None):
        """FB coefficients of a real image stack ``(n, N, N)``."""
        images = np.asarray(images)
        if images.ndim != 3 or images.shape[1] != images.shape[2]:
            raise ValueError("images must have shape (n, N, N)")
        if operator is None:
            operator = self.image_operator(images.shape[1])
        out = np.empty((len(images), self.n_coefficients), dtype=np.complex128)
        for start in range(0, len(images), batch_size):
            batch = np.asarray(images[start:start + batch_size], dtype=np.float64)
            out[start:start + len(batch)] = batch.reshape(len(batch), -1) @ operator
        return out

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
            fb_moment_version=FB_MOMENT_VERSION,
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
        if C.ndim != 2 or C.shape[1] != self.n_coefficients:
            raise ValueError("coefficients must have shape (n, n_coefficients)")
        if w.shape != (len(C),) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite with one value per row")
        return C, w

    def moments(self, C, w, orders=(1, 2, 3), batch_size=4096):
        """Weighted flat M1--M3 of coefficient rows ``C`` with masses ``w``.

        Orders not listed are returned as ``None``.
        """
        C, w = self._check(C, w)
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


def fb_objective_and_grad(theta, p, param, grid, mbar1, mbar2, mbar3,
                          lambdas=None, return_info=False,
                          contrast_scales=None, reflection_augmented=False):
    """FB counterpart of :func:`optimization.joint_objective_and_grad`.

    ``param`` must carry an ``fb_basis`` and be built on in-plane-free
    rotations (``grid.n_gamma == 1``); targets are the flat FB moments.

    With ``reflection_augmented``, targets are the real parts of the
    empirical moments (equal to augmenting every image with its mirror),
    and the model is symmetrized the same way: projections at antipodal
    viewing directions of a real volume are mirror images, so taking the
    real part of the predictions equals antipodally symmetrizing the
    viewing density, and the residuals compare real parts on both sides.
    """
    from .optimization import _validate_contrast_scales, _validate_lambdas

    basis = param.fb_basis
    if grid.n_gamma != 1:
        raise ValueError("FB moments use viewing directions only (n_gamma=1)")
    targets = (mbar1, mbar2, mbar3)
    if lambdas is None:
        lambdas = (1.0,) * 3
    weights = _validate_lambdas(lambdas, targets)
    contrast_scales = _validate_contrast_scales(contrast_scales, 3)
    p = np.asarray(p, dtype=float)
    if p.shape != (grid.Ns,):
        raise ValueError("p must have one probability per sphere node")
    for k, target in enumerate(targets):
        if np.shape(target) != basis.shapes[k]:
            raise ValueError(f"M{k+1} target does not match the FB layout")
    active = [bool(np.any(wt > 0)) for wt in weights]
    C = np.asarray(param.forward_images(theta, rots=grid.rots))
    predictions = basis.moments(
        C, p, orders=tuple(k + 1 for k in range(3) if active[k]))
    masked, residuals, clean = [None] * 3, [None] * 3, [None] * 3
    costs, misfits = np.zeros(3), np.full(3, np.nan)
    for k in range(3):
        if not active[k]:
            continue
        masked[k] = contrast_scales[k] * predictions[k]
        if reflection_augmented:
            # The residual is real, so the conjugate-linear gradient
            # contraction below yields exactly d sum w (Re pred - t)^2.
            masked[k] = masked[k].real.astype(np.complex128)
        residuals[k] = masked[k] - np.asarray(targets[k])
        misfits[k] = np.linalg.norm(residuals[k])
        costs[k] = float(np.sum(weights[k] * np.abs(residuals[k]) ** 2))
        clean[k] = contrast_scales[k] * weights[k] * residuals[k]
    G, grad_p = basis.moment_gradients(C, p, clean)
    grad_theta = param.adjoint_images(G, rots=grid.rots)
    f = float(costs.sum())
    if not return_info:
        return f, grad_theta, grad_p
    info = {"f": f, "contrast_scales": contrast_scales.copy()}
    for k in range(3):
        info.update({f"f{k+1}": float(costs[k]), f"m{k+1}": predictions[k],
                     f"masked_m{k+1}": masked[k],
                     f"misfit{k+1}": float(misfits[k])})
    return f, grad_theta, grad_p, info


def empirical_moments(images, moment_res, ell_max, batch_size=256):
    """Uniformly weighted FB M1--M3 of a real denoised image stack.

    Returns ``(basis, (m1, m2, m3), diagnostics)``.
    """
    images = np.asarray(images)
    if (images.ndim != 3 or len(images) == 0
            or images.shape[1] != images.shape[2]
            or images.shape[1] < moment_res or np.iscomplexobj(images)):
        raise ValueError(
            "images must be a nonempty real square stack at least moment_res")
    basis = FBMomentBasis(moment_res, ell_max)
    operator = basis.image_operator(images.shape[1])
    n = len(images)
    w = np.full(n, 1.0 / n)
    sums = [np.zeros(s, dtype=np.complex128) for s in basis.shapes]
    max_imag_c0 = max_abs_c0 = 0.0
    for start in range(0, n, batch_size):
        batch = np.asarray(images[start:start + batch_size], dtype=np.float64)
        if not np.all(np.isfinite(batch)):
            raise ValueError(f"non-finite denoised pixels in rows {start}:"
                             f"{start + len(batch)}")
        C = batch.reshape(len(batch), -1) @ operator
        c0 = C[:, basis.slices[0]]
        max_imag_c0 = max(max_imag_c0, float(np.max(np.abs(c0.imag))))
        max_abs_c0 = max(max_abs_c0, float(np.max(np.abs(c0))))
        for total, part in zip(sums, basis.moments(C, w[start:start + len(batch)])):
            total += part
    diagnostics = dict(
        q0_reality_relative_error=max_imag_c0 / max(max_abs_c0, 1e-300),
        image_side=int(images.shape[1]),
    )
    return basis, tuple(sums), diagnostics
