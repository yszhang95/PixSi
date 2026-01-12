# deconv3d.py
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any
import math

import torch
import torch.fft as tfft

import numpy as np
import matplotlib.pyplot as plt

Tensor = torch.Tensor
ArrayLike = Union[np.ndarray, Tensor]
Meas = Union[
    Tuple[Tuple[int, int], Union[int, float], float],   # ((x,y), t, q)
    Tuple[int, int, Union[int, float], float],          # (x, y, t, q)  (legacy)
]


# =============================================================================
# Kernel
# =============================================================================

class Kernel3D:
    """
    Build a symmetric 3D kernel bank from a (r+1)x(r+1) quadrant of cumulative traces.

    Stores:
      - K_cum:   (Lt, Kx, Ky) cumulative
      - K_delta: (Lt, Kx, Ky) per-interval (difference of cumulative)
    """

    def __init__(
        self,
        quad_cumulative: List[List[Sequence[float]]],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        *,
        pre_pad: int = 0,
        post_mode: str = "hold",
        sign_map: Optional[np.ndarray] = None,
        dc_mode: str = "none",
        swap_xy: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
    ):
        # IMPORTANT: default to CUDA if available (so users don't have to pass device)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        # ---- Validate quadrant and infer radius ----
        if not isinstance(quad_cumulative, (list, tuple)) or len(quad_cumulative) == 0:
            raise ValueError("quad_cumulative must be a (radius+1)×(radius+1) nested list.")
        r = len(quad_cumulative) - 1
        for row in quad_cumulative:
            if len(row) != r + 1:
                raise ValueError("quad_cumulative must be rectangular with shape (r+1)×(r+1).")

        self.radius = int(r)
        G = 2 * r + 1
        cx = cy = r  # center index in the full (2r+1)x(2r+1)
        self.sx, self.sy = cx, cy

        # ---- Determine target time length ----
        lengths = []
        for du in range(r + 1):
            for dv in range(r + 1):
                seq = quad_cumulative[du][dv]
                if seq is None:
                    raise ValueError(f"quad_cumulative[{du}][{dv}] is None.")
                lengths.append(len(seq))
        if len(lengths) == 0:
            raise ValueError("Empty quadrant sequences.")
        Lmax = max(lengths)
        if Lmax <= 0:
            raise ValueError("Quadrant sequences must be non-empty.")
        L_target = Lmax + int(pre_pad)

        # ---- Build full cumulative S_cum: shape (L_target, G, G) ----
        S_cum = np.zeros((L_target, G, G), dtype=np.float32)

        # Default sign map (all +1)
        if sign_map is None:
            sign_map_full = np.ones((G, G), dtype=np.float32)
        else:
            sign_map = np.asarray(sign_map, dtype=np.float32)
            if sign_map.shape != (G, G):
                raise ValueError(f"sign_map must have shape {(G, G)}, got {sign_map.shape}")
            sign_map_full = sign_map

        def pad_cum_1d(seq: Sequence[float]) -> np.ndarray:
            arr = np.asarray(seq, dtype=np.float32)
            if pre_pad > 0:
                arr = np.concatenate([np.zeros(pre_pad, dtype=np.float32), arr])
            if arr.shape[0] > L_target:
                raise ValueError("Sequence longer than target; reduce pre_pad or increase L_target.")
            if arr.shape[0] < L_target:
                if post_mode == "hold":
                    last = arr[-1] if arr.size else 0.0
                    arr = np.concatenate([arr, np.full(L_target - arr.shape[0], last, dtype=np.float32)])
                elif post_mode == "none":
                    raise ValueError("post_mode='none' but lengths differ; supply equal-length inputs.")
                else:
                    raise ValueError(f"Unknown post_mode: {post_mode}")
            return arr

        for du in range(r + 1):
            for dv in range(r + 1):
                c = pad_cum_1d(quad_cumulative[du][dv])

                placements = [
                    (cx + du, cy + dv),
                    (cx - du, cy + dv),
                    (cx + du, cy - dv),
                    (cx - du, cy - dv),
                ]
                for (ix, iy) in set(placements):
                    S_cum[:, ix, iy] = sign_map_full[ix, iy] * c

        # Optional spatial swaps/flips
        if swap_xy:
            S_cum = np.transpose(S_cum, (0, 2, 1))
        if flip_x:
            S_cum = S_cum[:, ::-1, :]
        if flip_y:
            S_cum = S_cum[:, :, ::-1]

        # ---- Cumulative -> per-interval along time ----
        K_delta_np = np.empty_like(S_cum)
        K_delta_np[0, :, :] = S_cum[0, :, :]
        K_delta_np[1:, :, :] = S_cum[1:, :, :] - S_cum[:-1, :, :]

        # ---- Optional normalization ----
        if dc_mode not in ("none", "global", "center"):
            raise ValueError(f"Unknown dc_mode: {dc_mode}")

        if dc_mode == "global":
            total = float(K_delta_np.sum())
            if not np.isfinite(total) or abs(total) < 1e-12:
                raise ValueError("Global gain near-zero/invalid; check inputs.")
            K_delta_np = K_delta_np / total
        elif dc_mode == "center":
            center_gain = float(K_delta_np[:, cx, cy].sum())
            if not np.isfinite(center_gain) or abs(center_gain) < 1e-12:
                raise ValueError("Center gain near-zero/invalid; check inputs.")
            K_delta_np = K_delta_np / center_gain

        # ---- Stash tensors ----
        self.K_cum = torch.tensor(S_cum, dtype=self.dtype, device=self.device)          # (Lt, Kx, Ky)
        self.K_delta = torch.tensor(K_delta_np, dtype=self.dtype, device=self.device)  # (Lt, Kx, Ky)

        self.Lt, self.Kx, self.Ky = map(int, self.K_delta.shape)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "Kernel3D":
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        self.K_delta = self.K_delta.to(device=device, dtype=dtype)
        self.K_cum   = self.K_cum.to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        return self


# =============================================================================
#  original NumPy kernel extractor 
# =============================================================================

def _to_numpy_kernel3d(kernel_or_K: Any) -> np.ndarray:
    """
    Accepts either:
      • an object with .K_delta (Lt, Kx, Ky) (NumPy or torch)
      • a NumPy array shaped (Lt, Kx, Ky)
    Returns: float64 np.ndarray (Lt, Kx, Ky).
    """
    if hasattr(kernel_or_K, "K_delta"):
        K = kernel_or_K.K_delta
        if torch.is_tensor(K):
            Knp = K.detach().cpu().numpy()
        else:
            Knp = np.asarray(K)
    else:
        Knp = np.asarray(kernel_or_K)

    if Knp.ndim != 3 or min(Knp.shape) < 2:
        raise ValueError("Expected kernel shaped (Lt, Kx, Ky) with sizes >= 2.")
    return Knp.astype(np.float64, copy=True)


# =============================================================================
# Small utilities
# =============================================================================

def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())

def _kernel_spatial_center(K: ArrayLike) -> Tuple[int, int]:
    """Return (cx, cy) for the spatial center index of K."""
    if torch.is_tensor(K):
        _, Kx, Ky = K.shape
        return int(Kx // 2), int(Ky // 2)
    _, Kx, Ky = K.shape
    return int(Kx // 2), int(Ky // 2)

def _as_torch3(x: ArrayLike, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(x), device=device, dtype=dtype)

def _as_measurement_list(measurements: Any) -> List[Tuple[Tuple[int,int], int, float]]:
    """
    Normalizes various measurement containers into a list of ((x,y), t, value).

    Accepts:
      - list/iterable of ((x,y), t, v) or (x,y,t,v)
      - numpy array (N,4) with columns [x,y,t,v]
      - torch tensor (N,4)
    """
    if measurements is None:
        return []
    if isinstance(measurements, np.ndarray):
        arr = np.asarray(measurements)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("If measurements is a NumPy array, expected shape (N,4) [x,y,t,v].")
        return [((int(x), int(y)), int(t), float(v)) for x, y, t, v in arr[:, :4]]
    if torch.is_tensor(measurements):
        arr = measurements.detach().cpu()
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("If measurements is a torch Tensor, expected shape (N,4) [x,y,t,v].")
        return [((int(x.item()), int(y.item())), int(t.item()), float(v.item())) for x, y, t, v in arr[:, :4]]

    out: List[Tuple[Tuple[int,int], int, float]] = []
    for m in measurements:
        if len(m) == 3:
            (x, y), t, v = m
            out.append(((int(x), int(y)), int(t), float(v)))
        elif len(m) == 4:
            x, y, t, v = m
            out.append(((int(x), int(y)), int(t), float(v)))
        else:
            raise ValueError("Each measurement must be ((x,y),t,v) or (x,y,t,v).")
    return out

def _group_by_pixel(meas: Iterable[Tuple[Tuple[int,int], int, float]]) -> Dict[Tuple[int,int], List[Tuple[int,float]]]:
    d: Dict[Tuple[int,int], List[Tuple[int,float]]] = {}
    for (x, y), t, v in meas:
        d.setdefault((int(x), int(y)), []).append((int(t), float(v)))
    for k, arr in d.items():
        arr.sort(key=lambda tv: tv[0])
        uniq_t = {}
        for t, v in arr:
            uniq_t[t] = v
        d[k] = sorted(uniq_t.items(), key=lambda tv: tv[0])
    return d

def _roi_from_keys(keys: Iterable[Tuple[int,int]]) -> Tuple[int,int,int,int]:
    xs = [x for (x, _) in keys]; ys = [y for (_, y) in keys]
    return min(xs), max(xs), min(ys), max(ys)


# =============================================================================
# Densify cumulative
# =============================================================================

def _densify_cumulative_torch(
    times: Tensor,
    values: Tensor,
    T: int,
    *,
    method: str = "linear",
    ramp_pre: int = 0,
) -> Tensor:
    """
    Torch version of dense cumulative reconstruction from sparse samples.
    - linear: piecewise linear interpolation (np.interp-like), with left=0 and right=values[-1]
    - zoh:    zero-order hold
    ramp_pre>0: insert a linear ramp in the window [t1-ramp_pre, t1] to avoid a vertical step.
    """
    if times.numel() == 0:
        return torch.zeros((T,), device=values.device, dtype=values.dtype)

    if method not in ("linear", "zoh"):
        raise ValueError("method must be 'linear' or 'zoh'")

    device = values.device
    dtype = values.dtype

    times = times.to(device=device, dtype=torch.long)
    values = values.to(device=device, dtype=dtype)

    # Ensure sorted
    if times.numel() > 1 and not bool(torch.all(times[1:] >= times[:-1]).item()):
        idx = torch.argsort(times)
        times = times[idx]
        values = values[idx]

    if method == "zoh":
        y = torch.zeros((T,), device=device, dtype=dtype)
        last_t = 0
        last_v = torch.zeros((), device=device, dtype=dtype)
        for tm, vm in zip(times.tolist(), values.tolist()):
            tm = int(tm)
            if tm > last_t:
                y[last_t:tm] = last_v
            if 0 <= tm < T:
                y[tm] = torch.tensor(vm, device=device, dtype=dtype)
            last_v = torch.tensor(vm, device=device, dtype=dtype)
            last_t = tm + 1
        if last_t < T:
            y[last_t:] = last_v
        return y

    # linear
    t_grid = torch.arange(T, device=device, dtype=torch.long)
    idx = torch.searchsorted(times, t_grid, right=False)  # [T]
    m = int(times.numel())

    y = torch.empty((T,), device=device, dtype=dtype)

    left_mask = idx == 0
    right_mask = idx >= m
    mid_mask = (~left_mask) & (~right_mask)

    y[left_mask] = 0.0
    y[right_mask] = values[-1]

    if bool(mid_mask.any().item()):
        i1 = idx[mid_mask]
        i0 = i1 - 1
        t0 = times[i0].to(dtype=dtype)
        t1 = times[i1].to(dtype=dtype)
        v0 = values[i0]
        v1 = values[i1]
        tt = t_grid[mid_mask].to(dtype=dtype)

        denom = (t1 - t0).clamp_min(1.0)
        w = (tt - t0) / denom
        y[mid_mask] = v0 + w * (v1 - v0)

    # Optional pre-ramp
    t1i = int(times[0].item())
    if ramp_pre and ramp_pre > 0:
        v1 = values[0]
        a = max(0, t1i - int(ramp_pre))
        if a < t1i:
            seg = torch.arange(a, t1i + 1, device=device, dtype=dtype)
            y[a:t1i + 1] = (seg - float(a)) / max(1.0, float(t1i - a)) * v1
        if a > 0:
            y[:a] = 0.0
    else:
        if t1i > 0:
            y[:t1i] = 0.0

    return y


# =============================================================================
# FFT helpers
# =============================================================================

def _rfftn3(a: Tensor, shape: Tuple[int, int, int]) -> Tensor:
    return tfft.rfftn(a, s=shape, dim=(0, 1, 2))

def _irfftn3(A: Tensor, shape: Tuple[int, int, int]) -> Tensor:
    return tfft.irfftn(A, s=shape, dim=(0, 1, 2))

def _pad3(a: Tensor, shape: Tuple[int, int, int]) -> Tensor:
    out = torch.zeros(shape, device=a.device, dtype=a.dtype)
    T, X, Y = a.shape
    out[:T, :X, :Y] = a
    return out

def _padK(K: Tensor, shape: Tuple[int, int, int]) -> Tensor:
    out = torch.zeros(shape, device=K.device, dtype=K.dtype)
    Lt, Kx, Ky = K.shape
    out[:Lt, :Kx, :Ky] = K
    return out


# =============================================================================
# 3D forward model
# =============================================================================

def _forward3d_pre_kernel(q: ArrayLike, K: ArrayLike, out_shape: Tuple[int, int, int]) -> ArrayLike:
    """
    Linear forward model: dy = (q * K) with:
      - time crop starting at (Lt-1)
      - spatial crop starting at (cx,cy)

    If q/K are torch tensors -> returns torch tensor.
    If numpy arrays -> runs via torch on CPU and returns numpy.
    """
    is_torch = torch.is_tensor(q) or torch.is_tensor(K)
    device = q.device if torch.is_tensor(q) else (K.device if torch.is_tensor(K) else torch.device("cpu"))
    dtype = q.dtype if torch.is_tensor(q) else (K.dtype if torch.is_tensor(K) else torch.float32)

    qt = _as_torch3(q, device=device, dtype=dtype)
    Kt = _as_torch3(K, device=device, dtype=dtype)

    T, Nx, Ny = map(int, out_shape)
    Lt, Kx, Ky = map(int, Kt.shape)
    cx, cy = _kernel_spatial_center(Kt)

    nT = T + Lt - 1
    nX = Nx + Kx - 1
    nY = Ny + Ky - 1
    nfft = (_next_pow2(nT), _next_pow2(nX), _next_pow2(nY))

    buf_q = torch.zeros(nfft, device=device, dtype=dtype)
    buf_q[:T, :Nx, :Ny] = qt

    buf_K = torch.zeros(nfft, device=device, dtype=dtype)
    buf_K[:Lt, :Kx, :Ky] = Kt

    y_full = _irfftn3(_rfftn3(buf_q, nfft) * _rfftn3(buf_K, nfft), nfft)
    out = y_full[(Lt - 1):(Lt - 1 + T), cx:(cx + Nx), cy:(cy + Ny)]

    if is_torch:
        return out
    return out.detach().cpu().numpy()


# =============================================================================
# 3D Wiener deconvolution
# =============================================================================

def _wiener_deconv_3d(
    dy: ArrayLike, K: ArrayLike,
    *, lam0: float = 1e-3, lam_hf: float = 0.0, lam_exp: float = 2.0,
    taper_frac_t: float = 0.0, undo_advance: bool = True
) -> ArrayLike:
    """
    3D Wiener deconvolution.

    If inputs are torch -> returns torch (on same device).
    If inputs are numpy -> runs via torch on CPU and returns numpy.
    """
    is_torch = torch.is_tensor(dy) or torch.is_tensor(K)
    device = dy.device if torch.is_tensor(dy) else (K.device if torch.is_tensor(K) else torch.device("cpu"))
    dtype = dy.dtype if torch.is_tensor(dy) else (K.dtype if torch.is_tensor(K) else torch.float32)

    dyt = _as_torch3(dy, device=device, dtype=dtype)
    Kt = _as_torch3(K, device=device, dtype=dtype)

    T, Nx, Ny = map(int, dyt.shape)
    Lt, Kx, Ky = map(int, Kt.shape)
    cx, cy = _kernel_spatial_center(Kt)

    nfft_t = _next_pow2(T + Lt - 1)
    nfft_x = _next_pow2(Nx + Kx - 1)
    nfft_y = _next_pow2(Ny + Ky - 1)
    nfft = (nfft_t, nfft_x, nfft_y)

    Y = _rfftn3(_pad3(dyt, nfft), nfft)
    H = _rfftn3(_padK(Kt, nfft), nfft)
    H2 = (H.conj() * H).real

    # λ(kt) depends only on folded temporal freq
    kt = torch.arange(nfft_t, device=device, dtype=torch.float32)
    fnyq = max(1, nfft_t // 2)
    k_fold = torch.minimum(kt, torch.tensor(nfft_t, device=device, dtype=torch.float32) - kt)
    if lam_hf > 0.0:
        lam_vec = float(lam0) + float(lam_hf) * (k_fold / float(fnyq)).pow(float(lam_exp))
    else:
        lam_vec = torch.full((nfft_t,), float(lam0), device=device, dtype=torch.float32)
    lam_f = lam_vec.to(dtype=dtype)[:, None, None]

    # Optional time taper
    if taper_frac_t > 0.0:
        m = max(1, int(round(taper_frac_t * fnyq)))
        taper_1d = torch.ones((nfft_t,), device=device, dtype=dtype)
        edge = 0.5 * (1.0 + torch.cos(torch.linspace(0.0, math.pi, m, device=device, dtype=dtype)))
        taper_1d[fnyq - m:fnyq] *= edge
        hi_start = fnyq + 1
        hi_end = min(nfft_t, hi_start + m)
        if hi_end > hi_start:
            taper_1d[hi_start:hi_end] *= edge[:hi_end - hi_start]
        taper = taper_1d[:, None, None]
    else:
        taper = 1.0

    # Phase ramps to undo kernel centering
    ctype = torch.complex64 if dtype == torch.float32 else torch.complex128
    twopi = 2.0 * math.pi
    if undo_advance:
        phase_t = torch.exp((-1j * twopi) * kt * float(Lt - 1) / float(nfft_t)).to(ctype)[:, None, None]
    else:
        phase_t = torch.ones((nfft_t, 1, 1), device=device, dtype=ctype)

    kx = torch.arange(nfft_x, device=device, dtype=torch.float32)
    ky = torch.arange(nfft_y // 2 + 1, device=device, dtype=torch.float32)
    phase_x = torch.exp((-1j * twopi) * kx * float(cx) / float(nfft_x)).to(ctype)[None, :, None]
    phase_y = torch.exp((-1j * twopi) * ky * float(cy) / float(nfft_y)).to(ctype)[None, None, :]

    phase = phase_t * phase_x * phase_y

    denom = torch.clamp(H2 + lam_f, min=1e-20)
    G = (H.conj() / denom.to(H.dtype)) * phase * taper
    q_full = _irfftn3(Y * G, nfft)
    q = q_full[:T, :Nx, :Ny]

    if is_torch:
        return q
    return q.detach().cpu().numpy()


def _apply_wiener_effective_filter_to_qtrue_3d(
    q_true: ArrayLike, K: ArrayLike,
    *, lam0: float = 1e-3, lam_hf: float = 0.0, lam_exp: float = 2.0,
    taper_frac_t: float = 0.0,
    use_gaussian: bool = False,
    gauss_sigma_frac: float = 0.2,
) -> ArrayLike:
    """
    Apply the effective Wiener reconstruction transfer function to q_true so it is comparable to q_hat.
    """
    is_torch = torch.is_tensor(q_true) or torch.is_tensor(K)
    device = q_true.device if torch.is_tensor(q_true) else (K.device if torch.is_tensor(K) else torch.device("cpu"))
    dtype = q_true.dtype if torch.is_tensor(q_true) else (K.dtype if torch.is_tensor(K) else torch.float32)

    qt = _as_torch3(q_true, device=device, dtype=dtype)
    Kt = _as_torch3(K, device=device, dtype=dtype)

    T, Nx, Ny = map(int, qt.shape)
    Lt, Kx, Ky = map(int, Kt.shape)

    nfft_t = _next_pow2(T + Lt - 1)
    nfft_x = _next_pow2(Nx + Kx - 1)
    nfft_y = _next_pow2(Ny + Ky - 1)
    nfft = (nfft_t, nfft_x, nfft_y)

    Q = _rfftn3(_pad3(qt, nfft), nfft)
    H = _rfftn3(_padK(Kt, nfft), nfft)
    H2 = (H.conj() * H).real

    kt = torch.arange(nfft_t, device=device, dtype=torch.float32)
    fnyq = max(1, nfft_t // 2)
    k_fold = torch.minimum(kt, torch.tensor(nfft_t, device=device, dtype=torch.float32) - kt)
    if lam_hf > 0.0:
        lam_vec = float(lam0) + float(lam_hf) * (k_fold / float(fnyq)).pow(float(lam_exp))
    else:
        lam_vec = torch.full((nfft_t,), float(lam0), device=device, dtype=torch.float32)
    lam_f = lam_vec.to(dtype=dtype)[:, None, None]

    if taper_frac_t > 0.0:
        m = max(1, int(round(taper_frac_t * fnyq)))
        taper_1d = torch.ones((nfft_t,), device=device, dtype=dtype)
        edge = 0.5 * (1.0 + torch.cos(torch.linspace(0.0, math.pi, m, device=device, dtype=dtype)))
        taper_1d[fnyq - m:fnyq] *= edge
        hi_start = fnyq + 1
        hi_end = min(nfft_t, hi_start + m)
        if hi_end > hi_start:
            taper_1d[hi_start:hi_end] *= edge[:hi_end - hi_start]
        taper = taper_1d[:, None, None]
    else:
        taper = 1.0

    if use_gaussian:
        sigma_bins = max(1e-9, float(gauss_sigma_frac) * float(fnyq))
        Fg = torch.exp(-0.5 * (k_fold / float(sigma_bins)).pow(2.0)).to(dtype=dtype)[:, None, None]
        Fg[0, 0, 0] = 1.0
    else:
        Fg = 1.0

    W = (H2 / torch.clamp(H2 + lam_f, min=1e-20)) * taper * Fg
    q_full = _irfftn3(Q * W.to(Q.dtype), nfft)
    out = q_full[:T, :Nx, :Ny]

    if is_torch:
        return out
    return out.detach().cpu().numpy()


# =============================================================================
# Adjoint
# =============================================================================

def _adjoint3d_pre_kernel(r: ArrayLike, K: ArrayLike, out_shape: Tuple[int, int, int]) -> ArrayLike:
    """
    Adjoint of _forward3d_pre_kernel, satisfying <F q, r> = <q, F* r>.
    """
    is_torch = torch.is_tensor(r) or torch.is_tensor(K)
    device = r.device if torch.is_tensor(r) else (K.device if torch.is_tensor(K) else torch.device("cpu"))
    dtype = r.dtype if torch.is_tensor(r) else (K.dtype if torch.is_tensor(K) else torch.float32)

    rt = _as_torch3(r, device=device, dtype=dtype)
    Kt = _as_torch3(K, device=device, dtype=dtype)

    T, Nx, Ny = map(int, out_shape)
    Lt, Kx, Ky = map(int, Kt.shape)
    cx, cy = _kernel_spatial_center(Kt)

    nT = T + Lt - 1
    nX = Nx + Kx - 1
    nY = Ny + Ky - 1
    nfft = (nT, nX, nY)

    buf_r = torch.zeros(nfft, device=device, dtype=dtype)
    t0 = Lt - 1
    buf_r[t0:t0 + T, cx:cx + Nx, cy:cy + Ny] = rt

    buf_K = torch.zeros(nfft, device=device, dtype=dtype)
    buf_K[:Lt, :Kx, :Ky] = Kt

    g_full = _irfftn3(_rfftn3(buf_r, nfft) * _rfftn3(buf_K, nfft).conj(), nfft)
    grad = g_full[:T, :Nx, :Ny]

    if is_torch:
        return grad
    return grad.detach().cpu().numpy()


# =============================================================================
# Global time alignment ("model")
# =============================================================================

def _estimate_global_shift_model_3d(dy_meas: ArrayLike, y_model: ArrayLike) -> int:
    """
    Estimate a single integer lag Δ maximizing global cross-correlation over time,
    aggregated over all pixels.
    """
    dyt = dy_meas if torch.is_tensor(dy_meas) else torch.tensor(np.asarray(dy_meas), device="cpu", dtype=torch.float32)
    ymt = y_model if torch.is_tensor(y_model) else torch.tensor(np.asarray(y_model), device="cpu", dtype=torch.float32)

    T = int(dyt.shape[0])
    Y_meas = tfft.rfft(dyt, dim=0)
    Y_mod  = tfft.rfft(ymt, dim=0)
    cross_power = (Y_meas * Y_mod.conj()).sum(dim=(1, 2))
    r = tfft.irfft(cross_power, n=T, dim=0)
    tau = int(torch.argmax(r).item())
    if tau > T // 2:
        tau -= T
    return tau

def _frac_shift_time_cube(q: ArrayLike, delta: float) -> ArrayLike:
    """
    Fractional shift along time using a phase ramp in temporal frequency.
    """
    if not torch.is_tensor(q):
        qt = torch.tensor(np.asarray(q), device="cpu", dtype=torch.float32)
        is_torch = False
    else:
        qt = q
        is_torch = True

    T = int(qt.shape[0])
    Q = tfft.rfft(qt, dim=0)
    k = torch.arange(Q.shape[0], device=qt.device, dtype=torch.float32)
    phase = torch.exp((1j * 2.0 * math.pi) * k * (float(delta) / float(T))).to(Q.dtype)[:, None, None]
    out = tfft.irfft(Q * phase, n=T, dim=0)

    if is_torch:
        return out
    return out.detach().cpu().numpy()


# =============================================================================
# Kernel time-support canonicalization (CPU helper)
# =============================================================================

def canonicalize_kernel_time_support(K: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Space-variant kernel bank K[t, dx, dy] -> Kcanon[Tmax, dx, dy]
    For each (dx,dy) find its last 'significant' time index and left-pad
    so that ALL kernels end at t = Tmax-1.
    """
    assert K.ndim == 3
    Lt, Kx, Ky = K.shape
    last = np.zeros((Kx, Ky), dtype=int)
    for i in range(Kx):
        for j in range(Ky):
            col = np.abs(K[:, i, j])
            nz = np.flatnonzero(col > eps)
            last[i, j] = (nz[-1] if nz.size else 0)
    Tmax = int(last.max()) + 1

    Kc = np.zeros((Tmax, Kx, Ky), dtype=np.float64)
    for i in range(Kx):
        for j in range(Ky):
            Lij = last[i, j] + 1
            shift = Tmax - Lij
            if Lij > 0:
                Kc[shift:Tmax, i, j] = K[:Lij, i, j]
    return Kc


# =============================================================================
# Public class
# =============================================================================

class Deco3D:
    """
    Deconvolution on real measurements with model-alignment.

    - Heavy operations (FFTs, Wiener deconv, Stage-B refinement) run on torch (GPU if available).
    - Measurement densification is still per-pixel Python bookkeeping, but writes into device tensors.
    """

    def __init__(self, kernel_or_K: Any, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        # Choose device from kernel if possible
        if device is None:
            if hasattr(kernel_or_K, "K_delta") and torch.is_tensor(kernel_or_K.K_delta):
                device = kernel_or_K.K_delta.device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        if hasattr(kernel_or_K, "K_delta") and torch.is_tensor(kernel_or_K.K_delta):
            self.K = kernel_or_K.K_delta.to(device=self.device, dtype=self.dtype)
        else:
            self.K = _as_torch3(kernel_or_K, device=self.device, dtype=self.dtype)

        self.Lt, self.Kx, self.Ky = map(int, self.K.shape)
        self.cx, self.cy = _kernel_spatial_center(self.K)

        # Will be filled on run()
        self.roi_origin: Tuple[int, int] = (0, 0)
        self.grid_shape: Tuple[int, int, int] = (0, 0, 0)  # (T, Nx, Ny)

    # ---------- core pipeline ----------

    def _build_dense_from_measurements(
        self,
        measurements: Any,
        *,
        measurement_type: str = "cumulative",
        T: Optional[int] = None,
        roi_margin: int = 0,
        densify_method: str = "linear",
        ramp_pre: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Dict[Tuple[int,int], np.ndarray], Dict[Tuple[int,int], np.ndarray]]:
        groups = _group_by_pixel(_as_measurement_list(measurements))
        if not groups:
            raise ValueError("No measurements provided.")

        x_min, x_max, y_min, y_max = _roi_from_keys(groups.keys())
        x_min -= roi_margin; x_max += roi_margin
        y_min -= roi_margin; y_max += roi_margin

        def to_local(ix: int, iy: int) -> Tuple[int, int]:
            return ix - x_min, iy - y_min

        t_max_meas = max((t for arr in groups.values() for (t, _) in arr), default=0)
        T_auto = int(t_max_meas + 1 + max(0, self.Lt - 1))
        Tn = int(T) if T is not None else T_auto

        Nx, Ny = x_max - x_min + 1, y_max - y_min + 1
        y_cum_dense = torch.zeros((Tn, Nx, Ny), device=self.device, dtype=self.dtype)
        dy_meas = torch.zeros_like(y_cum_dense)

        meas_times_dict: Dict[Tuple[int, int], np.ndarray] = {}
        meas_vals_dict: Dict[Tuple[int, int], np.ndarray] = {}

        if ramp_pre is None:
            ramp_pre = int(min(self.Lt, 32))

        for (gx, gy), arr in groups.items():
            lx, ly = to_local(gx, gy)
            times_np = np.array([t for (t, _) in arr], dtype=np.int64)
            vals_np = np.array([v for (_, v) in arr], dtype=np.float32)

            meas_times_dict[(gx, gy)] = times_np
            meas_vals_dict[(gx, gy)] = vals_np.astype(np.float64, copy=False)

            times = torch.from_numpy(times_np).to(device=self.device)
            vals = torch.from_numpy(vals_np).to(device=self.device, dtype=self.dtype)

            if measurement_type == "cumulative":
                yc = _densify_cumulative_torch(times, vals, Tn, method=densify_method, ramp_pre=int(ramp_pre))
                y_cum_dense[:, lx, ly] = yc
                dy = torch.empty_like(yc)
                dy[0] = yc[0]
                dy[1:] = yc[1:] - yc[:-1]
                dy_meas[:, lx, ly] = dy
            elif measurement_type == "incremental":
                dy = torch.zeros((Tn,), device=self.device, dtype=self.dtype)
                valid = (times >= 0) & (times < Tn)
                dy.index_add_(0, times[valid].to(torch.long), vals[valid])
                dy_meas[:, lx, ly] = dy
                y_cum_dense[:, lx, ly] = torch.cumsum(dy, dim=0)
            else:
                raise ValueError("measurement_type must be 'cumulative' or 'incremental'")

        self.roi_origin = (x_min, y_min)
        self.grid_shape = (Tn, Nx, Ny)
        return y_cum_dense, dy_meas, meas_times_dict, meas_vals_dict

    def _build_q_true(
        self,
        true_list: Optional[Any],
        *,
        true_is_incremental: bool = True,
        T: Optional[int] = None,
        roi_origin: Optional[Tuple[int, int]] = None,
        grid_xy: Optional[Tuple[int, int]] = None
    ) -> Tensor:
        if true_list is None:
            Tn, Nx, Ny = self.grid_shape
            return torch.zeros((Tn, Nx, Ny), device=self.device, dtype=self.dtype)

        groups = _group_by_pixel(_as_measurement_list(true_list))
        if not groups:
            Tn, Nx, Ny = self.grid_shape
            return torch.zeros((Tn, Nx, Ny), device=self.device, dtype=self.dtype)

        if T is None:
            t_max = max((t for arr in groups.values() for (t, _) in arr), default=0)
            T = int(max(self.grid_shape[0], t_max + 1))
        if roi_origin is None:
            roi_origin = self.roi_origin
        if grid_xy is None:
            grid_xy = self.grid_shape[1:]
        Nx, Ny = map(int, grid_xy)
        x0, y0 = roi_origin

        q_true = torch.zeros((int(T), Nx, Ny), device=self.device, dtype=self.dtype)

        for (gx, gy), arr in groups.items():
            lx, ly = int(gx - x0), int(gy - y0)
            if lx < 0 or ly < 0 or lx >= Nx or ly >= Ny:
                continue
            times_np = np.array([t for (t, _) in arr], dtype=np.int64)
            vals_np = np.array([v for (_, v) in arr], dtype=np.float32)
            times = torch.from_numpy(times_np).to(device=self.device, dtype=torch.long)
            vals = torch.from_numpy(vals_np).to(device=self.device, dtype=self.dtype)

            valid = (times >= 0) & (times < int(T))
            if not bool(valid.any().item()):
                continue

            if true_is_incremental:
                q_true.index_put_(
                    (times[valid],
                     torch.full_like(times[valid], lx),
                     torch.full_like(times[valid], ly)),
                    vals[valid],
                    accumulate=True
                )
            else:
                yc = _densify_cumulative_torch(times[valid], vals[valid], int(T), method="linear", ramp_pre=0)
                dy = torch.empty_like(yc)
                dy[0] = yc[0]
                dy[1:] = yc[1:] - yc[:-1]
                q_true[:, lx, ly] += dy

        return q_true

    def zero_first_nonzero_bin(self, dy: Tensor, eps: float = 1e-12, conserve_total: bool = False) -> Tensor:
        """
        For each pixel (x,y), find the first time index t where |dy[t,x,y]| > eps and set it to 0.
        If conserve_total=True, move that removed value into the next bin (t+1) when possible.
        """
        dy2 = dy.clone()
        T, Nx, Ny = dy2.shape
        for ix in range(Nx):
            for iy in range(Ny):
                col = dy2[:, ix, iy]
                nz = torch.nonzero(col.abs() > float(eps), as_tuple=False).view(-1)
                if nz.numel() == 0:
                    continue
                t0 = int(nz[0].item())
                v0 = col[t0].item()
                dy2[t0, ix, iy] = 0.0
                if conserve_total and (t0 + 1) < T:
                    dy2[t0 + 1, ix, iy] += float(v0)
        return dy2

    # ---------- public runner ----------

    def run(
        self,
        measurements: Any,
        *,
        measurement_type: str = "cumulative",
        true_list: Optional[Any] = None,
        true_is_incremental: bool = True,
        roi_margin: int = 0,
        T: Optional[int] = None,
        densify_method: str = "linear",
        ramp_pre: Optional[int] = None,
        # deconvolution controls (Stage A)
        lam0: float = 1e-3, lam_hf: float = 0.0, lam_exp: float = 2.0,
        taper_frac_t: float = 0.0,
        align: str = "model", align_fractional: bool = True,
        # Stage B: optional non-negative refinement
        refine_nonneg: bool = False,
        refine_iters: int = 0,
        refine_step: float = 0.05,
        refine_lam: float = 0.0,
        refine_verbose: bool = False,
        # plotting
        pixels_to_plot: Optional[List[Tuple[int, int]]] = None,
        show_maps: bool = False,
        show: bool = True,
        # return control
        return_torch: bool = False,
    ) -> Dict[str, Any]:
        print("Run GPU version of Deconvolution")
        y_cum_dense, dy_meas, meas_times_dict, meas_vals_dict = self._build_dense_from_measurements(
            measurements,
            measurement_type=measurement_type,
            T=T, roi_margin=roi_margin,
            densify_method=densify_method, ramp_pre=ramp_pre
        )
        Tn, Nx, Ny = self.grid_shape
        print("Grid Shape Used : ", self.grid_shape)

        q_true = self._build_q_true(
            true_list,
            true_is_incremental=true_is_incremental,
            T=Tn, roi_origin=self.roi_origin, grid_xy=(Nx, Ny)
        )

        # Stage A: Wiener deconvolution (GPU)
        q_hat = _wiener_deconv_3d(
            dy_meas, self.K,
            lam0=lam0, lam_hf=lam_hf, lam_exp=lam_exp,
            taper_frac_t=taper_frac_t, undo_advance=True
        )
        if not torch.is_tensor(q_hat):
            q_hat = torch.tensor(q_hat, device=self.device, dtype=self.dtype)

        # Model alignment (optional)
        dt_applied = 0.0
        if align == "model":
            y_model = _forward3d_pre_kernel(q_hat, self.K, (Tn, Nx, Ny))
            if not torch.is_tensor(y_model):
                y_model = torch.tensor(y_model, device=self.device, dtype=self.dtype)
            tau = _estimate_global_shift_model_3d(dy_meas, y_model)
            dt_applied = float(tau)
            print("Defined t0 : ", dt_applied)
            if align_fractional and abs(tau) > 0:
                q_hat = _frac_shift_time_cube(q_hat, -dt_applied)
                if not torch.is_tensor(q_hat):
                    q_hat = torch.tensor(q_hat, device=self.device, dtype=self.dtype)
            elif tau != 0:
                q_hat = torch.roll(q_hat, shifts=-int(tau), dims=0)

        q_hat_stageA = q_hat.clone()

        # Stage B: optional non-negative refinement via projected gradient descent (GPU)
        if refine_nonneg and refine_iters > 0:
            q = torch.clamp(q_hat, min=0.0)
            lam_pos = float(refine_lam)
            eta = float(refine_step)

            for it in range(int(refine_iters)):
                dy_model = _forward3d_pre_kernel(q, self.K, (Tn, Nx, Ny))
                if not torch.is_tensor(dy_model):
                    dy_model = torch.tensor(dy_model, device=self.device, dtype=self.dtype)
                resid = dy_model - dy_meas
                grad_data = _adjoint3d_pre_kernel(resid, self.K, (Tn, Nx, Ny))
                if not torch.is_tensor(grad_data):
                    grad_data = torch.tensor(grad_data, device=self.device, dtype=self.dtype)
                grad = grad_data + lam_pos * q
                q = torch.clamp(q - eta * grad, min=0.0)

                if refine_verbose and (it % max(1, refine_iters // 5) == 0 or it == refine_iters - 1):
                    data_term = float(resid.float().pow(2).mean().item())
                    reg_term = float(q.float().pow(2).mean().item())
                    print(f"[StageB it={it:03d}] data={data_term:.4e} reg={reg_term:.4e}")

            q_hat = q

        # -------------- plotting --------------
        x0, y0 = self.roi_origin
        if pixels_to_plot is None or len(pixels_to_plot) < 2:
            nonempty = [k for k, v in meas_times_dict.items() if len(v) > 0]
            if len(nonempty) >= 2:
                pixels_to_plot = [nonempty[0], nonempty[1]]
            else:
                pixels_to_plot = [(x0, y0), (x0, min(y0 + 1, y0 + Ny - 1))]

        if show:
            t = np.arange(Tn)

            q_true_eff = _apply_wiener_effective_filter_to_qtrue_3d(
                q_true, self.K,
                lam0=lam0, lam_hf=lam_hf, lam_exp=lam_exp,
                taper_frac_t=taper_frac_t,
                use_gaussian=False,
                gauss_sigma_frac=0.2
            )
            if torch.is_tensor(q_true_eff):
                q_true_eff_np = q_true_eff.detach().cpu().numpy()
            else:
                q_true_eff_np = np.asarray(q_true_eff)

            q_true_np = q_true.detach().cpu().numpy()
            q_hat_np = q_hat.detach().cpu().numpy()
            dy_meas_np = dy_meas.detach().cpu().numpy()
            y_cum_dense_np = y_cum_dense.detach().cpu().numpy()

            fig, axs = plt.subplots(3, 2, figsize=(12.5, 7.0), sharex=True)

            for j, (gx, gy) in enumerate(pixels_to_plot[:2]):
                lx, ly = gx - x0, gy - y0
                lx = int(np.clip(lx, 0, Nx - 1))
                ly = int(np.clip(ly, 0, Ny - 1))

                ax_top = axs[0, j]
                if q_true_np.size:
                    ax_top.plot(t, q_true_np[:, lx, ly], lw=2.0, label="True $q_{true}(t)$")
                    ax_top.plot(t, q_true_eff_np[:, lx, ly], lw=1.8, ls="--",
                                label="True filtered (Wiener eff)", alpha=0.95)
                ax_top.plot(t, q_hat_np[:, lx, ly], lw=1.8, label="Deconv $\\hat q(t)$", alpha=0.95)
                ax_top.set_title(f"Pixel ({gx},{gy})  [ROI ({lx},{ly})]")
                ax_top.set_ylabel("charge")
                ax_top.grid(alpha=0.25)
                ax_top.legend(loc="best", fontsize=9)

                tm = meas_times_dict.get((gx, gy), np.array([], dtype=int))
                vm = meas_vals_dict.get((gx, gy), np.array([], dtype=float))

                ax_mid = axs[1, j]
                ax_mid.plot(t, dy_meas_np[:, lx, ly], lw=1.5,
                            label="$\\Delta y_{meas}(t)$ (input to deconv)")
                ax_mid.set_ylabel("$\\Delta y$")
                ax_mid.grid(alpha=0.25)
                ax_mid.legend(loc="best", fontsize=9)

                ax_bot = axs[2, j]
                ax_bot.plot(t, y_cum_dense_np[:, lx, ly], ls=":", lw=1.5, alpha=0.9,
                            label="$y_{cum}^{dense}(t)$")
                if tm.size:
                    ax_bot.scatter(tm, vm, s=36, marker="o", zorder=5,
                                   label="meas (cum)" if measurement_type == "cumulative" else "meas (incr)")
                ax_bot.set_xlabel("time (samples)")
                ax_bot.set_ylabel("cumulative")
                ax_bot.set_ylim(bottom=0.0)
                ax_bot.grid(alpha=0.25)
                ax_bot.legend(loc="best", fontsize=9)

            plt.tight_layout()
            plt.show()

            if show_maps:
                q_true_sum = q_true_np.sum(axis=0) if q_true_np.size else np.zeros((Nx, Ny))
                q_hat_sum  = q_hat_np.sum(axis=0)
                meas_total = y_cum_dense_np[-1, :, :]

                fig, axm = plt.subplots(1, 3, figsize=(12.5, 3.8))
                im0 = axm[0].imshow(q_true_sum.T, origin="lower", aspect="equal")
                axm[0].set_title("Total true charge  ∑$_t q_{true}$"); plt.colorbar(im0, ax=axm[0], fraction=0.046, pad=0.04)
                im1 = axm[1].imshow(meas_total.T, origin="lower", aspect="equal")
                axm[1].set_title("Total measured (last $y_{cum}$)"); plt.colorbar(im1, ax=axm[1], fraction=0.046, pad=0.04)
                im2 = axm[2].imshow(q_hat_sum.T, origin="lower", aspect="equal")
                axm[2].set_title("Total deconvolved  ∑$_t \\hat q$"); plt.colorbar(im2, ax=axm[2], fraction=0.046, pad=0.04)
                for a in axm:
                    a.set_xlabel("x"); a.set_ylabel("y")
                plt.tight_layout(); plt.show()

        # ----- Total charge diagnostics -----
        total_true = float(q_true.double().sum().item()) if q_true.numel() else 0.0
        total_meas = float(y_cum_dense[-1].double().sum().item()) if y_cum_dense.numel() else 0.0
        total_stageA = float(q_hat_stageA.double().sum().item()) if q_hat_stageA.numel() else 0.0
        total_final = float(q_hat.double().sum().item()) if q_hat.numel() else 0.0

        print("\n=== Charge Summary ===")
        print(f"  Total true charge (sum q_true)            : {total_true:.6f}")
        print(f"  Total measured cumulative (last y_cum)    : {total_meas:.6f}")
        print(f"  Total deconvolved charge (Stage A q_hat)  : {total_stageA:.6f}")
        print(f"  Total refined charge (final q_hat)        : {total_final:.6f}")
        print("===========================================\n")

        if return_torch:
            y_cum_out: Any = y_cum_dense
            dy_out: Any = dy_meas
            q_true_out: Any = q_true
            q_hat_out: Any = q_hat
            K_out: Any = self.K
        else:
            y_cum_out = y_cum_dense.detach().cpu().numpy()
            dy_out = dy_meas.detach().cpu().numpy()
            q_true_out = q_true.detach().cpu().numpy()
            q_hat_out = q_hat.detach().cpu().numpy()
            K_out = self.K.detach().cpu().numpy()

        return dict(
            K=K_out, Lt=self.Lt, Kx=self.Kx, Ky=self.Ky,
            roi_origin=self.roi_origin, grid_shape=self.grid_shape,
            y_cum_dense=y_cum_out, dy_meas=dy_out,
            q_true=q_true_out, q_hat=q_hat_out,
            meas_times=meas_times_dict, meas_vals=meas_vals_dict,
            meta=dict(
                device=str(self.device),
                dtype=str(self.dtype).replace("torch.", ""),
                lam0=lam0, lam_hf=lam_hf, lam_exp=lam_exp, taper_frac_t=taper_frac_t,
                align=align, align_fractional=align_fractional,
                time_shift_applied_samples=dt_applied,
                pixels_plotted=pixels_to_plot[:2],
                stageB=dict(
                    enabled=bool(refine_nonneg and refine_iters > 0),
                    iters=refine_iters,
                    step=refine_step,
                    lam=refine_lam,
                ),
            )
        )
