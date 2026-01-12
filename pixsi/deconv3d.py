# deconv3d.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict
import math

import torch
import torch.fft as tfft
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

Tensor = torch.Tensor
Meas = Union[
    Tuple[Tuple[int, int], Union[int, float], float],   # ((x,y), t, q)
    Tuple[int, int, Union[int, float], float],          # (x, y, t, q)  (legacy)
]




# ---------------------------- Kernel ------------------------------ #



class Kernel3D:

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
        self.device = device if device is not None else torch.device("cpu")
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

        # Helper: pad a 1D cumulative sequence
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

        # Place quadrant and mirror into all 4 quadrants, centered at (cx,cy)
        for du in range(r + 1):
            for dv in range(r + 1):
                c = pad_cum_1d(quad_cumulative[du][dv])
                
                # 4 symmetric positions (dedup for du==0 or dv==0)
                placements = [
                    (cx + du, cy + dv),
                    (cx - du, cy + dv),
                    (cx + du, cy - dv),
                    (cx - du, cy - dv),
                ]
                for (ix, iy) in set(placements):
                    S_cum[:, ix, iy] = sign_map_full[ix, iy] * c
          

        # Optional spatial swaps/flips (apply to the spatial axes)
        if swap_xy:
            S_cum = np.transpose(S_cum, (0, 2, 1))
        if flip_x:
            S_cum = S_cum[:, ::-1, :]
        if flip_y:
            S_cum = S_cum[:, :, ::-1]

        # ---- Cumulative -> per-interval along time (left-zero baseline) ----
        K_delta_np = np.empty_like(S_cum)
        K_delta_np[0, :, :] = S_cum[0, :, :]
        K_delta_np[1:, :, :] = S_cum[1:, :, :] - S_cum[:-1, :, :] #Should be uncommnted if averaging FR but for taking just regular mid pixel do not need TODO: Needs a permanent fix

        # ---- Optional normalization (off by default) ----
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
        # else: "none" → do nothing

        # ---- Stash tensors ----
        self.K_cum = torch.tensor(S_cum, dtype=self.dtype, device=self.device)       # (Lt, Kx, Ky)
        self.K_delta = torch.tensor(K_delta_np, dtype=self.dtype, device=self.device) # (Lt, Kx, Ky)

        # Meta
        self.Lt, self.Kx, self.Ky = self.K_delta.shape

    # Convenience: move/cast after construction
    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "Kernel3D":
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        self.K_delta = self.K_delta.to(device=device, dtype=dtype)
        self.K_cum   = self.K_cum.to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        return self


# -------------------------- Utilities -------------------------- #

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # torch optional


# ----------------------- Kernel I/O helpers -----------------------

def _to_numpy_kernel3d(kernel_or_K: Any) -> np.ndarray:
    """
    Accepts either:
      • an object with .K_delta (Lt, Kx, Ky) (NumPy or torch)
      • a NumPy array shaped (Lt, Kx, Ky)
    Returns: float64 np.ndarray (Lt, Kx, Ky).
    """
    try:
        import torch
    except Exception:
        torch = None

    if hasattr(kernel_or_K, "K_delta"):
        K = kernel_or_K.K_delta
        if torch is not None and isinstance(K, torch.Tensor):
            Knp = K.detach().cpu().numpy()
        else:
            Knp = np.asarray(K)
    else:
        Knp = np.asarray(kernel_or_K)

    if Knp.ndim != 3 or min(Knp.shape) < 2:
        raise ValueError("Expected kernel shaped (Lt, Kx, Ky) with sizes >= 2.")
    return Knp.astype(np.float64, copy=True)


def _kernel_spatial_center(K: np.ndarray) -> Tuple[int, int]:
    """Return (cx, cy) for the spatial center index of K."""
    _, Kx, Ky = K.shape
    return Kx // 2, Ky // 2


# ----------------------- Small utilities -----------------------

def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _rfftn3(a: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """Real FFT over (t,x,y); ONLY the last axis (y) is half-spectrum."""
    return np.fft.rfftn(a, s=shape, axes=(0, 1, 2))


def _irfftn3(A: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """Inverse real FFT back to real space with output 'shape'."""
    return np.fft.irfftn(A, s=shape, axes=(0, 1, 2))


def _pad3(a: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    T, X, Y = a.shape
    out[:T, :X, :Y] = a
    return out


def _padK(K: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    Lt, Kx, Ky = K.shape
    out[:Lt, :Kx, :Ky] = K
    return out


def _densify_cumulative(
    times: np.ndarray, values: np.ndarray, T: int, *, method: str = "linear", ramp_pre: int = 0
) -> np.ndarray:
    """
    Build a dense cumulative signal from sparse (time, value) pairs.
    method='linear' connects points; method='zoh' is zero-order hold.
    ramp_pre>0: insert a linear ramp before the first sample to avoid a vertical step.
    """
    y = np.zeros(T, dtype=np.float64)
    if times.size == 0:
        return y
    if method not in ("linear", "zoh"):
        raise ValueError("method must be 'linear' or 'zoh'")

    t1 = int(times[0]); v1 = float(values[0])

    if method == "zoh":
        last_val, last_t = 0.0, 0
        for tm, val in zip(times, values):
            if tm > last_t:
                y[last_t:tm] = last_val
            y[tm] = val
            last_val, last_t = val, tm + 1
        if last_t < T:
            y[last_t:] = last_val
        return y

    # linear
    t = np.arange(T)
    y[:] = np.interp(t, times, values, left=0.0, right=values[-1])

    if ramp_pre > 0:
        a = max(0, t1 - ramp_pre)
        if a < t1:
            seg_len = t1 - a
            if seg_len > 0:
                ramp = np.linspace(0.0, v1, seg_len + 1)
                y[a : t1 + 1] = ramp
        y[:a] = 0.0
    else:
        y[:t1] = 0.0

    return y


# ----------------------- 3D forward model -----------------------

def _forward3d_pre_kernel(q: np.ndarray, K: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Linear forward model: dy = (q * K) where:
      • time is causal with length Lt, and the pre-kernel ends at the charge time,
        so we slice time with an advance of (Lt-1) like 1D.
      • spatial kernel is centered at (cx, cy).
    We compute full 3D convolution via FFTs and crop to 'out_shape' = (T,Nx,Ny).
    """
    T, Nx, Ny = out_shape
    Lt, Kx, Ky = map(int, K.shape)
    cx, cy = _kernel_spatial_center(K)

    # Sizes for linear convolution via FFT
    nT = T + Lt - 1
    nX = Nx + Kx - 1
    nY = Ny + Ky - 1
    nfft = (_next_pow2(nT), _next_pow2(nX), _next_pow2(nY))

    # Pad q into the big grid
    buf_q = np.zeros(nfft, dtype=np.float64)
    buf_q[:T, :Nx, :Ny] = q

    # Pad K into the big grid
    buf_K = np.zeros(nfft, dtype=np.float64)
    buf_K[:Lt, :Kx, :Ky] = K

    # 3D convolution via FFT
    QF = _rfftn3(buf_q, nfft)
    KF = _rfftn3(buf_K, nfft)
    YF = QF * KF
    y_full = _irfftn3(YF, nfft).real  # shape nfft

    # Crop: time start at (Lt-1), space start at (cx, cy)
    t0 = Lt - 1
    x0, y0 = cx, cy
    out = y_full[t0:t0 + T, x0:x0 + Nx, y0:y0 + Ny]
    return out


# ----------------------- 3D Wiener deconvolution -----------------------

def _wiener_deconv_3d(
    dy: np.ndarray, K: np.ndarray,
    *, lam0: float = 1e-3, lam_hf: float = 0.0, lam_exp: float = 2.0,
    taper_frac_t: float = 0.0, undo_advance: bool = True
) -> np.ndarray:
    """
    3D Wiener deconvolution:
      Q̂(kt,kx,ky) = H*(kt,kx,ky) / (|H|^2 + λ(kt)) · phase_t(kt) · phase_x(kx) · phase_y(ky) · Y(kt,kx,ky)
    where phase_* undo time and spatial centering of K at (Lt-1, cx, cy).
    λ depends only on temporal frequency (simple, effective).
    """
    T, Nx, Ny = map(int, dy.shape)
    Lt, Kx, Ky = map(int, K.shape)
    cx, cy = _kernel_spatial_center(K)

    # FFT sizes for linear conv
    nfft_t = _next_pow2(T + Lt - 1)
    nfft_x = _next_pow2(Nx + Kx - 1)
    nfft_y = _next_pow2(Ny + Ky - 1)
    nfft = (nfft_t, nfft_x, nfft_y)
    import matplotlib.pyplot as plt
    plt.plot(K[:,4,4])
    plt.show()
    # FFTs (only last axis is half-spectrum)
    Y = _rfftn3(_pad3(dy, nfft), nfft)         # (nfft_t, nfft_x, nfft_y//2+1)
    H = _rfftn3(_padK(K,  nfft), nfft)
    H2 = (H.conj() * H).real

    # ---- λ(kt) (temporal-only ridge), length nfft_t (full complex axis)
    kt = np.arange(nfft_t, dtype=np.float64)
    fnyq = max(1, nfft_t // 2)
    k_fold = np.minimum(kt, nfft_t - kt)                    # fold to [0, fnyq]
    if lam_hf > 0.0:
        lam_vec = lam0 + lam_hf * (k_fold / float(fnyq)) ** float(lam_exp)
    else:
        lam_vec = np.full(nfft_t, float(lam0), dtype=np.float64)
    lam_f = lam_vec[:, None, None]                          # (nfft_t,1,1)

    # ---- Optional temporal taper near Nyquist (symmetric)
    if taper_frac_t > 0.0:
        m = max(1, int(round(taper_frac_t * fnyq)))
        taper_1d = np.ones(nfft_t, dtype=np.float64)
        edge = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, m, endpoint=False)))
        taper_1d[fnyq - m:fnyq] *= edge
        taper_1d[fnyq + 1:fnyq + 1 + m] *= edge[:max(0, min(m, nfft_t - (fnyq + 1)))]
        taper = taper_1d[:, None, None]
    else:
        taper = 1.0

    # ---- Phase ramps to undo kernel centering
    if undo_advance:
        phase_t = np.exp(-2j * np.pi * kt * float(Lt - 1) / float(nfft_t))[:, None, None]
    else:
        phase_t = 1.0
    kx = np.arange(nfft_x, dtype=np.float64)
    ky = np.arange(nfft_y // 2 + 1, dtype=np.float64)
    phase_x = np.exp(-2j * np.pi * kx * float(cx) / float(nfft_x))[None, :, None]
    phase_y = np.exp(-2j * np.pi * ky * float(cy) / float(nfft_y))[None, None, :]

    phase = phase_t * phase_x * phase_y

    # ---- Inverse and crop
    denom = H2 + lam_f
    G = (H.conj() / np.maximum(denom, 1e-20)) * phase * taper
    QhatF = Y * G
    q_full = _irfftn3(QhatF, nfft).real
    q = q_full[:T, :Nx, :Ny]
    return q


def _apply_wiener_effective_filter_to_qtrue_3d(
    q_true: np.ndarray, K: np.ndarray,
    *, lam0: float = 1e-3, lam_hf: float = 0.0, lam_exp: float = 2.0,
    taper_frac_t: float = 0.0,
    use_gaussian: bool = False,
    gauss_sigma_frac: float = 0.2,
) -> np.ndarray:
    """
    Apply the effective reconstruction transfer function W_eff to q_true so it is comparable to q_hat.
    This matches the Wiener shrink |H|^2/(|H|^2+λ) plus optional taper and optional Gaussian multiplier.
    """
    T, Nx, Ny = map(int, q_true.shape)
    Lt, Kx, Ky = map(int, K.shape)

    nfft_t = _next_pow2(T + Lt - 1)
    nfft_x = _next_pow2(Nx + Kx - 1)
    nfft_y = _next_pow2(Ny + Ky - 1)
    nfft = (nfft_t, nfft_x, nfft_y)

    # FFTs
    Q = _rfftn3(_pad3(q_true, nfft), nfft)
    H = _rfftn3(_padK(K, nfft), nfft)
    H2 = (H.conj() * H).real  # shape (nfft_t, nfft_x, nfft_y//2+1)

    # ---- λ(kt) temporal-only ridge
    kt = np.arange(nfft_t, dtype=np.float64)
    fnyq = max(1, nfft_t // 2)
    k_fold = np.minimum(kt, nfft_t - kt)  # fold
    if lam_hf > 0.0:
        lam_vec = lam0 + lam_hf * (k_fold / float(fnyq)) ** float(lam_exp)
    else:
        lam_vec = np.full(nfft_t, float(lam0), dtype=np.float64)
    lam_f = lam_vec[:, None, None]

    # ---- taper in time (same as deconv)
    if taper_frac_t > 0.0:
        m = max(1, int(round(taper_frac_t * fnyq)))
        taper_1d = np.ones(nfft_t, dtype=np.float64)
        edge = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, m, endpoint=False)))
        taper_1d[fnyq - m:fnyq] *= edge
        taper_1d[fnyq + 1:fnyq + 1 + m] *= edge[:max(0, min(m, nfft_t - (fnyq + 1)))]
        taper = taper_1d[:, None, None]
    else:
        taper = 1.0

    # ---- optional Gaussian multiplier (temporal frequency only)
    if use_gaussian:
        # rfft bins for time axis are 0..nfft_t-1 (full complex axis here);
        # we want a symmetric gaussian in folded freq, like your λ uses.
        sigma_bins = max(1e-9, float(gauss_sigma_frac) * float(fnyq))
        Fg = np.exp(-0.5 * (k_fold / sigma_bins) ** 2)[:, None, None]
        Fg[0, 0, 0] = 1.0
    else:
        Fg = 1.0

    # ---- effective Wiener shrink
    W = (H2 / np.maximum(H2 + lam_f, 1e-20)) * taper * Fg

    # apply and invert
    Qf = Q * W
    q_full = _irfftn3(Qf, nfft).real
    return q_full[:T, :Nx, :Ny]


def _adjoint3d_pre_kernel(r: np.ndarray, K: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Adjoint of _forward3d_pre_kernel: maps residual in measurement space (T,Nx,Ny)
    back to charge space (T,Nx,Ny). Satisfies <F q, r> = <q, F* r>.

    F(q) = _forward3d_pre_kernel(q, K, out_shape)
    r    = residual in dy-space (same shape as F(q)).
    """
    T, Nx, Ny = out_shape
    Lt, Kx, Ky = map(int, K.shape)
    cx, cy = _kernel_spatial_center(K)

    # Same FFT box as forward
    nT = T + Lt - 1
    nX = Nx + Kx - 1
    nY = Ny + Ky - 1
    nfft = (nT, nX, nY)

    # Embed residual back into the full conv domain (inverse of cropping)
    buf_r = np.zeros(nfft, dtype=np.float64)
    t0 = Lt - 1
    x0, y0 = cx, cy
    buf_r[t0:t0 + T, x0:x0 + Nx, y0:y0 + Ny] = r

    # Pad kernel like in forward
    buf_K = np.zeros(nfft, dtype=np.float64)
    buf_K[:Lt, :Kx, :Ky] = K

    # Adjoint of conv: multiply by conj(KF) in Fourier space
    RF = _rfftn3(buf_r, nfft)
    KF = _rfftn3(buf_K, nfft)
    GF = RF * KF.conj()
    g_full = _irfftn3(GF, nfft).real

    # Crop back to q domain support
    grad = g_full[:T, :Nx, :Ny]
    return grad



# ----------------------- Global time alignment ("model") -----------------------

def _estimate_global_shift_model_3d(dy_meas: np.ndarray, y_model: np.ndarray) -> int:
    """
    Estimate a single integer lag Δ over time that maximizes the global cross-correlation
    between dy_meas and y_model, aggregated over all pixels.
    Inputs are per-interval (non-cumulative) waveforms.
    """
    T = int(dy_meas.shape[0])
    Y_meas = np.fft.rfft(dy_meas, axis=0)           # (Kt,Nx,Ny)
    Y_mod  = np.fft.rfft(y_model,  axis=0)          # (Kt,Nx,Ny)
    cross_power = (Y_meas * np.conj(Y_mod)).sum(axis=(1, 2))  # (Kt,)
    r = np.fft.irfft(cross_power, n=T)              # global xcorr over lags
    tau = int(np.argmax(r))
    if tau > T // 2:
        tau -= T
    return tau


def _frac_shift_time_cube(q: np.ndarray, delta: float) -> np.ndarray:
    """
    Fractional shift along time for the whole cube using a phase ramp in temporal frequency.
    """
    T = q.shape[0]
    Q = np.fft.rfft(q, axis=0)
    k = np.arange(Q.shape[0], dtype=np.float64)
    phase = np.exp(+2j * np.pi * k * (float(delta) / float(T)))[:, None, None]
    return np.fft.irfft(Q * phase, n=T, axis=0)


# ----------------------- Data ingestion helpers -----------------------

def _group_by_pixel(meas: Iterable[Tuple[Tuple[int,int], int, float]]) -> Dict[Tuple[int,int], List[Tuple[int,float]]]:
    d: Dict[Tuple[int,int], List[Tuple[int,float]]] = {}
    for (x, y), t, v in meas:
        d.setdefault((int(x), int(y)), []).append((int(t), float(v)))
    # sort times per pixel and collapse duplicates (keep last)
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


def canonicalize_kernel_time_support(K: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Space-variant kernel bank K[t, dx, dy] -> Kcanon[Tmax, dx, dy]
    For each (dx,dy) find its last 'significant' time index and left-pad
    so that ALL kernels end at t = Tmax-1. Prevents tail cropping and
    makes the single 'advance' (Lt-1) correct for every neighbor.
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
            shift = Tmax - Lij                 # left-pad so tail ends at Tmax-1
            if Lij > 0:
                Kc[shift:Tmax, i, j] = K[:Lij, i, j]
    return Kc

# ======================= Public class =======================

class Deco3D:
    """
    Deconvolution on real measurements with model-alignment.

    Inputs:
      - kernel_or_K: 3D kernel (Lt,Kx,Ky), e.g., 9x9 spatial footprint.
      - measurements: list of tuples [((x,y), t, value), ...]
          * If measurement_type='cumulative', 'value' is cumulative (non-decreasing).
          * If measurement_type='incremental', 'value' is per-interval current.

      - true_list (optional): same format as measurements, but representing true charge q(t)
          (interpreted as incremental by default: q[t_i] += value_i).

    Usage:
        deco = Deco3D(ker)
        out = deco.run(
            measurements,
            measurement_type="cumulative",
            true_list=true_data, true_is_incremental=True,
            pixels_to_plot=[(x1,y1),(x2,y2)]
        )
    """
    def __init__(self, kernel_or_K: Any):
        self.K = _to_numpy_kernel3d(kernel_or_K)#canonicalize_kernel_time_support(_to_numpy_kernel3d(kernel_or_K))
        #k_tmp = self.K.copy()
        #self.K*=1.5
        #self.K[:,4,4]=k_tmp[:,4,4]/1.5
        self.Lt, self.Kx, self.Ky = map(int, self.K.shape)
        self.cx, self.cy = _kernel_spatial_center(self.K)

        # Will be filled on run()
        self.roi_origin: Tuple[int,int] = (0,0)
        self.grid_shape: Tuple[int,int,int] = (0,0,0)  # (T,Nx,Ny)

    # ---------- core pipeline ----------

    def _build_dense_from_measurements(
        self,
        measurements: List[Tuple[Tuple[int,int], int, float]],
        *,
        measurement_type: str = "cumulative",
        T: Optional[int] = None,
        roi_margin: int = 0,
        densify_method: str = "linear",
        ramp_pre: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int,int], np.ndarray], Dict[Tuple[int,int], np.ndarray]]:
        """
        Returns:
          y_cum_dense (T,Nx,Ny), dy_meas (T,Nx,Ny),
          meas_times_dict[(x,y)] -> np.array([...]),
          meas_vals_dict[(x,y)]  -> np.array([...])
        """
        groups = _group_by_pixel(measurements)
        if not groups:
            raise ValueError("No measurements provided.")

        x_min, x_max, y_min, y_max = _roi_from_keys(groups.keys())
        x_min -= roi_margin; x_max += roi_margin
        y_min -= roi_margin; y_max += roi_margin

        # ROI mapping
        def to_local(ix: int, iy: int) -> Tuple[int,int]:
            return ix - x_min, iy - y_min

        # time length
        t_max_meas = max((t for arr in groups.values() for (t, _) in arr), default=0)
        T_auto = int(t_max_meas + 1 + max(0, self.Lt - 1))
        T = int(T) if T is not None else T_auto

        Nx, Ny = x_max - x_min + 1, y_max - y_min + 1
        y_cum_dense = np.zeros((T, Nx, Ny), dtype=np.float64)
        dy_meas = np.zeros_like(y_cum_dense)

        meas_times_dict: Dict[Tuple[int,int], np.ndarray] = {}
        meas_vals_dict:  Dict[Tuple[int,int], np.ndarray] = {}

        if ramp_pre is None:
            ramp_pre = int(min(self.Lt, 32))  # a modest default

        for (gx, gy), arr in groups.items():
            lx, ly = to_local(gx, gy)
            times = np.array([t for (t, _) in arr], dtype=int)
            vals  = np.array([v for (_, v) in arr], dtype=np.float64)
            meas_times_dict[(gx,gy)] = times
            meas_vals_dict[(gx,gy)]  = vals

            if measurement_type == "cumulative":
                yc = _densify_cumulative(times, vals, T, method=densify_method, ramp_pre=ramp_pre)
                y_cum_dense[:, lx, ly] = yc
                dy_meas[:, lx, ly] = np.diff(yc, axis=0, prepend=0.0)
            elif measurement_type == "incremental":
                # Directly place increments at recorded times and hold zeros elsewhere
                dy = np.zeros(T, dtype=np.float64)
                dy[times] += vals
                dy_meas[:, lx, ly] = dy
                y_cum_dense[:, lx, ly] = np.cumsum(dy)
            else:
                raise ValueError("measurement_type must be 'cumulative' or 'incremental'")

        self.roi_origin = (x_min, y_min)
        self.grid_shape = (T, Nx, Ny)
        return y_cum_dense, dy_meas, meas_times_dict, meas_vals_dict

    def _build_q_true(
        self,
        true_list: Optional[List[Tuple[Tuple[int,int], int, float]]],
        *,
        true_is_incremental: bool = True,
        T: Optional[int] = None,
        roi_origin: Optional[Tuple[int,int]] = None,
        grid_xy: Optional[Tuple[int,int]] = None
    ) -> np.ndarray:
        if true_list is None or len(true_list) == 0:
            Tn, Nx, Ny = self.grid_shape
            return np.zeros((Tn, Nx, Ny), dtype=np.float64)

        groups = _group_by_pixel(true_list)
        if T is None:
            t_max = max((t for arr in groups.values() for (t, _) in arr), default=0)
            T = int(max(self.grid_shape[0], t_max + 1))
        if roi_origin is None:
            roi_origin = self.roi_origin
        if grid_xy is None:
            grid_xy = self.grid_shape[1:]
        Nx, Ny = grid_xy
        x0, y0 = roi_origin

        q_true = np.zeros((T, Nx, Ny), dtype=np.float64)
        for (gx, gy), arr in groups.items():
            lx, ly = int(gx - x0), int(gy - y0)
            if lx < 0 or ly < 0 or lx >= Nx or ly >= Ny:
                continue  # outside ROI
            times = np.array([t for (t, _) in arr], dtype=int)
            vals  = np.array([v for (_, v) in arr], dtype=np.float64)
            if true_is_incremental:
                q_true[times, lx, ly] += vals
            else:
                # interpret as cumulative; convert to increments
                dense = _densify_cumulative(times, vals, T, method="linear", ramp_pre=0)
                q_true[:, lx, ly] += np.diff(dense, axis=0, prepend=0.0)
        return q_true

    def zero_first_nonzero_bin(self,dy: np.ndarray, eps: float = 1e-12, conserve_total: bool = False) -> np.ndarray:
        """
        For each pixel (x,y), find the first time index t where |dy[t,x,y]| > eps
        and set dy[t,x,y] = 0.
        If conserve_total=True, move that removed value into the next bin (t+1) when possible.
        """
        dy2 = dy.copy()
        T, Nx, Ny = dy2.shape

        for ix in range(Nx):
            for iy in range(Ny):
                col = dy2[:, ix, iy]
                nz = np.flatnonzero(np.abs(col) > eps)
                if nz.size == 0:
                    continue
                t0 = int(nz[0])
                v0 = float(col[t0])
                dy2[t0, ix, iy] = 0.0
                if conserve_total and t0 + 1 < T:
                    dy2[t0 + 1, ix, iy] += v0

        return dy2
    
    # ---------- public runner ----------

    def run(
            self,
            measurements: List[Tuple[Tuple[int,int], int, float]],
            *,
            measurement_type: str = "cumulative",
            true_list: Optional[List[Tuple[Tuple[int,int], int, float]]] = None,
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
            pixels_to_plot: Optional[List[Tuple[int,int]]] = None,
            show_maps: bool = False,
            show: bool = True
        ) -> Dict[str, Any]:

        # Build dense cumulative + derivative from sparse measurements
        y_cum_dense, dy_meas, meas_times_dict, meas_vals_dict = self._build_dense_from_measurements(
            measurements,
            measurement_type=measurement_type,
            T=T, roi_margin=roi_margin,
            densify_method=densify_method, ramp_pre=ramp_pre
        )
        Tn, Nx, Ny = self.grid_shape
        print("Grid Shape Used : ",self.grid_shape)
        # True charge (optional)
        q_true = self._build_q_true(
            true_list,
            true_is_incremental=true_is_incremental,
            T=Tn, roi_origin=self.roi_origin, grid_xy=(Nx, Ny)
        )

        #dy_meas = self.zero_first_nonzero_bin(dy_meas, eps=0.1, conserve_total=False)

        # Deconvolution
        q_hat = _wiener_deconv_3d(
            dy_meas, self.K,
            lam0=lam0, lam_hf=lam_hf, lam_exp=lam_exp,
            taper_frac_t=taper_frac_t, undo_advance=True
        )

        # Model alignment
        dt_applied = 0.0
        if align == "model":
            y_model = _forward3d_pre_kernel(q_hat, self.K, (Tn, Nx, Ny))
            tau = _estimate_global_shift_model_3d(dy_meas, y_model)
            dt_applied = float(tau)
            print("Defined t0 : ",dt_applied)
            if align_fractional and abs(tau) > 0:
                q_hat = _frac_shift_time_cube(q_hat, -dt_applied)
            elif tau != 0:
                q_hat = np.roll(q_hat, shift=-int(tau), axis=0)

        # Keep a copy of raw Stage-A solution for debugging/comparison
        q_hat_stageA = q_hat.copy()

        # ----- Stage B: optional non-negative refinement via projected gradient descent -----
        if refine_nonneg and refine_iters > 0:
            Tn, Nx, Ny = self.grid_shape

            # Start from non-negative version of Stage A
            q = np.clip(q_hat, 0.0, None)
            lam_pos = float(refine_lam)
            eta = float(refine_step)

            for it in range(refine_iters):
                # Forward model: dy_model = K * q
                dy_model = _forward3d_pre_kernel(q, self.K, (Tn, Nx, Ny))
                resid = dy_model - dy_meas

                # Backproject residual into charge space (adjoint)
                grad_data = _adjoint3d_pre_kernel(resid, self.K, (Tn, Nx, Ny))
                grad = grad_data + lam_pos * q

                # Gradient step + projection to q >= 0
                q = q - eta * grad
                q = np.clip(q, 0.0, None)

                if refine_verbose and (it % max(1, refine_iters // 5) == 0 or it == refine_iters - 1):
                    data_term = float(np.mean(resid**2))
                    reg_term = float(np.mean(q**2))
                    print(f"[StageB it={it:03d}] data={data_term:.4e} reg={reg_term:.4e}")

            # Use refined solution as final q_hat
            q_hat = q


        # -------------- plotting --------------
        x0, y0 = self.roi_origin
        if pixels_to_plot is None or len(pixels_to_plot) < 2:
            # pick two non-empty pixels if not provided
            nonempty = [k for k, v in meas_times_dict.items() if len(v) > 0]
            if len(nonempty) >= 2:
                pixels_to_plot = [nonempty[0], nonempty[1]]
            else:
                pixels_to_plot = [(x0, y0), (x0, min(y0 + 1, y0 + Ny - 1))]

        if show:
            t = np.arange(Tn)
            q_true = q_true #/ 1000.0

            # Three rows: [0]=charge, [1]=per-interval dy, [2]=cumulative
            fig, axs = plt.subplots(3, 2, figsize=(12.5, 7.0), sharex=True)

            for j, (gx, gy) in enumerate(pixels_to_plot[:2]):
                lx, ly = gx - x0, gy - y0
                lx = int(np.clip(lx, 0, Nx - 1))
                ly = int(np.clip(ly, 0, Ny - 1))

                # -------- Row 0: charges (true vs deconvolved) --------
                
                q_true_eff = _apply_wiener_effective_filter_to_qtrue_3d(
                    q_true, self.K,
                    lam0=lam0, lam_hf=lam_hf, lam_exp=lam_exp,
                    taper_frac_t=taper_frac_t,
                    use_gaussian=False,          # set True if/when you add gaussian option to 3D deconv
                    gauss_sigma_frac=0.2
                )
                
                ax_top = axs[0, j]
                if q_true.size:
                    ax_top.plot(t, q_true[:, lx, ly], lw=2.0,
                                label="True $q_{true}(t)$")
                    ax_top.plot(t, q_true_eff[:, lx, ly], lw=1.8, ls="--",
                                label="True filtered (Wiener eff)", alpha=0.95)
                q_shift=np.zeros_like(q_hat[:, lx, ly])
                q_shift[:]=q_hat[:, lx, ly]
                ax_top.plot(t, q_shift, lw=1.8,
                            label="Deconv $\\hat q(t)$", alpha=0.95)
                ax_top.set_title(f"Pixel ({gx},{gy})  [ROI ({lx},{ly})]")
                ax_top.set_ylabel("charge")
                ax_top.grid(alpha=0.25)
                ax_top.legend(loc="best", fontsize=9)

                # Get sparse measurement times/values
                tm = meas_times_dict.get((gx, gy), np.array([], dtype=int))
                vm = meas_vals_dict.get((gx, gy), np.array([], dtype=float))

                # -------- Row 1: per-interval measurement dy --------
                ax_mid = axs[1, j]
                ax_mid.plot(t, dy_meas[:, lx, ly], lw=1.5,
                            label="$\\Delta y_{meas}(t)$ (input to deconv)")

                # Also show sparse increments at measurement times
                if tm.size and False:
                    if measurement_type == "cumulative":
                        # increments = differences of cumulative samples
                        inc_sparse = np.diff(vm, prepend=0.0)
                    else:
                        # already incremental at the given times
                        inc_sparse = vm
                    markerline, stemlines, baseline = ax_mid.stem(
                        tm, inc_sparse,
                        linefmt="C1-", markerfmt="C1o", basefmt="k-"
                    )
                    # Optional styling (modern Matplotlib needs this)
                    plt.setp(markerline, markersize=6)
                    plt.setp(stemlines, linewidth=1.0)
                    # Add label manually
                    markerline.set_label("sparse $\\Delta y$")

                ax_mid.set_ylabel("$\\Delta y$")
                ax_mid.grid(alpha=0.25)
                ax_mid.legend(loc="best", fontsize=9)

                # -------- Row 2: cumulative (dense + samples) --------
                ax_bot = axs[2, j]
                ax_bot.plot(t, y_cum_dense[:, lx, ly],
                            ls=":", lw=1.5, alpha=0.9,
                            label="$y_{cum}^{dense}(t)$")

                if tm.size:
                    ax_bot.scatter(tm, vm, s=36, marker="o", zorder=5,
                                   label="meas (cum)" if measurement_type == "cumulative"
                                                        else "meas (incr)")

                ax_bot.set_xlabel("time (samples)")
                ax_bot.set_ylabel("cumulative")
                ax_bot.set_ylim(bottom=0.0)
                ax_bot.grid(alpha=0.25)
                ax_bot.legend(loc="best", fontsize=9)

            plt.tight_layout()
            plt.show()

            if show_maps:
                # 2D maps: ∑_t q_true, last measured cumulative, ∑_t q_hat

                q_true_sum = q_true.sum(axis=0) if q_true.size else np.zeros((Nx, Ny))
                q_hat_sum  = q_hat.sum(axis=0)
                meas_total = y_cum_dense[-1, :, :]

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
        total_true = float(np.sum(q_true)) if q_true.size else 0.0
        total_meas = float(np.sum(y_cum_dense[-1]))  # last cumulative sample
        total_stageA = float(np.sum(q_hat_stageA))   # before Stage B
        total_final = float(np.sum(q_hat))           # after Stage B refinement (or same as stageA if B disabled)

        print("\n=== Charge Summary ===")
        print(f"  Total true charge (sum q_true)            : {total_true:.6f}")
        print(f"  Total measured cumulative (last y_cum)    : {total_meas:.6f}")
        print(f"  Total deconvolved charge (Stage A q_hat)  : {total_stageA:.6f}")
        print(f"  Total refined charge (final q_hat)        : {total_final:.6f}")
        print("===========================================\n")

        return dict(
            K=self.K, Lt=self.Lt, Kx=self.Kx, Ky=self.Ky,
            roi_origin=self.roi_origin, grid_shape=self.grid_shape,
            y_cum_dense=y_cum_dense, dy_meas=dy_meas,
            q_true=q_true, q_hat=q_hat,
            meas_times=meas_times_dict, meas_vals=meas_vals_dict,
            meta=dict(
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
