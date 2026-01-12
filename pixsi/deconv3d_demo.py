# deconv3d_demo.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import matplotlib.pyplot as plt

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # only used for type hints if torch exists


# ----------------------- Kernel I/O helpers -----------------------

def _to_numpy_kernel3d(kernel_or_K: Any) -> np.ndarray:
    """
    Accepts either:
      • an object with .K_delta (Lt, Kx, Ky) possibly torch.Tensor, or
      • a NumPy array shaped (Lt, Kx, Ky).
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


def _first_crossing(y: np.ndarray, thr: float) -> Optional[int]:
    idx = np.flatnonzero(y >= thr)
    return int(idx[0]) if idx.size else None


def _time_centroid_1d(x: np.ndarray) -> float:
    w = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    s = w.sum()
    if s <= 0.0:
        return float(x.size / 2.0)
    t = np.arange(x.size, dtype=np.float64)
    return float((t * w).sum() / s)


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


# ----------------------- 3D FFT helpers -----------------------

def _rfftn3(a: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """rFFTN over (t,x,y) -> freq space (k_t, k_x, k_y) with k_t non-negative."""
    return np.fft.rfftn(a, s=shape, axes=(0, 1, 2))


def _irfftn3(A: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """inverse rFFTN back to real (t,x,y) with output 'shape'."""
    return np.fft.irfftn(A, s=shape, axes=(0, 1, 2))


# ----------------------- Forward model (3D) -----------------------

def _forward3d_pre_kernel(q: np.ndarray, K: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Linear forward model: dy = (q * K) where:
      • time is causal with length Lt, and the pre-kernel *ends* at charge time,
        so we slice time with an advance of (Lt-1) like 1D.
      • spatial kernel is centered at (cx,cy).
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

    # Crop:
    #   time: start at (Lt-1)
    #   space: start at (cx, cy)
    t0 = Lt - 1
    x0 = cx
    y0 = cy
    out = y_full[t0:t0 + T, x0:x0 + Nx, y0:y0 + Ny]
    return out


# ----------------------- 3D Wiener deconvolution -----------------------

def _wiener_deconv_3d(
    dy: np.ndarray, K: np.ndarray,
    *,
    lam0: float = 1e-3,
    lam_hf: float = 0.0,
    lam_exp: float = 2.0,
    taper_frac_t: float = 0.0,
    undo_advance: bool = True,
    use_gaussian: bool = True,          # default OFF
    gauss_sigma_frac: float = 0.1,      # sigma as fraction of Nyquist (0..1-ish)
    plot_filter: bool = True,          # default OFF
    plot_filter_nshow: int = 256,       # how many centered time samples to show
) -> np.ndarray:
    """
    3D Wiener deconvolution:
      Q̂ = H* / (|H|^2 + λ) · phase · taper · Y
    Optionally replaces λ(kt) HF shaping with a multiplicative Gaussian LPF in kt:
      G *= exp(-0.5 * (k_fold/sigma_bins)^2), with F(0)=1.
    """
    T, Nx, Ny = map(int, dy.shape)
    Lt, Kx, Ky = map(int, K.shape)
    cx, cy = _kernel_spatial_center(K)

    # FFT sizes for linear conv
    nfft_t = _next_pow2(T + Lt - 1)
    nfft_x = _next_pow2(Nx + Kx - 1)
    nfft_y = _next_pow2(Ny + Ky - 1)
    nfft = (nfft_t, nfft_x, nfft_y)

    # FFTs (only last axis is half-spectrum)
    Y = _rfftn3(_pad3(dy, nfft), nfft)
    H = _rfftn3(_padK(K,  nfft), nfft)
    H2 = (H.conj() * H).real

    # ---- kt grid (0..nfft_t-1), fold to [0..Nyquist] for symmetric shaping
    kt = np.arange(nfft_t, dtype=np.float64)
    fnyq = max(1, nfft_t // 2)
    k_fold = np.minimum(kt, nfft_t - kt)  # [0..fnyq]

    # ---- λ term (temporal-only ridge)
    if use_gaussian:
        # constant ridge only; Gaussian LPF applied multiplicatively below
        lam_vec = np.full(nfft_t, float(lam0), dtype=np.float64)
    else:
        # original HF-shaped ridge
        if lam_hf > 0.0:
            lam_vec = float(lam0) + float(lam_hf) * (k_fold / float(fnyq)) ** float(lam_exp)
        else:
            lam_vec = np.full(nfft_t, float(lam0), dtype=np.float64)

    lam_f = lam_vec[:, None, None]

    # ---- Optional temporal taper near Nyquist (symmetric)
    if taper_frac_t > 0.0:
        m = max(1, int(round(taper_frac_t * fnyq)))
        taper_1d = np.ones(nfft_t, dtype=np.float64)
        edge = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, m, endpoint=False)))
        taper_1d[fnyq - m:fnyq] *= edge
        # mirror side (guard for array end)
        hi0 = fnyq + 1
        hi1 = min(nfft_t, hi0 + m)
        taper_1d[hi0:hi1] *= edge[:(hi1 - hi0)]
        taper = taper_1d[:, None, None]
    else:
        taper = 1.0

    # ---- Phase ramps to undo kernel centering
    if undo_advance:
        phase_t = np.exp(-2j * np.pi * kt * float(Lt - 1) / float(nfft_t))[:, None, None]
    else:
        phase_t = 1.0

    kx = np.arange(nfft_x, dtype=np.float64)
    phase_x = np.exp(-2j * np.pi * kx * float(cx) / float(nfft_x))[None, :, None]

    ky = np.arange(nfft_y // 2 + 1, dtype=np.float64)
    phase_y = np.exp(-2j * np.pi * ky * float(cy) / float(nfft_y))[None, None, :]

    phase = phase_t * phase_x * phase_y

    # ---- Inverse filter
    denom = H2 + lam_f
    G = (H.conj() / np.maximum(denom, 1e-20)) * phase * taper

    # ---- Optional multiplicative Gaussian LPF in temporal frequency
    if use_gaussian:
        # sigma in bins (fraction of Nyquist)
        sigma_bins = max(1e-9, float(gauss_sigma_frac) * float(fnyq))
        F = np.exp(-0.5 * (k_fold / sigma_bins) ** 2)     # shape (nfft_t,)
        F[0] = 1.0
        G *= F[:, None, None]
        if plot_filter:
            import matplotlib.pyplot as plt

            # Frequency-domain plot (0..Nyquist)
            fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
            ax[0].plot(k_fold[:fnyq + 1], F[:fnyq + 1], lw=2)
            ax[0].set_title("Gaussian LPF in frequency (temporal bins)")
            ax[0].set_xlabel("k (0..Nyquist)")
            ax[0].set_ylabel("F(k)")
            ax[0].grid(alpha=0.25)

            # Time-domain impulse response via IFFT of the full symmetric spectrum
            F_full = F.astype(np.complex128)               # length nfft_t
            f_time = np.fft.ifft(F_full).real
            f_time_shift = np.fft.fftshift(f_time)

            # show a centered window
            nfft_t = int(F_full.size)
            n = np.arange(nfft_t) - (nfft_t // 2)
            half = int(max(8, min(plot_filter_nshow, nfft_t) // 2))
            mid = nfft_t // 2
            sl = slice(max(0, mid - half), min(nfft_t, mid + half))

            ax[1].plot(n[sl], f_time_shift[sl], lw=2)
            ax[1].set_title("Impulse response in time domain (ifft, fftshift)")
            ax[1].set_xlabel("time index n (centered)")
            ax[1].set_ylabel("f[n]")
            ax[1].grid(alpha=0.25)

            plt.tight_layout()
            plt.show()


    # ---- Apply and invert, then crop back to (T,Nx,Ny)
    QhatF = Y * G
    q_full = _irfftn3(QhatF, nfft).real
    return q_full[:T, :Nx, :Ny]




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


# ----------------------- Global time alignment ("model") -----------------------

def _estimate_global_shift_model_3d(dy_meas: np.ndarray, y_model: np.ndarray) -> int:
    """
    Estimate a single integer lag Δ over time that maximizes the global cross-correlation
    between dy_meas and y_model, aggregated over all pixels.
    """
    T = int(dy_meas.shape[0])
    # FFT along time only, aggregate cross-power over (x,y)
    Y_meas = np.fft.rfft(dy_meas, axis=0)
    Y_mod  = np.fft.rfft(y_model,  axis=0)
    cross_power = (Y_meas * np.conj(Y_mod)).sum(axis=(1, 2))  # shape (Kt,)
    r = np.fft.irfft(cross_power, n=T)
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


# ----------------------- Synthetic source builder -----------------------

def _inject_gaussian_source(
    T: int, Nx: int, Ny: int,
    pos: Tuple[int, int], t0: float, sigma: float, total_charge: float
) -> np.ndarray:
    """
    Return q[t,x,y] with a single Gaussian in time at spatial pixel 'pos'.
    Area under time curve equals total_charge.
    """
    x0, y0 = pos
    t = np.arange(T, dtype=np.float64)
    g = np.exp(-0.5 * ((t - float(t0)) / float(sigma)) ** 2)
    g /= max(g.sum(), 1e-20)
    g *= float(total_charge)
    q = np.zeros((T, Nx, Ny), dtype=np.float64)
    if 0 <= x0 < Nx and 0 <= y0 < Ny:
        q[:, x0, y0] += g
    return q


# ----------------------- Public demo -----------------------

def run_demo(
    kernel_or_K: Any,
    *,
    # grid / time
    T: int = 1024, Nx: int = 13, Ny: int = 13,
    # two sources
    pos_a: Tuple[int, int] = (7, 7), t0_a: float = 380.0, sigma_a: float = 4.0, Q_a: float = 100.0,
    pos_b: Tuple[int, int] = (7, 8), t0_b: float = 400.0, sigma_b: float = 4.0, Q_b: float = 100.0,
    # measurements
    threshold: float = 5.0, nsamples: int = 7, spacing: int = 16,
    densify_method: str = "linear", ramp_pre: Optional[int] = None,
    # deconvolution
    lam0: float = 1e-3, lam_hf: float = 0.0, lam_exp: float = 2.0,
    taper_frac_t: float = 0.0,
    align: str = "model", align_fractional: bool = True,
    clamp_nonneg: bool = False,
    deconv_domain: str = "dy",
    # plotting
    show: bool = True
) -> Dict[str, Any]:
    """
    Full 3D deconvolution demo over a 13x13 patch with two injected sources.
    Steps:
      1) Build q_true[t,x,y] with two temporal Gaussians at pos_a, pos_b.
      2) Forward with 3D kernel → dy_true (per-interval current); cumulative y_cum_true = cumsum_t(dy_true).
      3) Per-pixel sparse cumulative measurements after threshold crossing (nsamples, spacing).
      4) Densify per pixel → y_cum_dense; dy_meas = diff_t(y_cum_dense).
      5) 3D Wiener deconvolution to q_hat; optional "model" alignment (global Δ).
      6) Plots:
           • 1D: for pos_a and pos_b, show q_true, scatter(y_meas), q_hat.
           • 2D: maps of ∑_t q_true, last measured cumulative, ∑_t q_hat.
    """
    # --- kernel
    K = _to_numpy_kernel3d(kernel_or_K)
    Lt, Kx, Ky = K.shape
    if ramp_pre is None:
        ramp_pre = int(min(Lt, 2 * spacing))

    # --- true sources and forward model
    q_true = (
        _inject_gaussian_source(T, Nx, Ny, pos_a, t0_a, sigma_a, Q_a) +
        _inject_gaussian_source(T, Nx, Ny, pos_b, t0_b, sigma_b, Q_b)
    )
    
    q_a = _inject_gaussian_source(T, Nx, Ny, pos_a, t0_a, sigma_a, Q_a)
    q_b = _inject_gaussian_source(T, Nx, Ny, pos_b, t0_b, sigma_b, Q_b)
    
    Kb = K.copy()
    #Kb *= 3.0
    #Kb[:, 4, 4] = K[:, 4, 4]
    dy_true = (_forward3d_pre_kernel(q_a, Kb, (T, Nx, Ny))+_forward3d_pre_kernel(q_b, K, (T, Nx, Ny)))
    y_cum_true = np.cumsum(dy_true, axis=0)

    # --- sparse measurements per pixel (times & values)
    meas_times: List[List[np.ndarray]] = [[None for _ in range(Ny)] for _ in range(Nx)]
    meas_vals:  List[List[np.ndarray]] = [[None for _ in range(Ny)] for _ in range(Nx)]
    last_meas_value = np.zeros((Nx, Ny), dtype=np.float64)

    for ix in range(Nx):
        for iy in range(Ny):
            yc = y_cum_true[:, ix, iy]
            idx0 = _first_crossing(yc, threshold)
            if idx0 is None:
                # no measurements for this pixel
                meas_times[ix][iy] = np.array([], dtype=int)
                meas_vals[ix][iy]  = np.array([], dtype=np.float64)
                last_meas_value[ix, iy] = 0.0
                continue
            times = idx0 + spacing * np.arange(nsamples, dtype=int)
            times = times[(times >= 0) & (times < T)]
            vals = yc[times].astype(np.float64, copy=True)
            meas_times[ix][iy] = times
            meas_vals[ix][iy]  = vals
            last_meas_value[ix, iy] = float(vals[-1]) if vals.size else 0.0

    # --- densify cumulative & derivative from measurements
    y_cum_dense = np.zeros((T, Nx, Ny), dtype=np.float64)
    for ix in range(Nx):
        for iy in range(Ny):
            tarr = meas_times[ix][iy]
            varr = meas_vals[ix][iy]
            y_cum_dense[:, ix, iy] = _densify_cumulative(
                tarr, varr, T, method=densify_method, ramp_pre=ramp_pre
            )
    dy_meas = np.diff(y_cum_dense, axis=0, prepend=0.0)
    
    if deconv_domain not in ("dy", "cum"):
        raise ValueError("deconv_domain must be 'dy' or 'cum'")

    if deconv_domain == "dy":
        meas_for_deconv = dy_meas
        K_use = K
    else:
        # cumulative deconvolution: y_cum_dense ≈ q * cumsum(K)
        meas_for_deconv = y_cum_dense
        K_use = np.cumsum(K, axis=0)
    
    # --- 3D deconvolution
    q_hat = _wiener_deconv_3d(
        meas_for_deconv, K_use,
        lam0=lam0, lam_hf=lam_hf, lam_exp=lam_exp,
        taper_frac_t=taper_frac_t, undo_advance=True
    )
    if clamp_nonneg:
        q_hat = np.maximum(q_hat, 0.0)

    # --- "model" alignment (recommended)
    dt_applied = 0.0
    if align == "model":
        y_model = _forward3d_pre_kernel(q_hat, K, (T, Nx, Ny))
        tau = _estimate_global_shift_model_3d(dy_meas, y_model)
        print("Determined Shift : ", tau)
        if align_fractional:
            q_hat = _frac_shift_time_cube(q_hat, -float(tau))
            dt_applied += -float(tau)
        else:
            q_hat = np.roll(q_hat, shift=-int(tau), axis=0)
            dt_applied += -int(tau)

    # --- maps for visualization
    q_true_sum = q_true.sum(axis=0)     # (Nx,Ny)
    q_hat_sum  = q_hat.sum(axis=0)      # (Nx,Ny)
    meas_total = last_meas_value        # (Nx,Ny)

    # --- 1D plots for the two pixels of interest
    if show:
        t = np.arange(T)

        # 3 rows x 2 cols:
        # row0: charge (q_true vs q_hat)
        # row1: CURRENT INPUT to deconv (dy_true vs dy_meas)  <-- NEW
        # row2: cumulative (y_cum_true vs y_cum_dense + samples)
        fig, axs = plt.subplots(3, 2, figsize=(12.5, 10.2), sharex=True)
        pixels = [("Pixel A", pos_a), ("Pixel B", pos_b)]

        for j, (title, (px, py)) in enumerate(pixels):

            # --- Row 0: charge
            ax_q = axs[0, j]
            ax_q.plot(t, q_true[:, px, py], lw=2.0, label="True $q_{true}(t)$")
            ax_q.plot(t, q_hat[:,  px, py], lw=1.8, label="Deconv $\\hat q(t)$", alpha=0.95)
            ax_q.set_title(f"{title} @ ({px},{py})")
            ax_q.set_ylabel("charge")
            ax_q.grid(alpha=0.25)
            ax_q.legend(loc="best", fontsize=9)

            # --- Row 1: CURRENT (input to deconvolution)
            ax_i = axs[1, j]
            ax_i.plot(t, dy_true[:, px, py], ls="--", lw=1.7, alpha=0.95, label="$dy^{true}(t)$ (sim)")
            ax_i.plot(t, dy_meas[:, px, py], lw=1.7, alpha=0.95, label="$dy^{meas}(t)$ (from meas+dense)")
            ax_i.set_ylabel("induced current")
            ax_i.grid(alpha=0.25)
            ax_i.legend(loc="best", fontsize=9)

            # --- Row 2: cumulative (what you already had)
            ax_c = axs[2, j]
            ax_c.plot(t, y_cum_true[:, px, py], ls="--", lw=1.5, alpha=0.9, label="$y_{cum}^{true}(t)$")
            ax_c.plot(t, y_cum_dense[:, px, py], ls=":",  lw=1.5, alpha=0.9, label="$y_{cum}^{dense}(t)$")
            tm = meas_times[px][py]; vm = meas_vals[px][py]
            if tm.size:
                ax_c.scatter(tm, vm, s=36, marker="o", zorder=5, label="meas (cum)")
            ax_c.set_xlabel("time (samples)")
            ax_c.set_ylabel("cumulative")
            ax_c.set_ylim(bottom=0.0)
            ax_c.grid(alpha=0.25)
            ax_c.legend(loc="best", fontsize=9)

        plt.tight_layout()
        plt.show()

    return dict(
        K=K, Lt=Lt, Kx=Kx, Ky=Ky,
        q_true=q_true, dy_true=dy_true, y_cum_true=y_cum_true,
        meas_times=meas_times, meas_vals=meas_vals,
        y_cum_dense=y_cum_dense, dy_meas=dy_meas,
        q_hat=q_hat, q_true_sum=q_true_sum, q_hat_sum=q_hat_sum,
        meas_total=meas_total,
        meta=dict(
            grid=(T, Nx, Ny), sources=dict(pos_a=pos_a, t0_a=t0_a, pos_b=pos_b, t0_b=t0_b),
            threshold=threshold, nsamples=nsamples, spacing=spacing,
            lam0=lam0, lam_hf=lam_hf, lam_exp=lam_exp, taper_frac_t=taper_frac_t,
            align=align, align_fractional=align_fractional, time_shift_applied=dt_applied
        )
    )

