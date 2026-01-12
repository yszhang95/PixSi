# single_pixel_time_demo.py
from __future__ import annotations
from typing import Any, Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

ArrayLike = Union[np.ndarray, "torch.Tensor"]

# ---------------- I/O helpers ----------------
def _to_numpy_center_trace(kernel_or_k1d: Any) -> np.ndarray:
    try:
        import torch
    except Exception:
        torch = None
    if hasattr(kernel_or_k1d, "K_delta"):
        K = kernel_or_k1d.K_delta
        if torch is not None and isinstance(K, torch.Tensor):
            Knp = K.detach().cpu().numpy()
        else:
            Knp = np.array(K)
        cx = Knp.shape[1] // 2
        cy = Knp.shape[2] // 2
        k_pre = Knp[:, cx, cy].astype(np.float64, copy=True)
    else:
        k_pre = np.asarray(kernel_or_k1d, dtype=np.float64).ravel().copy()
    if k_pre.ndim != 1 or k_pre.size < 2:
        raise ValueError("Expected 1D kernel with length >= 2.")
    return k_pre

def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def _first_crossing(y: np.ndarray, thr: float) -> Optional[int]:
    idx = np.flatnonzero(y >= thr)
    return int(idx[0]) if idx.size else None

# ---------------- Forward model ----------------
def _forward_pre_kernel(q: np.ndarray, k_pre: np.ndarray, T: int) -> np.ndarray:
    """Pre-kernel that ENDS at the charge time (advance by Lt-1)."""
    Lt = int(k_pre.size)
    y_full = np.convolve(q, k_pre, mode="full")          # len T+Lt-1
    return y_full[Lt - 1 : Lt - 1 + T]                   # len T

# ---------------- Densify cumulative ----------------
def _densify_cumulative(
    times: np.ndarray,
    values: np.ndarray,
    T: int,
    *,
    method: str = "linear",
    ramp_pre: int = 0
) -> np.ndarray:
    """
    Build dense cumulative from sparse samples.
    method='linear' connects adjacent points with lines.
    ramp_pre>0: add a linear ramp from t1-ramp_pre to t1 to avoid a vertical jump at the first sample.
    """
    y = np.zeros(T, dtype=np.float64)
    if times.size == 0:
        return y

    if method not in ("linear", "zoh"):
        raise ValueError("method must be 'linear' or 'zoh'")

    t1 = int(times[0])
    v1 = float(values[0])

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

    # method == 'linear'
    t = np.arange(T)

    # Start with simple linear between measured points; hold after last
    y[:] = np.interp(t, times, values, left=0.0, right=values[-1])

    # Pre-ramp instead of a vertical step at the first measurement
    if ramp_pre > 0:
        a = max(0, t1 - ramp_pre)
        # ramp from 0 at 'a' to v1 at 't1' (inclusive)
        if a < t1:
            seg_len = t1 - a
            if seg_len > 0:
                ramp = np.linspace(0.0, v1, seg_len + 1)
                y[a:t1 + 1] = ramp
        # keep strictly zero before 'a'
        y[:a] = 0.0
    else:
        y[:t1] = 0.0

    return y

# ---------------- Deconvolution ----------------
def _wiener_deconv_1d_with_phase(
    dy: np.ndarray,
    k_pre: np.ndarray,
    *,
    lam0: float = 1e-2,
    lam_hf: float = 5e-2,     # ignored when use_gaussian=True
    lam_exp: float = 2.0,     # ignored when use_gaussian=True
    advance_m: int,
    taper_frac: float = 0.1,
    # NEW:
    use_gaussian: bool = False,
    gauss_sigma_frac: float | None = None,   # fraction of Nyquist, e.g. 0.2
) -> np.ndarray:
    """
    Frequency-domain Wiener inverse with:
      • phase term to undo the forward advance by m=Lt-1
      • EITHER classic λ(f) shaping (default),
        OR multiply by a Gaussian F(ω) with F(0)=1 when use_gaussian=True.
    """
    Nt = int(dy.size)
    Lt = int(k_pre.size)
    nfft = _next_pow2(Nt + Lt - 1)

    Y = np.fft.rfft(dy, n=nfft)
    H = np.fft.rfft(k_pre, n=nfft)
    H2 = (H.conj() * H).real

    # rfft bins 0..nfft/2
    k = np.arange(nfft // 2 + 1, dtype=np.float64)
    fnyq = max(1, nfft // 2)

    # Phase to undo forward advance by m samples
    phase = np.exp(-2j * np.pi * k * float(advance_m) / float(nfft))

    if not use_gaussian:
        # Original λ(f) shaping
        fscaled = (k / fnyq) ** float(lam_exp)
        lam_f = float(lam0) + float(lam_hf) * fscaled
        denom = H2 + lam_f
        G = (H.conj() / np.maximum(denom, 1e-20)) * phase
    else:
        # Constant ridge + Gaussian multiplier F(ω)
        denom = H2 + float(lam0)
        G = (H.conj() / np.maximum(denom, 1e-20)) * phase

        # Gaussian width as fraction of Nyquist → convert to bins
        if gauss_sigma_frac is None or gauss_sigma_frac <= 0.0:
            # safe default ~ gentle LP
            gauss_sigma_frac = 0.2
        sigma_bins = max(1e-9, gauss_sigma_frac * fnyq)
        F = np.exp(-0.5 * (k / sigma_bins) ** 2)
        F[0] = 1.0  # enforce exactly 1 at DC
        G *= F

    # Optional gentle HF taper near Nyquist
    if taper_frac > 0.0:
        nb = G.size
        m = max(1, int(round(taper_frac * nb)))
        x = np.linspace(0.0, np.pi, m, endpoint=False)
        taper = 0.5 * (1.0 + np.cos(x))
        w = np.ones(nb, dtype=np.float64)
        w[-m:] *= taper
        G *= w

    Q_hat = Y * G
    q_full = np.fft.irfft(Q_hat, n=nfft)
    return q_full[:Nt].astype(np.float64, copy=False)


def _estimate_shift_xcorr(x: np.ndarray, y: np.ndarray) -> int:
    """
    Estimate integer-sample lag Δ maximizing cross-correlation r_xy[Δ].
    Positive Δ means x is ahead of y by Δ samples (i.e., y should be advanced).
    """
    n = 1 << int(np.ceil(np.log2(x.size + y.size - 1)))
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    r = np.fft.irfft(X * np.conj(Y), n=n)          # cross-corr via FFT
    lags = np.arange(n) - (y.size - 1)             # [- (Ly-1) .. (n-Ly)]
    tau = int(lags[np.argmax(r)])
    return tau

def _time_centroid(x: np.ndarray) -> float:
    """Time centroid using nonnegative weights (clip negatives)."""
    w = np.maximum(x.astype(np.float64, copy=False), 0.0)
    s = w.sum()
    if s <= 0.0:
        return float(x.size / 2.0)
    t = np.arange(x.size, dtype=np.float64)
    return float((t * w).sum() / s)

def _estimate_shift_centroid_dy_k(dy: np.ndarray, k_pre: np.ndarray) -> float:
    """
    centroid(dy) ≈ centroid(q) + centroid(k).
    Δ ≈ centroid(dy) - centroid(k). (fractional allowed)
    """
    return _time_centroid(dy) - _time_centroid(k_pre)

def _frac_shift_real(x: np.ndarray, delta: float) -> np.ndarray:
    """
    Fractional circular shift by 'delta' samples using rFFT phase ramp.
    For small |delta| (≲ few samples) with adequate padding this approximates a linear shift well.
    """
    n = 1 << int(np.ceil(np.log2(x.size)))
    X = np.fft.rfft(x, n=n)
    k = np.arange(X.size, dtype=np.float64)
    phase = np.exp(+2j * np.pi * k * (float(delta) / float(n)))
    y = np.fft.irfft(X * phase, n=n)[:x.size]
    return y


# ---------------- Public demo ----------------
def run_demo(
    kernel_or_k1d: Any,
    *,
    T: int = 1024,
    t0: int = 1400,
    sigma: float = 12.0,
    total_charge: float = 100.0,
    threshold: float = 5.0,
    spacing: int = 16,
    # deconvolution knobs
    lam0: float = 1e-2,      # base Tikhonov
    lam_hf: float = 5e-2,    # extra HF damping
    lam_exp: float = 2.0,    # HF exponent (2≈quadratic)
    taper_frac: float = 0.1, # extra safety in the top band
    # densification knob
    ramp_pre: Optional[int] = None,  # None -> auto=min(Lt, 2*spacing)
    interp: str = "linear",
    clamp_nonneg: bool = False,
    show: bool = True,
    use_gaussian: bool = True,
    gauss_sigma_frac: float = 0.05,
    align: str = "off",             # "off", "dy_k", "centroid", "model"
    align_fractional: bool = False  # use sub-sample phase shift
) -> Dict[str, np.ndarray]:

    k_pre = _to_numpy_center_trace(kernel_or_k1d)
    Lt = k_pre.size
    advance_m = Lt - 1
    t = np.arange(T, dtype=np.float64)

    # True charge: Gaussian at t0 (area = total_charge)
    g = np.exp(-0.5 * ((t - float(t0)) / float(sigma)) ** 2)
    #g2 = np.exp(-0.5 * ((t + 100 - float(t0)) / float(sigma)) ** 2)
    #g=g+g2
    g /= max(g.sum(), 1e-20)
    
    q_true = total_charge * g

    # Forward: induced increments and cumulative
    dy = _forward_pre_kernel(q_true, k_pre, T)
    y_cum = np.cumsum(dy)

    # Measurements: first crossing + 4 more every 'spacing'
    idx0 = _first_crossing(y_cum, threshold)
    if idx0 is None:
        scale = (threshold * 1.25) / (np.max(y_cum) + 1e-20)
        q_true *= scale
        dy = _forward_pre_kernel(q_true, k_pre, T)
        y_cum = np.cumsum(dy)
        idx0 = _first_crossing(y_cum, threshold)

    times = [idx0 + i * spacing for i in range(7)]
    times = np.array([tt for tt in times if 0 <= tt < T], dtype=int)
    y_meas = y_cum[times].copy()

    # Densify cumulative with a PRE-RAMP to avoid a jump at first sample
    if ramp_pre is None:
        ramp_pre = int(min(Lt, 2 * spacing))
    y_cum_dense = _densify_cumulative(
        times, y_meas, T, method=interp, ramp_pre=ramp_pre
    )

    # Per-interval current from densified cumulative
    dy_meas = np.diff(y_cum_dense, prepend=0.0)
    #plt.plot(y_cum_dense)
    #plt.plot(dy_meas)
    #plt.title("Calculated $dy_{meas}$")
    #plt.show()
    # Deconvolution: Wiener with freq-dependent lambda + phase undo
    
    dt_applied = 0.0  # cumulative shift applied to dy (pre) or q_hat (post)

    if align == "dy_k":
        tau = _estimate_shift_xcorr(dy_meas, k_pre)  # integer samples
        if align_fractional:
            dy_meas = _frac_shift_real(dy_meas, -float(tau))
            dt_applied += -float(tau)
        else:
            dy_meas = np.roll(dy_meas, -int(tau))
            dt_applied += -int(tau)

    elif align == "centroid":
        delta = _estimate_shift_centroid_dy_k(dy_meas, k_pre)  # may be fractional
        if align_fractional:
            dy_meas = _frac_shift_real(dy_meas, -float(delta))
            dt_applied += -float(delta)
        else:
            dy_meas = np.roll(dy_meas, -int(round(delta)))
            dt_applied += -int(round(delta))
    
    q_hat = _wiener_deconv_1d_with_phase(
        dy_meas, k_pre,
        lam0=lam0, lam_hf=lam_hf, lam_exp=lam_exp,
        advance_m=advance_m, taper_frac=taper_frac,
        use_gaussian=use_gaussian, gauss_sigma_frac=gauss_sigma_frac
    )
    if clamp_nonneg:
        q_hat = np.maximum(q_hat, 0.0)

    if align == "model":
    # build model measurement from q_hat
        y_model = _forward_pre_kernel(q_hat, k_pre, T=dy_meas.size)
        tau2 = _estimate_shift_xcorr(dy_meas, y_model)
        if align_fractional:
            q_hat = _frac_shift_real(q_hat, -float(tau2))
            dt_applied += -float(tau2)
        else:
            q_hat = np.roll(q_hat, -int(tau2))
            dt_applied += -int(tau2)

    if show:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, q_true, lw=2, label="True charge $q_{true}[t]$")
        ax.plot(t, y_cum, lw=1.8, label="Cumulative current $y_{cum}[t]$", alpha=0.9)
        ax.scatter(times, y_meas, s=40, label="Measurements (cumulative)", zorder=5)
        ax.plot(t, q_hat, lw=2, label="Deconvolved $\\hat q[t]$", alpha=0.95)
        ax.set_xlabel("time (samples)")
        #ax.set_title("Single-pixel: ramped-linear densify + frequency-shaped Wiener deconvolution")
        ax.legend(loc="best"); ax.grid(True, alpha=0.25)
        plt.tight_layout(); plt.show()

    return dict(
        t=t, k_pre=k_pre, q_true=q_true, dy=dy, y_cum=y_cum,
        meas_times=times, meas_values=y_meas,
        y_cum_dense=y_cum_dense, dy_meas=dy_meas,
        q_hat=q_hat, lam0=lam0, lam_hf=lam_hf, lam_exp=lam_exp,
        taper_frac=taper_frac, ramp_pre=ramp_pre, Lt=Lt
    )


def run_perfect_reco_check(
    kernel_or_k1d,
    *,
    T: int = 1024,
    t0: int = 380,
    sigma: float = 4.0,
    total_charge: float = 100.0,
    lam0: float = 1e-3,
    taper_frac: float = 0.0,
    use_gaussian: bool = True,
    gauss_sigma_frac: float = 0.20,
    show: bool = True,
):
    """
    Identity test: q_true --(conv by k_pre)--> dy  --(our deconv)--> q_hat  ~ q_true
    """
    k_pre = _to_numpy_center_trace(kernel_or_k1d)
    Lt = int(k_pre.size)
    advance_m = Lt - 1

    t = np.arange(T, dtype=np.float64)
    g = np.exp(-0.5 * ((t - float(t0)) / float(sigma)) ** 2)
    g /= max(g.sum(), 1e-20)
    q_true = total_charge * g

    # Forward with the *non-cumulative* response (per-interval center trace)
    dy = _forward_pre_kernel(q_true, k_pre, T)

    # Deconvolve directly from dy (no cumulative, no densify)
    q_hat = _wiener_deconv_1d_with_phase(
        dy, k_pre,
        lam0=lam0, lam_hf=0.0, lam_exp=2.0,
        advance_m=advance_m, taper_frac=taper_frac,
        use_gaussian=use_gaussian, gauss_sigma_frac=gauss_sigma_frac
    )

    # Report error
    denom = np.max(np.abs(q_true)) + 1e-12
    rel_rmse = np.sqrt(np.mean((q_true - q_hat) ** 2)) / denom
    print(f"[perfect-reco] relRMSE={rel_rmse:.3e}  "
          f"(lam0={lam0:g}, gauss_sigma_frac={gauss_sigma_frac:g}, taper={taper_frac:g})")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(t, q_true, lw=2, label="True $q_{true}[t]$")
        ax.plot(t, q_hat,  lw=1.8, label="Deconvolved $\\hat q[t]$", alpha=0.95)
        ax.set_xlabel("time (samples)")
        ax.legend(loc="best"); ax.grid(True, alpha=0.25)
        ax.set_title("Perfect inversion check (no densify / no thresholding)")
        plt.tight_layout(); plt.show()

    return dict(t=t, q_true=q_true, q_hat=q_hat, dy=dy, rel_rmse=rel_rmse)
