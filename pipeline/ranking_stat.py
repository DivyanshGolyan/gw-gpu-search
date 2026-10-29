"""Coincidence ranking statistic for coinc_rank.py, modelled on PyCBC's O3/O4 statistic
(Davies et al. 2020, arXiv:2002.08291), using only what the batched/multi_filter peak files carry.

The score of an H1+L1 coincidence is a log likelihood ratio, "how much more likely is this pair under
'astrophysical signal' than under 'chance noise'":

  lnL = -ln n_H(newsnr_H) - ln n_L(newsnr_L)      noise: how common is a trigger this loud, in THIS template
        + ln p_sig(dt, dphi, r) - ln p_noise       H1/L1 consistency: time lag, phase lag, amplitude ratio
        + 3 ln(sigma_net / sigma_ref)              optional: more sensitive template/time -> more signals
        - ln(local trigger-rate factor)            optional: noisy stretches count as noisier

Pieces:
  noise fits    For each detector and template, triggers above a threshold are fitted with an exponential
                tail n(x) = rate * alpha * exp(-alpha (x - thr)). Fits are pooled over the K templates
                nearest in duration (PyCBC smooths over duration too), so a short high-mass template that
                rings on glitches gets its own flat tail and therefore a large noise density.
  phasetd       Monte Carlo of isotropic sources seen by H1 and L1 (antenna patterns from pycbc.detector),
                with SNR measurement noise, histogrammed in (dt, dphi, r) where r = ln(snr_L/snr_H) minus
                ln(sigma_L/sigma_H). Noise is taken as uniform over the same box.
  local rate    Count of loud (newsnr >= 6) triggers across the whole bank within +-W s, excluding +-1 s
                around the trigger itself, relative to the detector median: a proxy for PyCBC's psdvar.
"""
import os
import numpy as np

KAPPA = -1j          # h_cross = KAPPA * h_plus for face-on (2,2) IMRPhenomD in pycbc's convention (checked numerically)


def template_durations(bank_path, f_low=20.0):
    import h5py
    from pycbc import pnutils
    with h5py.File(bank_path, 'r') as f:
        m1, m2, s1, s2 = (f[k][:] for k in ('mass1', 'mass2', 'spin1z', 'spin2z'))
    return np.array([pnutils.get_imr_duration(a, b, c, d, f_low, approximant='IMRPhenomD')
                     for a, b, c, d in zip(m1, m2, s1, s2)])


def fit_noise(tid, rank, ntmp, live, thr=6.0, order=None, pool=50):
    """Per-template exponential tail fit, pooled over `pool` neighbours in `order` (template ids sorted by
    duration; may be a subset, the others get no fit). Returns (ln_rate_alpha, alpha), each of length ntmp:
    ln n(x) = ln_rate_alpha - alpha (x - thr)."""
    m = rank >= thr
    n = np.bincount(tid[m], minlength=ntmp).astype(float)
    s = np.bincount(tid[m], weights=rank[m] - thr, minlength=ntmp)
    order = np.arange(ntmp) if order is None else order
    # running sums over a centred window of `pool` templates in duration order
    cn = np.concatenate([[0], np.cumsum(n[order])]); cs = np.concatenate([[0], np.cumsum(s[order])])
    k = len(order)
    i = np.arange(k); lo = np.clip(i - pool // 2, 0, max(k - pool, 0)); hi = lo + min(pool, k)
    pn, ps = cn[hi] - cn[lo], cs[hi] - cs[lo]
    alpha_o = (pn + 1) / (ps + 1 / 6.0)            # weak prior: alpha ~ 6 with one pseudo-trigger
    rate_o = (pn + 1) / (hi - lo) / live
    alpha = np.full(ntmp, 6.0); rate = np.full(ntmp, 1e-30)
    alpha[order] = alpha_o; rate[order] = rate_o
    return np.log(rate * alpha), alpha


def ln_noise(tid, rank, fit, thr=6.0):
    lra, alpha = fit
    return lra[tid] - alpha[tid] * (rank - thr)


def local_rate(t, rank, win=32.0, excl=1.0, loud=6.0):
    """Relative local rate of loud triggers (any template) around each trigger time, excluding +-excl s."""
    tl = np.sort(t[rank >= loud])
    def count(a, b):
        return np.searchsorted(tl, b, 'right') - np.searchsorted(tl, a, 'left')
    c = count(t - win, t + win) - count(t - excl, t + excl)
    rate = (c + 1) / (2 * (win - excl))
    med = np.median(rate)
    return rate / med


class PhaseTD:
    """Signal density of (dt, dphi, r) for H1-L1 coincidences, normalised against a uniform noise density."""
    def __init__(self, window=0.012, path=None, n=4_000_000, f_eff=40.0, seed=7, ifos=('H1', 'L1'),
                 nt=24, nphi=32, rmax=2.5, nr=25):
        self.window, self.rmax = window, rmax
        self.te = np.linspace(-window, window, nt + 1); self.pe = np.linspace(-np.pi, np.pi, nphi + 1)
        self.re = np.linspace(-rmax, rmax, nr + 1)
        if path and os.path.exists(path):
            d = np.load(path); self.lnp = d['lnp']; return
        rng = np.random.default_rng(seed)
        from pycbc.detector import Detector
        D1, D2 = Detector(ifos[0]), Detector(ifos[1])
        ra = rng.uniform(0, 2 * np.pi, n); dec = np.arcsin(rng.uniform(-1, 1, n))
        psi = rng.uniform(0, np.pi, n); ci = rng.uniform(-1, 1, n); t0 = 1372349000
        g = []
        for D in (D1, D2):
            fp, fc = D.antenna_pattern(ra, dec, psi, t0)
            g.append(fp * (1 + ci ** 2) / 2 + KAPPA * fc * ci)
        dt_true = D2.time_delay_from_detector(D1, ra, dec, t0)          # t_L - t_H
        norm = np.sqrt(np.abs(g[0]) ** 2 + np.abs(g[1]) ** 2)
        rho = 6.0 * rng.uniform(0, 1, n) ** (-1 / 3.0)                  # network SNR, uniform in volume
        ph0 = np.exp(1j * rng.uniform(0, 2 * np.pi, n))
        z = [rho * gi / norm * ph0 + (rng.normal(size=n) + 1j * rng.normal(size=n)) for gi in g]
        a = [np.abs(zi) for zi in z]
        ok = (a[0] > 5) & (a[1] > 5)
        st = [1.0 / (2 * np.pi * f_eff * np.maximum(ai, 5)) for ai in a]
        dt = dt_true + rng.normal(size=n) * np.hypot(st[0], st[1]) + rng.uniform(-0.5, 0.5, n) / 2048
        dphi = np.angle(z[1] * np.conj(z[0]))
        r = np.log(a[1] / a[0])
        ok &= np.abs(dt) <= window
        H, _ = np.histogramdd(np.c_[dt[ok], dphi[ok], np.clip(r[ok], -rmax, rmax - 1e-9)],
                              bins=(self.te, self.pe, self.re))
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, 0.7, mode=('nearest', 'wrap', 'nearest')) + 0.5
        vol = np.diff(self.te)[:, None, None] * np.diff(self.pe)[None, :, None] * np.diff(self.re)[None, None, :]
        dens = H / (H.sum()) / vol
        noise = 1.0 / ((2 * window) * (2 * np.pi) * (2 * rmax))
        self.lnp = np.log(dens / noise)
        if path:
            np.savez(path, lnp=self.lnp)

    def __call__(self, dt, dphi, r):
        it = np.clip(np.searchsorted(self.te, dt) - 1, 0, len(self.te) - 2)
        ip = np.clip(np.searchsorted(self.pe, np.angle(np.exp(1j * dphi))) - 1, 0, len(self.pe) - 2)
        ir = np.clip(np.searchsorted(self.re, r) - 1, 0, len(self.re) - 2)
        out = self.lnp[it, ip, ir]
        # beyond the amplitude-ratio range: keep falling, 3 nats per unit of ln ratio
        return out - 3.0 * np.clip(np.abs(r) - self.rmax, 0, None)
