"""Eccentric BBH templates (SEOBNRv5EHM, (2,2) mode, aligned equal spins) from one ODE solve per base shape.

A base is (q, chi, e_b, anom_b) solved at total mass mtot_b from orbit-averaged 22 frequency fstart_b. GR has no
built-in mass scale, so the template of the same base at total mass M is h(f; M) = (M/Mb)^2 h_b(f M/Mb): a cheap
resample of the solved base. Heavier rescales start lower in frequency and at a later point of the orbit, so one
solve with fstart_b below the analysis band covers a whole family of masses and orbital phases (see
results/ecc_wavegen/REPORT.md).

Bank rows carry mass1, mass2, spin1z = spin2z, and the base columns ecc_b, anom_b, mtot_b, fstart_b (q comes from
mass1/mass2). Templates are face-on h+ in the frequency domain, peak of |h22| at t = 0, like PyCBC's FD templates.

Two consumers:
  coarse_generate_ecc  amp / phase-spline coefficients at fp_filter's coarse nodes x (imr_interp layout)
  fd_on_grid           complex FD templates on a uniform grid (bank placement, fitting factors)
"""
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import numpy as np

FS = 4096.0
PAD = int(os.environ.get("ECC_PAD", 8))
RTOL = 1e-7          # 2.2x faster than the default 1e-11, min match 0.9998 (ecc_wavegen REPORT)
BASE_COLS = ('ecc_b', 'anom_b', 'mtot_b', 'fstart_b')
_CACHE = OrderedDict()
_CACHE_MAX = int(os.environ.get('ECC_BASE_CACHE', 400))


def _patch_tolerance():
    import pyseobnr.models.SEOBNRv5EHM as EHM
    if getattr(EHM.SEOBNRv5EHM_opt, '_ecc_patched', False):
        return
    orig = EHM.SEOBNRv5EHM_opt._default_settings

    def patched():
        d = orig(); d.update(rtol=RTOL, atol=RTOL * 1e-1); return d
    EHM.SEOBNRv5EHM_opt._default_settings = staticmethod(patched)
    EHM.SEOBNRv5EHM_opt._ecc_patched = True


def base_key(q, chi, e, anom, mtot_b, fstart_b):
    return tuple(round(float(v), 10) for v in (q, chi, e, anom, mtot_b, fstart_b))


def solve_base(key):
    """Fine one-sided FD of the base: (f, amp, unwrapped phase), peak at t = 0, distance 1 Mpc, f >= 0.8 fstart_b."""
    import warnings, logging
    warnings.filterwarnings('ignore'); logging.getLogger('pyseobnr').setLevel(logging.ERROR)
    from pyseobnr.generate_waveform import GenerateWaveform
    _patch_tolerance()
    q, chi, e, anom, mb, fb = key
    p = dict(mass1=mb * q / (1 + q), mass2=mb / (1 + q), spin1z=chi, spin2z=chi, eccentricity=e, rel_anomaly=anom,
             f22_start=fb, f_ref=fb, deltaT=1 / FS, approximant='SEOBNRv5EHM', distance=1.0, inclination=0.0,
             phi_ref=0.0, mode_array=[(2, 2)], lmax_nyquist=1)
    hp, hc = GenerateWaveform(p).generate_td_polarizations()
    h = np.asarray(hp.data.data, np.float64)
    ipk = int(np.argmax(hp.data.data ** 2 + hc.data.data ** 2))
    r = min(int(4.0 / fb * FS), ipk // 2)                # taper the abrupt start over ~4 cycles
    h[:r] *= 0.5 * (1 - np.cos(np.pi * np.arange(r) / r))
    n = 1 << int(np.ceil(np.log2(PAD * len(h))))         # zero padding: phase step <= 2 pi / PAD per bin
    H = np.fft.rfft(h, n) / FS
    f = np.arange(len(H)) * FS / n
    H *= np.exp(2j * np.pi * f * ipk / FS)               # peak of |h22| to t = 0
    k = f >= 0.8 * fb
    return f[k], np.abs(H[k]).astype(np.float32), np.unwrap(np.angle(H[k])).astype(np.float64)


def _base(key):
    v = _CACHE.get(key)
    if v is None:
        v = _CACHE[key] = solve_base(key)
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
    return v


def _rows_keys(rows):
    out = []
    for r in rows:
        q = float(r['mass1']) / float(r['mass2'])
        out.append((base_key(q, r['spin1z'], r['ecc_b'], r['anom_b'], r['mtot_b'], r['fstart_b']),
                    float(r['mass1']) + float(r['mass2'])))
    return out


def eval_rows(keys_m, fpts, distance=1.0):
    """amp, phase of each (key, M) at frequencies fpts (same for all rows). Zero amplitude outside the base."""
    amp = np.zeros((len(keys_m), len(fpts))); ph = np.zeros((len(keys_m), len(fpts)))
    for j, (key, M) in enumerate(keys_m):
        fb, ab, pb = _base(key)
        s = M / key[4]
        g = fpts * s
        amp[j] = np.interp(g, fb, ab, left=0.0, right=0.0) * s ** 2 / distance
        ph[j] = np.interp(g, fb, pb)
    return amp, ph


def _coarse_worker(args):
    from scipy.interpolate import CubicSpline
    idx, keys_m, x, distance = args
    amp, ph = eval_rows(keys_m, x, distance)
    coef = np.stack([CubicSpline(x, p).c for p in ph])
    step = float(np.abs(np.diff(ph, axis=1)).max()) if len(ph) else 0.0
    return idx, amp, coef, step


_POOL = None


def _pool(procs):
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        _POOL = ProcessPoolExecutor(procs, mp_context=mp.get_context('spawn'))
    return _POOL


def coarse_generate_ecc(rows, x, flow, procs=7, distance=1.0):
    """imr_interp.coarse_generate for eccentric rows. Rows of one base always go to the same worker, so each base
    is solved once per worker while it stays in that worker's cache (rows are tiled by tau0, a family spans tiles)."""
    keys_m = _rows_keys(rows)
    groups = [[] for _ in range(procs)]
    for i, (k, M) in enumerate(keys_m):
        groups[hash(k) % procs].append(i)
    # a fixed worker per base needs one task queue per worker: emulate with one task per group, run in order
    futs = [_pool(procs).submit(_coarse_worker, (g, [keys_m[i] for i in g], x, distance)) for g in groups if g]
    amp = np.zeros((len(rows), len(x))); coef = np.zeros((len(rows), 4, len(x) - 1)); step = 0.0
    for fu in futs:
        idx, a, c, s = fu.result()
        amp[idx] = a; coef[idx] = c; step = max(step, s)
    return amp, coef, step


def fd_on_grid(keys_m, f):
    """Complex FD templates (rows) on frequencies f."""
    amp, ph = eval_rows(keys_m, f)
    return amp * np.exp(1j * ph)


def e_at(f_to, e_from, f_from):
    """Leading-order secular eccentricity evolution e ~ f^(-19/18) (only for labelling templates, not physics)."""
    return e_from * (f_from / f_to) ** (19 / 18)
