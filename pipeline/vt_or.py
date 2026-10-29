#!/usr/bin/env python
"""Found-count VT on the frozen eccentric injections: aligned alone, eccentric bank alone, and aligned OR eccentric at
FAR/2 each (precessing-lane method). An injection is found by a ranking when a foreground candidate within --dt s of its
H1 arrival time has far_incl_per_yr <= the threshold. VT = sum of vt_weight over found injections inside the coincident
segments. Ratio and OR-minus-aligned CIs: bootstrap over injections (paired). Also writes paired outcomes per group
(both / aligned only / eccentric only / neither) and a per-injection table (row, tc, mtot, e20, weight, both FARs).

Usage: vt_or.py INJ.hdf ALIGNED_inj.hdf ECC_inj.hdf COINC_SEGMENTS.txt OUT.json [--far 1 0.1] [--mtot 45 100]
"""
import argparse, json
import numpy as np, h5py

ap = argparse.ArgumentParser()
ap.add_argument('inj'); ap.add_argument('al'); ap.add_argument('ecc'); ap.add_argument('segs'); ap.add_argument('out')
ap.add_argument('--far', type=float, nargs='+', default=[1.0, 0.1]); ap.add_argument('--dt', type=float, default=0.1)
ap.add_argument('--mtot', type=float, nargs=2, default=[45.0, 100.0], help='eccentric bank mass range (detector frame)')
ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=3)
a = ap.parse_args()

with h5py.File(a.inj) as f:
    I = {k: f[k][:] for k in ('tc', 'ra', 'dec', 'mass1', 'mass2', 'eccentricity', 'vt_weight')}
try:
    from pycbc.detector import Detector
    tH = I['tc'] + Detector('H1').time_delay_from_earth_center(I['ra'], I['dec'], I['tc'])
except ImportError:                      # local fallback: |delay| < 21 ms, inside --dt
    tH = I['tc']
seg = np.loadtxt(a.segs, ndmin=2)
live = np.zeros(len(tH), bool)
for s, e in seg:
    live |= (I['tc'] >= s) & (I['tc'] < e)


def best_far(path):
    with h5py.File(path) as f:
        ct = f['foreground/H1_time'][:]; cf = f['foreground/far_incl_per_yr'][:]
    o = np.argsort(ct); ct, cf = ct[o], cf[o]
    lo, hi = np.searchsorted(ct, tH - a.dt), np.searchsorted(ct, tH + a.dt)
    return np.array([cf[l:h].min() if h > l else np.inf for l, h in zip(lo, hi)])


F_al, F_ecc = best_far(a.al), best_far(a.ecc)
M = I['mass1'] + I['mass2']; e = I['eccentricity']; w = I['vt_weight']
inbox = live & (M >= a.mtot[0]) & (M <= a.mtot[1])
groups = {}
for name, m in (('', live), ('box ', inbox)):   # control e20 < 0.15 (aligned bank loses nothing there), then target bins
    for lab, lo, hi in (('all', 0, 1), ('e20<0.15', 0, 0.15), ('e20>=0.15', 0.15, 1), ('e20 0.15-0.25', 0.15, 0.25),
                        ('e20 0.25-0.35', 0.25, 0.35), ('e20 0.35-0.45', 0.35, 1)):
        groups[name + lab] = m & (e >= lo) & (e < hi)
rng = np.random.default_rng(a.seed)
res = dict(args=vars(a), n_live=int(live.sum()), n_total=len(tH), groups={})
for x in a.far:
    found = dict(aligned=F_al <= x, eccentric=F_ecc <= x, OR=np.fmin(F_al, F_ecc) <= x / 2)
    for g, m in groups.items():
        idx = np.flatnonzero(m)
        vt = {k: float(np.sum(w[idx] * v[idx])) for k, v in found.items()}
        r = {}
        b = rng.integers(0, len(idx), (a.boot, len(idx))) if len(idx) else np.zeros((a.boot, 0), int)
        d = (w[idx][b] * (found['OR'][idx][b].astype(float) - found['aligned'][idx][b])).sum(1)
        r['OR_minus_aligned'] = dict(diff=vt['OR'] - vt['aligned'], p5=float(np.percentile(d, 5)) if len(idx) else 0.0,
                                     p95=float(np.percentile(d, 95)) if len(idx) else 0.0)
        for k in ('eccentric', 'OR'):
            b = rng.integers(0, len(idx), (a.boot, len(idx))) if len(idx) else np.zeros((a.boot, 0), int)
            num = (w[idx][b] * found[k][idx][b]).sum(1); den = (w[idx][b] * found['aligned'][idx][b]).sum(1)
            rat = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
            r[k] = dict(ratio=vt[k] / vt['aligned'] if vt['aligned'] > 0 else None,
                        p5=float(np.nanpercentile(rat, 5)), p95=float(np.nanpercentile(rat, 95)))
        fa, fe = found['aligned'][idx], found['eccentric'][idx]
        res['groups'][f'far {x:g} {g}'] = dict(n=len(idx), n_found={k: int(v[idx].sum()) for k, v in found.items()},
                                               paired=dict(both=int((fa & fe).sum()), aligned_only=int((fa & ~fe).sum()),
                                                           eccentric_only=int((~fa & fe).sum()), neither=int((~fa & ~fe).sum())),
                                               vt=vt, vt_ratio_vs_aligned=r)
res['per_injection'] = dict(columns=['row', 'tc', 'mtot', 'e20', 'vt_weight', 'live', 'far_aligned', 'far_eccentric'],
                           rows=[[i, float(I['tc'][i]), float(M[i]), float(e[i]), float(w[i]), bool(live[i]),
                                  float(F_al[i]) if np.isfinite(F_al[i]) else None, float(F_ecc[i]) if np.isfinite(F_ecc[i]) else None]
                                 for i in range(len(tH))])
json.dump(res, open(a.out, 'w'), indent=1)
for k, v in res['groups'].items():
    r = v['vt_ratio_vs_aligned']
    print(f"{k:28s} n={v['n']:4d} found al/ecc/OR {v['n_found']['aligned']}/{v['n_found']['eccentric']}/{v['n_found']['OR']}  "
          f"OR/al {r['OR']['ratio'] or float('nan'):.3f} [{r['OR']['p5']:.3f}, {r['OR']['p95']:.3f}]  "
          f"ecc/al {r['eccentric']['ratio'] or float('nan'):.3f}")
