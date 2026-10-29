#!/usr/bin/env python
"""Candidate list across weeks and layers: every foreground coincidence with exclusive FAR below --far-thr,
cross-matched with known O4 events and alerts (fetch_catalogs.py CSV). Candidates within --win s of each
other in different layers are one candidate (best FAR kept, layers listed).

A candidate is KNOWN when a catalog event or a public alert lies within --win s of its H1 time, else NEW.
Each NEW candidate gets data-quality context, all from local inputs plus the GWOSC timeline:
  IFO_FLAG_ok      the candidate passes GWOSC CBC_CAT1, CBC_CAT2 and NO_CBC_HW_INJ (False = vetoed / hardware
                   injection). Measured 2026-10-28: in the O4a and O4b open data these flags remove 0 s beyond
                   DATA, and GWOSC states there were no CBC hardware injections in O4a/O4b observing time.
  gate_dt          distance to the nearest autogate in either detector (from --gates), s
  env_H1/env_L1    number of single-detector peaks with SNR >= 8 within +-8 s, excluding +-0.2 s around the
                   candidate (glitchy neighbourhood), from --peaks if given
  short            template duration < 0.2 s (high-mass, glitch-like)
Usage:
  candidates.py --ranked base:wk01=rank/s.hdf [--ranked ...] --known o4_known.csv --out DIR
                [--gates wk01=gates.txt ...] [--peaks wk01:H1=... --peaks wk01:L1=...] [--cacert BUNDLE]
"""
import argparse, csv, json, os, subprocess
from collections import defaultdict
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument('--ranked', action='append', required=True, help='LAYER:WEEK=path to coinc_rank search output')
ap.add_argument('--known', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--far-thr', type=float, default=1.0, help='per yr, exclusive FAR')
ap.add_argument('--win', type=float, default=1.0)
ap.add_argument('--gates', action='append', default=[], help='WEEK=text file: gps ifo')
ap.add_argument('--peaks', action='append', default=[], help='WEEK:IFO=merged peaks hdf')
ap.add_argument('--cacert', default=None)
ap.add_argument('--no-dq', action='store_true', help='skip the GWOSC timeline queries')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---- candidates from every ranked output
rows = []
for spec in a.ranked:
    lw, path = spec.split('=', 1)
    layer, week = lw.split(':')
    with h5py.File(path) as f:
        g = f['foreground']
        far = g['far_excl_per_yr'][:]
        floor = float(f.attrs['far_floor_per_yr'])
        for i in np.nonzero(far < a.far_thr)[0]:
            rows.append(dict(layer=layer, week=week, gps=float(g['H1_time'][i]), L1_time=float(g['L1_time'][i]),
                             far_per_yr=max(float(far[i]), floor), at_floor=bool(far[i] <= floor), far_floor=floor,
                             n_louder_bg=(0 if ('far_is_upper_limit' in g and bool(g['far_is_upper_limit'][i])) or far[i] <= 0
                                          else int(round(float(far[i]) / floor))),
                             stat=float(g['stat'][i]), net_snr=float(g['network_snr'][i]),
                             H1_snr=float(g['H1_snr'][i]), L1_snr=float(g['L1_snr'][i]),
                             H1_newsnr=float(g['H1_newsnr'][i]), L1_newsnr=float(g['L1_newsnr'][i]),
                             template_id=int(g['template_id'][i]), mass1=float(g['mass1'][i]),
                             mass2=float(g['mass2'][i]), spin1z=float(g['spin1z'][i]), spin2z=float(g['spin2z'][i]),
                             **{k: float(g[k][i]) for k in ('gate_dt', 'env_H1', 'env_L1') if k in g}))   # fg_summary.py

# ---- Poisson 90% upper limit on each FAR: n louder background events in T_b = 1/floor years
from scipy.stats import chi2 as _chi2
for r in rows:
    nb = r['n_louder_bg']
    r['far_ul90_per_yr'] = float(_chi2.ppf(0.9, 2 * (nb + 1)) / 2 * r['far_floor'])

# ---- merge across layers
rows.sort(key=lambda r: r['gps'])
cands = []
for r in rows:
    if cands and r['gps'] - cands[-1]['gps'] <= a.win:
        c = cands[-1]
        c['layers'] = ','.join(sorted(set(c['layers'].split(',')) | {r['layer']}))
        if r['far_per_yr'] < c['far_per_yr']:
            c.update({k: v for k, v in r.items() if k != 'layer'})
    else:
        cands.append(dict(r, layers=r['layer']))
for c in cands:
    c.pop('layer', None)

# ---- cross-match
known = list(csv.DictReader(open(a.known)))
kt = np.array([float(k['gps']) for k in known])
o = np.argsort(kt); kt = kt[o]; known = [known[i] for i in o]
for c in cands:
    lo, hi = np.searchsorted(kt, c['gps'] - a.win), np.searchsorted(kt, c['gps'] + a.win)
    m = [known[j] for j in range(lo, hi)]
    c['known'] = ';'.join(f"{k['name']}|{k['source']}|far={k['far_per_yr'] or 'na'}{('|' + k['note']) if k['note'] else ''}" for k in m)
    c['status'] = 'KNOWN' if m else 'NEW'
    # nearest anything, for context (a candidate 3 s from an alert is still interesting to know about)
    j = np.searchsorted(kt, c['gps'])
    near = [x for x in (j - 1, j) if 0 <= x < len(kt)]
    jn = min(near, key=lambda x: abs(kt[x] - c['gps']))
    c['nearest_known'] = f"{known[jn]['name']}|{known[jn]['source']}|dt={kt[jn] - c['gps']:+.1f}s"

# ---- data-quality context for NEW ones
def template_duration(m1, m2, s1, s2):
    from pycbc.pnutils import get_imr_duration
    return float(get_imr_duration(m1, m2, s1, s2, 20.0, approximant='IMRPhenomD'))

gates = defaultdict(list)
for spec in a.gates:
    w, p = spec.split('=', 1)
    for l in open(p):
        if l.strip() and not l.startswith('#'):
            gates[w].append(float(l.split()[0]))
peaks = {}
for spec in a.peaks:
    wi, p = spec.split('=', 1)
    peaks[tuple(wi.split(':'))] = p


def dq_segments(ifo, flag, t):
    run = 'O4a_4KHZ_R1' if t < 1389456018 else 'O4b_4KHZ_R1' if t < 1422118818 else 'O4c_4KHZ_R1'
    url = f'https://gwosc.org/timeline/segments/json/{run}/{ifo}_{flag}/{int(t) - 1}/2/'
    cmd = ['curl', '-sf', '--retry', '3', url] + (['--cacert', a.cacert] if a.cacert else [])
    try:
        segs = json.loads(subprocess.run(cmd, check=True, capture_output=True).stdout)['segments']
    except Exception:
        return None
    return any(s0 <= t < s1 for s0, s1 in segs)


for c in cands:
    if c['status'] != 'NEW':
        continue
    c['duration_s'] = template_duration(c['mass1'], c['mass2'], c['spin1z'], c['spin2z'])
    c['short'] = c['duration_s'] < 0.2
    g = np.array(gates.get(c['week'], []))
    if len(g):
        c['gate_dt'] = float(np.min(np.abs(g - c['gps'])))
    for ifo in ('H1', 'L1'):
        p = peaks.get((c['week'], ifo))
        if p:
            with h5py.File(p) as f:
                t = f['end_time']
                lo, hi = np.searchsorted(t, c['gps'] - 8), np.searchsorted(t, c['gps'] + 8)
                tt, ss = t[lo:hi], f['snr'][lo:hi]
            c[f'env_{ifo}'] = int(((ss >= 8) & (np.abs(tt - c['gps']) > 0.2)).sum())
        if not a.no_dq:
            for flag in ('CBC_CAT1', 'CBC_CAT2', 'NO_CBC_HW_INJ'):
                ok = dq_segments(ifo, flag, c['gps'])
                c[f'{ifo}_{flag}_ok'] = ok   # True = passes the flag; False = vetoed; None = query failed

cols = ['status', 'gps', 'far_per_yr', 'far_ul90_per_yr', 'n_louder_bg', 'at_floor', 'layers', 'week', 'stat', 'net_snr', 'H1_snr', 'L1_snr',
        'H1_newsnr', 'L1_newsnr', 'mass1', 'mass2', 'spin1z', 'spin2z', 'template_id', 'known', 'nearest_known',
        'duration_s', 'short', 'gate_dt', 'env_H1', 'env_L1', 'H1_CBC_CAT1_ok', 'H1_CBC_CAT2_ok',
        'L1_CBC_CAT1_ok', 'L1_CBC_CAT2_ok', 'H1_NO_CBC_HW_INJ_ok', 'L1_NO_CBC_HW_INJ_ok']   # at_floor True: far_per_yr is an upper limit (the week's floor)
with open(f'{a.out}/candidates.tsv', 'w') as f:
    f.write('\t'.join(cols) + '\n')
    for c in sorted(cands, key=lambda c: c['far_per_yr']):
        f.write('\t'.join('' if c.get(k) is None else (f'{c[k]:.6g}' if isinstance(c[k], float) and k != 'gps' else
                                                         f'{c[k]:.3f}' if k == 'gps' else str(c[k])) for k in cols) + '\n')
json.dump(dict(far_thr=a.far_thr, win_s=a.win, n=len(cands), n_new=sum(c['status'] == 'NEW' for c in cands),
               inputs=a.ranked, known=a.known, candidates=cands), open(f'{a.out}/candidates.json', 'w'), indent=1)
print(f"{len(cands)} candidates with FAR < {a.far_thr}/yr, {sum(c['status'] == 'NEW' for c in cands)} NEW -> {a.out}/candidates.tsv")
