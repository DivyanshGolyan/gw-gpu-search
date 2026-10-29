#!/usr/bin/env python
"""Head-to-head on LVK injections: found at IFAR >= --ifar yr by us (coinc_rank injection run) vs by PyCBC
(pycbc_far from the LVK release), split by whether precession/higher modes matter (aligned-22 match).

Usage: compare_pycbc.py --inj lvk_week1_xphm.hdf --match lvk_week1_match.npz --ours inj_run.hdf --out DIR
The injection run must be ranked with the search run's background (coinc_rank --background-from/--fits-from).
"""
import argparse, json, os
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument('--inj', required=True); ap.add_argument('--match', required=True); ap.add_argument('--ours', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--ifar', type=float, default=1.0)
ap.add_argument('--dt', type=float, default=0.1, help='s, H1 arrival time vs candidate H1 time')
ap.add_argument('--match-thr', type=float, default=0.95)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

with h5py.File(a.inj) as f:
    I = {k: f[k][:] for k in f.keys()}
M = np.load(a.match)
mmin = np.fmin(M['match_H1'], M['match_L1'])
from pycbc.detector import Detector
tH = I['tc'] + Detector('H1').time_delay_from_earth_center(I['ra'], I['dec'], I['tc'])

with h5py.File(a.ours) as f:
    ct = f['foreground/H1_time'][:]; cfar = f['foreground/far_incl_per_yr'][:]; cst = f['foreground/stat'][:]
o = np.argsort(ct); ct, cfar, cst = ct[o], cfar[o], cst[o]
lo = np.searchsorted(ct, tH - a.dt); hi = np.searchsorted(ct, tH + a.dt)
our_far = np.full(len(tH), np.inf); our_stat = np.full(len(tH), np.nan)
for i in np.nonzero(hi > lo)[0]:
    j = lo[i] + np.argmin(cfar[lo[i]:hi[i]])
    our_far[i], our_stat[i] = cfar[j], cst[j]

thr = 1.0 / a.ifar
us, pc = our_far < thr, I['pycbc_far'] < thr
groups = {'all': np.ones(len(tH), bool), f'match>={a.match_thr} (aligned-22 enough)': mmin >= a.match_thr,
          f'match<{a.match_thr} (precession/HM matter)': mmin < a.match_thr, 'match unknown': np.isnan(mmin)}
summary = {}
for g, m in groups.items():
    w = I['weights'][m]
    summary[g] = dict(n=int(m.sum()), found_us=int((us & m).sum()), found_pycbc=int((pc & m).sum()),
                      both=int((us & pc & m).sum()), us_only=int((us & ~pc & m).sum()), pycbc_only=int((~us & pc & m).sum()),
                      weighted_ratio_us_over_pycbc=float(w[us[m]].sum() / max(w[pc[m]].sum(), 1e-300)))
print(json.dumps(summary, indent=1))
json.dump(dict(ifar_yr=a.ifar, dt_s=a.dt, match_thr=a.match_thr, groups=summary), open(f'{a.out}/head_to_head.json', 'w'), indent=1)

cols = ['lvk_row', 'tc', 'mass1', 'mass2', 'spin1z', 'spin2z', 'snr_net', 'pycbc_far']
hdr = cols + ['our_far', 'our_stat', 'match_H1', 'match_L1']
for name, m in (('found_by_us_not_pycbc', us & ~pc), ('found_by_pycbc_not_us', ~us & pc)):
    idx = np.nonzero(m)[0]
    idx = idx[np.argsort(-I['snr_net'][idx])]
    with open(f'{a.out}/{name}.tsv', 'w') as f:
        f.write('\t'.join(hdr) + '\n')
        for i in idx:
            f.write('\t'.join(f'{v:.6g}' for v in [I[c][i] for c in cols] + [our_far[i], our_stat[i], M['match_H1'][i], M['match_L1'][i]]) + '\n')
    print(name, len(idx), '->', f'{a.out}/{name}.tsv')
