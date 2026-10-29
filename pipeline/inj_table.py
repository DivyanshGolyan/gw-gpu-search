#!/usr/bin/env python
"""One auditable per-injection table for a head-to-head (review item R1): every injection in the subset,
our outcome and PyCBC's, each with its own IFAR, plus the parameters needed to judge a disagreement.

Columns: lvk_row, tc, mass1_det, mass2_det, spin1z, spin2z, chi_p, snr_opt_H/L (LVK optimal SNR),
  match_H1/L1 (aligned-22 match), our_far, our_stat, our_snr_H1/L1, our_template, our_m1/m2,
  pycbc_far, pycbc_snr_H/L, pycbc_mode (coinc / single-H / single-L / none, from PyCBC's recovered SNRs),
  class (both / ours_only / pycbc_only / neither at IFAR >= --ifar yr).
Also writes a JSON with the counts and their arithmetic, split by pycbc_mode.
Usage: inj_table.py --inj SUBSET.hdf --lvk o4a_sens.hdf --match MATCH.npz --ours INJ_RUN.hdf --out PREFIX
"""
import argparse, json
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument('--inj', required=True); ap.add_argument('--lvk', required=True)
ap.add_argument('--match', required=True); ap.add_argument('--ours', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--ifar', type=float, default=1.0)
ap.add_argument('--dt', type=float, default=0.1)
a = ap.parse_args()

with h5py.File(a.inj) as f:
    I = {k: f[k][:] for k in f.keys()}
row = I['lvk_row']
with h5py.File(a.lvk) as f:
    E = f['events']
    L = {k: E[k][row] for k in ('mass1_detector', 'mass2_detector', 'spin1z', 'spin2z', 'chi_p', 'snr_H', 'snr_L',
                                'observed_snr_H', 'observed_snr_L', 'pycbc_far', 'pycbc_rec_snr_H', 'pycbc_rec_snr_L',
                                'pycbc_gps_time')}
M = np.load(a.match)
from pycbc.detector import Detector
tH = I['tc'] + Detector('H1').time_delay_from_earth_center(I['ra'], I['dec'], I['tc'])

with h5py.File(a.ours) as f:
    g = f['foreground']
    F = {k: g[k][:] for k in ('H1_time', 'far_incl_per_yr', 'stat', 'H1_snr', 'L1_snr', 'template_id', 'mass1', 'mass2')
         if k in g}
o = np.argsort(F['H1_time']); F = {k: v[o] for k, v in F.items()}
n = len(tH)
our = {k: np.full(n, np.nan) for k in ('far', 'stat', 'snr_H1', 'snr_L1', 'template', 'm1', 'm2')}
our['far'][:] = np.inf
lo, hi = np.searchsorted(F['H1_time'], tH - a.dt), np.searchsorted(F['H1_time'], tH + a.dt)
for i in np.nonzero(hi > lo)[0]:
    j = lo[i] + np.argmin(F['far_incl_per_yr'][lo[i]:hi[i]])
    our['far'][i] = F['far_incl_per_yr'][j]; our['stat'][i] = F['stat'][j]
    our['snr_H1'][i] = F['H1_snr'][j]; our['snr_L1'][i] = F['L1_snr'][j]
    our['template'][i] = F['template_id'][j]
    if 'mass1' in F:
        our['m1'][i], our['m2'][i] = F['mass1'][j], F['mass2'][j]

ph, pl, pf = L['pycbc_rec_snr_H'], L['pycbc_rec_snr_L'], L['pycbc_far']
found_pc = pf < 1 / a.ifar
hok, lok = np.isfinite(ph) & (ph > 0), np.isfinite(pl) & (pl > 0)
mode = np.where(hok & lok, 'coinc', np.where(hok, 'single-H', np.where(lok, 'single-L', 'none')))
mode = np.where(np.isfinite(pf), mode, 'none')
found_us = our['far'] < 1 / a.ifar
cls = np.where(found_us & found_pc, 'both', np.where(found_us, 'ours_only', np.where(found_pc, 'pycbc_only', 'neither')))

cols = [('lvk_row', row, '%d'), ('tc', I['tc'], '%.3f'), ('mass1_det', L['mass1_detector'], '%.2f'),
        ('mass2_det', L['mass2_detector'], '%.2f'), ('spin1z', L['spin1z'], '%.3f'), ('spin2z', L['spin2z'], '%.3f'),
        ('chi_p', L['chi_p'], '%.3f'), ('snr_opt_H', L['snr_H'], '%.2f'), ('snr_opt_L', L['snr_L'], '%.2f'),
        ('snr_obs_H', L['observed_snr_H'], '%.2f'), ('snr_obs_L', L['observed_snr_L'], '%.2f'),
        ('match_H1', M['match_H1'], '%.3f'), ('match_L1', M['match_L1'], '%.3f'),
        ('our_far', our['far'], '%.4g'), ('our_stat', our['stat'], '%.3f'), ('our_snr_H1', our['snr_H1'], '%.2f'),
        ('our_snr_L1', our['snr_L1'], '%.2f'), ('our_template', our['template'], '%.0f'), ('our_m1', our['m1'], '%.2f'),
        ('our_m2', our['m2'], '%.2f'), ('pycbc_far', pf, '%.4g'), ('pycbc_snr_H', ph, '%.2f'), ('pycbc_snr_L', pl, '%.2f')]
with open(a.out + '.tsv', 'w') as f:
    f.write('\t'.join([c for c, _, _ in cols] + ['pycbc_mode', 'class']) + '\n')
    for i in range(n):
        f.write('\t'.join([fmt % v[i] if np.isfinite(v[i]) else ('inf' if v[i] == np.inf else 'nan') for _, v, fmt in cols]
                          + [mode[i], cls[i]]) + '\n')

cnt = {c: int((cls == c).sum()) for c in ('both', 'ours_only', 'pycbc_only', 'neither')}
by_mode = {m: {c: int(((cls == c) & (mode == m)).sum()) for c in cnt} for m in ('coinc', 'single-H', 'single-L', 'none')}
summary = dict(ifar_yr=a.ifar, n=n, found_ours=int(found_us.sum()), found_pycbc=int(found_pc.sum()), classes=cnt,
               arithmetic=f"ours {found_us.sum()} = both {cnt['both']} + ours_only {cnt['ours_only']}; "
                          f"pycbc {found_pc.sum()} = both {cnt['both']} + pycbc_only {cnt['pycbc_only']}; "
                          f"difference {found_pc.sum() - found_us.sum()} = pycbc_only - ours_only",
               by_pycbc_mode=by_mode,
               pycbc_found_coinc_only=int((found_pc & (mode == 'coinc')).sum()),
               inputs=dict(inj=a.inj, ours=a.ours, match=a.match))
json.dump(summary, open(a.out + '.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
