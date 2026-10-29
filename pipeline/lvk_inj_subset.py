#!/usr/bin/env python
"""Pick a one-pass subset of the LVK GWTC-4.0 O4a sensitivity injections (zenodo 16740117) inside our
analysed segments and bank range, >= --sep s apart, and write it in PyCBC injection format, keeping the
LVK row index and PyCBC's own FAR so found/missed can be compared one-to-one.
Usage: lvk_inj_subset.py LVK.hdf segments.txt OUT.hdf [--sep 100] [--seed 1] [--exclude EARLIER_SUBSET.hdf ...]"""
import argparse, json
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument('lvk'); ap.add_argument('segments'); ap.add_argument('out')
ap.add_argument('--sep', type=float, default=100.0); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--edge', type=float, default=10.0)
ap.add_argument('--exclude', nargs='*', default=[], help='earlier subset files whose lvk_row must not be reused')
a = ap.parse_args()
segs = [tuple(map(float, l.split()[:2])) for l in open(a.segments) if l.strip()]
s0 = np.array([x for x, _ in segs]); s1 = np.array([y for _, y in segs])
with h5py.File(a.lvk) as f:
    E = f['events']
    t = E['time_geocenter'][:]
    i = np.searchsorted(s0, t, 'right') - 1
    ok = (i >= 0) & (t - s0[np.clip(i, 0, None)] >= a.edge) & (s1[np.clip(i, 0, None)] - t >= a.edge)
    m1d, m2d = E['mass1_detector'][:], E['mass2_detector'][:]
    ok &= (m2d >= 8) & (m1d <= 200) & (m2d <= 100) & (m1d / m2d <= 8)   # bank_v2 range
    for ex in a.exclude:
        with h5py.File(ex) as g:
            ok[g['lvk_row'][:]] = False
    cand = np.nonzero(ok)[0]
    rng = np.random.default_rng(a.seed); rng.shuffle(cand)
    # greedy: keep a candidate if no kept injection is within sep (random order keeps it unbiased in params)
    kept, times = [], np.array([])
    for j in cand:
        k = np.searchsorted(times, t[j])
        if (k == 0 or t[j] - times[k - 1] >= a.sep) and (k == len(times) or times[k] - t[j] >= a.sep):
            times = np.insert(times, k, t[j]); kept.append(j)
    kept = np.sort(np.array(kept))
    col = lambda n: E[n][:][kept]
    rec = dict(tc=col('time_geocenter'), mass1=col('mass1_detector'), mass2=col('mass2_detector'),
               spin1x=col('spin1x'), spin1y=col('spin1y'), spin1z=col('spin1z'),
               spin2x=col('spin2x'), spin2y=col('spin2y'), spin2z=col('spin2z'),
               distance=col('luminosity_distance'), ra=col('right_ascension'), dec=col('declination'),
               inclination=col('inclination'), polarization=col('polarization'), coa_phase=col('coa_phase'),
               f_ref=col('fref'), f_lower=np.full(len(kept), 15.0),
               lvk_row=kept, pycbc_far=col('pycbc_far'), gstlal_far=col('gstlal_far'), mbta_far=col('mbta_far'),
               snr_net=col('snr_net'), weights=col('weights'))
    apx = set(x.decode() if isinstance(x, bytes) else x for x in col('approximant'))
    n_cand = len(cand)
from pycbc.io import FieldArray
from pycbc.inject import InjectionSet
assert len(apx) == 1, apx
InjectionSet.write(a.out, FieldArray.from_kwargs(**rec), static_args={'approximant': apx.pop(), 'taper': 'start'},
                   injtype='cbc', source='zenodo 16740117 GWTC-4.0 O4a sensitivity estimates', n_candidates=n_cand,
                   subsample_fraction=len(kept) / n_cand)
print(json.dumps(dict(n_candidates=n_cand, n_kept=len(kept), pycbc_found_ifar1=int((rec['pycbc_far'] < 1).sum()),
                      any_found_ifar1=int((np.minimum.reduce([rec['pycbc_far'], rec['gstlal_far'], rec['mbta_far']]) < 1).sum()))))
