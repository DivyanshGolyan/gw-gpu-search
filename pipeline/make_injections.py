#!/usr/bin/env python
"""Week-1 injection set for tuning and validating the ranking statistic (PyCBC hdf injection format).

Population (detector frame, inside the bank_v2 interior so bank edges don't dominate):
  m1, m2   log-uniform, 9 <= m2 <= m1 <= 150, m1/m2 <= 6
  spins    spin1z, spin2z independent uniform [-0.9, 0.9] (the bank has s1z = s2z, so this also tests it)
  sky      isotropic; inclination isotropic; polarization, coa_phase uniform
  model    IMRPhenomD, f_lower 20 Hz (same family as the bank: this set is for the ranking, not for coverage)
Times: one every --spacing s (+-10 s jitter) inside the coincident segments, >= 10 s from segment
boundaries and >= 30 s from every GWTC-4 event in the window.
Distance: drawn so that the expected network optimal SNR (with the O4a harmonic-mean PSD) is log-uniform
in [--snr-lo, --snr-hi]. Since SNR ~ 1/distance, distance is log-uniform between per-injection bounds
dist_lo, dist_hi, which are stored. For a uniform-in-volume population, weight each injection by
vt_weight = d^3 * ln(dist_hi / dist_lo) / 3 (Mpc^3, the per-injection volume factor); sum of weights over
found injections / N is the sensitive volume (times 4 pi / 3 conventions as you like), at a fixed FAR.
Expected SNRs (opt_snr_H1, opt_snr_L1, opt_snr_net) are stored; they use one PSD for both detectors, so
they are approximate (inferred), only for bookkeeping.
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--segments', required=True); ap.add_argument('--events', required=True)
ap.add_argument('--psd', required=True, help='two-column f, PSD text file')
ap.add_argument('--out', required=True)
ap.add_argument('--spacing', type=float, default=100.0)
ap.add_argument('--snr-lo', type=float, default=4.0); ap.add_argument('--snr-hi', type=float, default=40.0)
ap.add_argument('--seed', type=int, default=20261027)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

segs = [tuple(map(float, l.split()[:2])) for l in open(a.segments) if l.strip()]
ev = np.array([e['gps'] for e in json.load(open(a.events))])
tc = []
for s0, s1 in segs:
    t = np.arange(s0 + 10 + a.spacing / 2, s1 - 10, a.spacing)
    tc.append(t + rng.uniform(-10, 10, len(t)))
tc = np.concatenate(tc)
seg0 = np.array([s for s, _ in segs]); seg1 = np.array([e for _, e in segs])
i = np.searchsorted(seg0, tc, 'right') - 1
tc = tc[(tc - seg0[i] >= 10) & (seg1[i] - tc >= 10)]
tc = tc[np.min(np.abs(tc[:, None] - ev[None, :]), axis=1) >= 30]
n = len(tc)

lm = np.log([9.0, 150.0])
m1, m2 = [], []
while len(m1) < n:
    x, y = np.exp(rng.uniform(*lm, 4 * n)), np.exp(rng.uniform(*lm, 4 * n))
    a_, b_ = np.maximum(x, y), np.minimum(x, y)
    ok = a_ / b_ <= 6
    m1 += list(a_[ok]); m2 += list(b_[ok])
m1, m2 = np.array(m1[:n]), np.array(m2[:n])
s1, s2 = rng.uniform(-0.9, 0.9, n), rng.uniform(-0.9, 0.9, n)
ra, dec = rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n))
inc, pol, phase = np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n)

# expected SNR at 1 Mpc: sigma(1 Mpc) per waveform x antenna/inclination factor per detector
from pycbc.waveform import get_fd_waveform
from pycbc.detector import Detector
from pycbc.filter import sigma
from pycbc.types import FrequencySeries
f, p = np.loadtxt(a.psd, unpack=True)
df = 1.0 / 16; fk = np.arange(0, 1024 + df, df)
psd = FrequencySeries(np.interp(fk, f, p, left=np.inf, right=np.inf), delta_f=df)
sig1 = np.array([sigma(get_fd_waveform(approximant='IMRPhenomD', mass1=x, mass2=y, spin1z=u, spin2z=v,
                                       delta_f=df, f_lower=20.0, distance=1.0)[0].astype(complex), psd=psd,
                       low_frequency_cutoff=20.0, high_frequency_cutoff=1000.0)
                 for x, y, u, v in zip(m1, m2, s1, s2)])
snr1 = {}
for ifo in ('H1', 'L1'):
    fp, fc = Detector(ifo).antenna_pattern(ra, dec, pol, tc)
    snr1[ifo] = sig1 * np.abs(fp * (1 + np.cos(inc) ** 2) / 2 + 1j * fc * np.cos(inc))
net1 = np.hypot(snr1['H1'], snr1['L1'])
target = np.exp(rng.uniform(np.log(a.snr_lo), np.log(a.snr_hi), n))
dist = net1 / target
dlo, dhi = net1 / a.snr_hi, net1 / a.snr_lo
w = dist ** 3 * np.log(dhi / dlo) / 3.0

from pycbc.io import FieldArray
from pycbc.inject import InjectionSet
rec = FieldArray.from_kwargs(tc=tc, mass1=m1, mass2=m2, spin1z=s1, spin2z=s2, ra=ra, dec=dec,
                             inclination=inc, polarization=pol, coa_phase=phase, distance=dist,
                             dist_lo=dlo, dist_hi=dhi, vt_weight=w,
                             opt_snr_H1=snr1['H1'] / dist, opt_snr_L1=snr1['L1'] / dist, opt_snr_net=target)
InjectionSet.write(a.out, rec, static_args={'approximant': 'IMRPhenomD', 'f_lower': 20.0, 'f_ref': 20.0,
                                            'taper': 'start'},
                   injtype='cbc', seed=a.seed, population=__doc__.split('Times:')[0])
print(json.dumps(dict(n=n, tc=[float(tc.min()), float(tc.max())], dist_mpc=[float(dist.min()), float(np.median(dist)), float(dist.max())],
                      frac_net_snr_ge_8=float((target >= 8).mean()))))
