#!/usr/bin/env python
"""Frozen eccentric BBH injection set (PyCBC hdf InjectionSet) for VT at fixed FAR on a data slice.

Population (detector frame), the region the eccentric add-on targets:
  Mtot     log-uniform in [--mtot], q = m1/m2 uniform in [1, --qmax], m2 >= --mmin
  spins    spin1z, spin2z independent uniform [-0.5, 0.5]
  ecc      eccentricity uniform in [0, --emax] at the orbit-averaged 22 frequency f_lower = 20 Hz, rel_anomaly uniform
  model    SEOBNRv5EHM, all default modes (pyseobnr's PyCBC plugin; rel_anomaly reaches pyseobnr through the
           injection row), f_lower = f_ref = 20 Hz, taper start
  sky      isotropic; inclination isotropic; polarization, coa_phase uniform
Times: one every --spacing s (+-5 s) inside the coincident segments that start before --t-end, >= 10 s from segment edges.
Distance: expected network optimal SNR log-uniform in [--snr-lo, --snr-hi] (one O4a harmonic-mean PSD for both
detectors, computed from the actual EHM polarizations and antenna patterns), so distance is log-uniform between the
stored per-injection dist_lo, dist_hi and vt_weight = d^3 ln(dist_hi / dist_lo) / 3 gives VT for a uniform-in-volume
population (same convention as injections/week1/SPEC.md).
Note: injections start at 20 Hz, so eccentric harmonics emitted before the 22 mode reaches 20 Hz are absent (as in
Wang & Nitz 2508.05018); this only makes real signals slightly louder than the injections.

Usage: make_ecc_injections.py --segments coinc_segments.txt --psd psd.txt --out inj.hdf [--t-end GPS]
"""
import argparse, json, logging, warnings
import numpy as np
from multiprocessing import Pool

ap = argparse.ArgumentParser()
ap.add_argument('--segments', required=True); ap.add_argument('--psd', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--t-start', type=float, default=0); ap.add_argument('--t-end', type=float, default=1e12)
ap.add_argument('--spacing', type=float, default=50.0)
ap.add_argument('--mtot', type=float, nargs=2, default=[30.0, 100.0]); ap.add_argument('--qmax', type=float, default=4.0)
ap.add_argument('--mmin', type=float, default=8.0); ap.add_argument('--chimax', type=float, default=0.5)
ap.add_argument('--emax', type=float, default=0.45)
ap.add_argument('--snr-lo', type=float, default=5.0); ap.add_argument('--snr-hi', type=float, default=30.0)
ap.add_argument('--procs', type=int, default=4); ap.add_argument('--seed', type=int, default=20261028)
a = ap.parse_args()
FS = 4096.0; DF = 1 / 16


def sigma_hp_hc(args):
    """sigma (1 Mpc) of h+ and hx and their cross term, for the projected SNR of any antenna pattern."""
    warnings.filterwarnings('ignore'); logging.getLogger('pyseobnr').setLevel(logging.ERROR)
    m1, m2, s1, s2, e, an, inc, ph = args
    from pyseobnr.generate_waveform import GenerateWaveform
    p = dict(mass1=m1, mass2=m2, spin1z=s1, spin2z=s2, eccentricity=e, rel_anomaly=an, f22_start=20.0, f_ref=20.0,
             deltaT=1 / FS, approximant='SEOBNRv5EHM', distance=1.0, inclination=inc, phi_ref=ph)
    try:
        hp, hc = GenerateWaveform(p).generate_td_polarizations()
    except Exception:
        return None
    n = int(FS / DF)
    f = np.fft.rfftfreq(n, 1 / FS)
    pf, pv = np.loadtxt(a.psd).T
    w = np.where((f >= 20) & (f <= 1000), 1 / np.interp(f, pf, pv, left=np.inf, right=np.inf), 0.0)
    r = int(0.1 * FS)
    tap = np.ones(len(hp.data.data)); tap[:r] = 0.5 * (1 - np.cos(np.pi * np.arange(r) / r))
    P = np.fft.rfft(hp.data.data * tap, n) / FS; X = np.fft.rfft(hc.data.data * tap, n) / FS
    ip = lambda u, v: 4 * DF * np.sum((np.conj(u) * v).real * w)
    return ip(P, P), ip(X, X), ip(P, X)


if __name__ == '__main__':
    rng = np.random.default_rng(a.seed)
    segs = [tuple(map(float, l.split()[:2])) for l in open(a.segments) if l.strip()]
    segs = [(s0, s1) for s0, s1 in segs if s0 >= a.t_start and s0 < a.t_end]
    tc = []
    for s0, s1 in segs:
        t = np.arange(s0 + 10 + a.spacing / 2, s1 - 10, a.spacing)
        tc.append(t + rng.uniform(-5, 5, len(t)))
    tc = np.concatenate(tc); n0 = len(tc)
    M = np.exp(rng.uniform(*np.log(a.mtot), 4 * n0)); q = rng.uniform(1, a.qmax, 4 * n0)
    ok = M / (1 + q) >= a.mmin
    M, q = M[ok][:n0], q[ok][:n0]
    m1, m2 = M * q / (1 + q), M / (1 + q)
    s1, s2 = rng.uniform(-a.chimax, a.chimax, n0), rng.uniform(-a.chimax, a.chimax, n0)
    ecc, anom = rng.uniform(0, a.emax, n0), rng.uniform(0, 2 * np.pi, n0)
    ra, dec = rng.uniform(0, 2 * np.pi, n0), np.arcsin(rng.uniform(-1, 1, n0))
    inc, pol, phase = np.arccos(rng.uniform(-1, 1, n0)), rng.uniform(0, np.pi, n0), rng.uniform(0, 2 * np.pi, n0)
    with Pool(a.procs) as pool:
        S = pool.map(sigma_hp_hc, list(zip(m1, m2, s1, s2, ecc, anom, inc, phase)), chunksize=4)
    good = np.array([s is not None for s in S])
    print(f'{n0} times, {int((~good).sum())} waveform failures dropped', flush=True)
    S = np.array([s for s in S if s is not None])
    keep = lambda x: x[good]
    tc, m1, m2, s1, s2, ecc, anom, ra, dec, inc, pol, phase = map(keep, (tc, m1, m2, s1, s2, ecc, anom, ra, dec, inc, pol, phase))
    from pycbc.detector import Detector
    snr1 = {}
    for ifo in ('H1', 'L1'):
        fp, fc = Detector(ifo).antenna_pattern(ra, dec, pol, tc)
        snr1[ifo] = np.sqrt(np.maximum(fp ** 2 * S[:, 0] + fc ** 2 * S[:, 1] + 2 * fp * fc * S[:, 2], 0))
    net1 = np.hypot(snr1['H1'], snr1['L1'])
    n = len(tc)
    target = np.exp(rng.uniform(np.log(a.snr_lo), np.log(a.snr_hi), n))
    dist = net1 / target
    dlo, dhi = net1 / a.snr_hi, net1 / a.snr_lo
    w = dist ** 3 * np.log(dhi / dlo) / 3.0
    from pycbc.io import FieldArray
    from pycbc.inject import InjectionSet
    rec = FieldArray.from_kwargs(tc=tc, mass1=m1, mass2=m2, spin1z=s1, spin2z=s2, eccentricity=ecc, rel_anomaly=anom,
                                 ra=ra, dec=dec, inclination=inc, polarization=pol, coa_phase=phase, distance=dist,
                                 dist_lo=dlo, dist_hi=dhi, vt_weight=w,
                                 opt_snr_H1=snr1['H1'] / dist, opt_snr_L1=snr1['L1'] / dist, opt_snr_net=target)
    InjectionSet.write(a.out, rec, static_args={'approximant': 'SEOBNRv5EHM', 'f_lower': 20.0, 'f_ref': 20.0, 'taper': 'start'},
                       injtype='cbc', seed=a.seed, population=__doc__.split('Usage:')[0])
    print(json.dumps(dict(n=n, tc=[float(tc.min()), float(tc.max())], dist_mpc=[float(dist.min()), float(np.median(dist)), float(dist.max())])))
