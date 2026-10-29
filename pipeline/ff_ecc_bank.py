#!/usr/bin/env python
"""Fitting factor of a placed eccentric bank (and of bank_v2, paired) on SEOBNRv5EHM signals inside the bank's box.

Signals: all modes, random orientation, chi1z/chi2z uniform +-chimax, e20 uniform in [--e20], Mtot log-uniform in
[--mtot] (the box), q <= qmax, m2 >= 8. FF maximised over template, time (lags 4x zero-padded) and phase.
Usage: ff_ecc_bank.py PSD ECC_BANK.hdf ALIGNED_BANK.hdf OUT.json --mtot 50 55 [--n 150]
"""
import argparse, json, os, sys, time
import numpy as np, h5py
from multiprocessing import Pool
from scipy import fft as sfft
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_templates as et

ap = argparse.ArgumentParser()
ap.add_argument('psd'); ap.add_argument('ebank'); ap.add_argument('abank'); ap.add_argument('out')
ap.add_argument('--mtot', type=float, nargs=2, required=True); ap.add_argument('--e20', type=float, nargs=2, default=[0.15, 0.45])
ap.add_argument('--qmax', type=float, default=4.0); ap.add_argument('--chimax', type=float, default=0.5)
ap.add_argument('--n', type=int, default=150); ap.add_argument('--procs', type=int, default=4); ap.add_argument('--seed', type=int, default=11)
ap.add_argument('--df', type=float, default=1 / 16)
a = ap.parse_args()
FS = 4096.0; N = int(FS / a.df)
f = np.arange(0, 1024 + a.df / 2, a.df); k0 = int(20 / a.df); fs = f[k0:]; nb = len(fs); NFFT = sfft.next_fast_len(4 * nb)
p = np.loadtxt(a.psd); asd = np.sqrt(np.interp(fs, p[:, 0], p[:, 1]))
mc = lambda m1, m2: (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def wn(h):
    w = h / asd
    return (w / np.sqrt(np.sum(np.abs(w) ** 2, -1, keepdims=True))).astype(np.complex64)


with h5py.File(a.ebank) as b:
    rows = [dict(zip(('mass1', 'mass2', 'spin1z', 'ecc_b', 'anom_b', 'mtot_b', 'fstart_b'), v)) for v in
            zip(*(b[k][:] for k in ('mass1', 'mass2', 'spin1z', 'ecc_b', 'anom_b', 'mtot_b', 'fstart_b')))]
EMC = np.array([mc(r['mass1'], r['mass2']) for r in rows])
with h5py.File(a.abank) as b:
    AB = np.stack([b['mass1'][:], b['mass2'][:], b['spin1z'][:], b['spin2z'][:]], 1)
AMC = mc(AB[:, 0], AB[:, 1])
EW = AW = None


def init():
    global EW, AW
    from pycbc.waveform import get_fd_waveform
    EW = wn(et.fd_on_grid(et._rows_keys(rows), fs))
    sel = np.nonzero((AMC > 0.6 * EMC.min()) & (AMC < 1.9 * EMC.max()))[0]
    AW = {}
    for i in sel:
        hp, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=AB[i, 0], mass2=AB[i, 1], spin1z=AB[i, 2], spin2z=AB[i, 3],
                                delta_f=a.df, f_lower=20.0, f_final=1024.0)
        x = np.zeros(len(f), complex); n = min(len(hp), len(f)); x[:n] = hp.numpy()[:n]; AW[i] = wn(x[k0:])
    AW = (np.array(list(AW.keys())), np.stack(list(AW.values())))


def ff(s, W):
    best = 0.0
    for j in range(0, len(W), 128):
        z = np.zeros((min(128, len(W) - j), NFFT), np.complex64); z[:, :nb] = np.conj(W[j:j + 128]) * s
        best = max(best, float(np.abs(sfft.ifft(z, axis=1)).max()) * NFFT)
    return best


def work(i):
    import warnings, logging
    warnings.filterwarnings('ignore'); logging.getLogger('pyseobnr').setLevel(logging.ERROR)
    if EW is None:
        init()
    from pyseobnr.generate_waveform import GenerateWaveform
    rng = np.random.default_rng([a.seed, i])
    while True:
        M = np.exp(rng.uniform(*np.log(a.mtot))); q = rng.uniform(1, a.qmax)
        if M / (1 + q) >= 8: break
    r = dict(i=i, m1=M * q / (1 + q), m2=M / (1 + q), s1=rng.uniform(-a.chimax, a.chimax), s2=rng.uniform(-a.chimax, a.chimax),
             e=rng.uniform(*a.e20), anom=rng.uniform(0, 2 * np.pi), inc=np.arccos(rng.uniform(-1, 1)), phi=rng.uniform(0, 2 * np.pi), psi=rng.uniform(0, np.pi))
    par = dict(mass1=r['m1'], mass2=r['m2'], spin1z=r['s1'], spin2z=r['s2'], eccentricity=r['e'], rel_anomaly=r['anom'], f22_start=20.0,
               f_ref=20.0, deltaT=1 / FS, approximant='SEOBNRv5EHM', distance=100.0, inclination=r['inc'], phi_ref=r['phi'])
    try:
        hp, hc = GenerateWaveform(par).generate_td_polarizations()
    except Exception as ex:
        r['err'] = repr(ex)[:100]; return r
    h = (np.cos(2 * r['psi']) * hp.data.data + np.sin(2 * r['psi']) * hc.data.data)[-N:].copy()
    m = int(0.1 * FS); h[:m] *= 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
    x = np.zeros(N); x[:len(h)] = h
    s = wn(np.fft.rfft(x)[:len(f)][k0:])
    mcs = mc(r['m1'], r['m2'])
    ek = np.nonzero((EMC > 0.75 * mcs) & (EMC < 1.35 * mcs))[0]
    r['ff_ecc_bank'] = ff(s, EW[ek]) if len(ek) else 0.0
    ak = np.nonzero((AMC[AW[0]] > 0.7 * mcs) & (AMC[AW[0]] < 1.9 * mcs))[0]
    r['ff_aligned'] = ff(s, AW[1][ak])
    return r


if __name__ == '__main__':
    t = time.time()
    with Pool(a.procs) as pool:
        R = pool.map(work, range(a.n))
    G = [r for r in R if 'err' not in r]
    fe = np.array([r['ff_ecc_bank'] for r in G]); fa = np.array([r['ff_aligned'] for r in G])
    summ = dict(n=len(G), ff_ecc_bank=np.percentile(fe, [5, 25, 50, 75]).tolist(), ff_aligned=np.percentile(fa, [5, 25, 50, 75]).tolist(),
                vt_ratio_snr_cubed=float(np.mean(fe ** 3) / np.mean(fa ** 3)), sec=time.time() - t)
    json.dump(dict(args=vars(a), summary=summ, rows=R), open(a.out, 'w'), indent=1)
    print(json.dumps(summ))
