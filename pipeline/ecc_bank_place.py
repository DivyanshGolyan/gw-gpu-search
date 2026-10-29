#!/usr/bin/env python
"""Greedy stochastic placement of an eccentric BBH bank (SEOBNRv5EHM 22-mode, equal aligned spins).

Candidates come in families: one ODE solve per base (q, chi, e_b) at total mass Mb from fstart_b, then masses
Mb * (1 + dlnm)^k up to Mb * fam_span by rescaling (ecc_templates.py). Fine mass steps also sweep the orbital
phase at band entry, so anomaly needs no separate grid. Candidates are shuffled, whitened, normalised, and
accepted if their time- and phase-maximised match to every accepted template nearby in (tau0, tau3) is < MM.
Rows outside the target box (Mtot, q, e20 estimated from e_b by leading-order evolution) are dropped.

Usage: ecc_bank_place.py PSD OUT.hdf --mtot 40 50 --e20 0.15 0.45 --mm 0.95 [--gpu]
"""
import argparse, hashlib, os, sys, time
import numpy as np, h5py
from scipy import fft as sfft
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecc_templates as et

ap = argparse.ArgumentParser()
ap.add_argument('psd'); ap.add_argument('out')
ap.add_argument('--mtot', type=float, nargs=2, default=[30.0, 100.0]); ap.add_argument('--qmax', type=float, default=4.0)
ap.add_argument('--mmin', type=float, default=8.0); ap.add_argument('--chimax', type=float, default=0.5)
ap.add_argument('--e20', type=float, nargs=2, default=[0.15, 0.45])
ap.add_argument('--mm', type=float, default=0.95)
ap.add_argument('--fstart-b', type=float, default=18.0); ap.add_argument('--fam-span', type=float, default=1.25)
ap.add_argument('--dlnm', type=float, default=0.0025)
ap.add_argument('--flow', type=float, default=20.0); ap.add_argument('--fmax', type=float, default=1024.0)
ap.add_argument('--df', type=float, default=1 / 8)
ap.add_argument('--bases-per-chunk', type=int, default=56)
ap.add_argument('--stop-frac', type=float, default=0.003, help='stop when a chunk accepts less than this fraction of its candidates')
ap.add_argument('--max-chunks', type=int, default=100000)
ap.add_argument('--procs', type=int, default=4); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--gpu', action='store_true')
ap.add_argument('--max-templates', type=int, default=200000, help='rows preallocated for accepted templates (GPU memory: rows x 64 KB)')
a = ap.parse_args()

f = np.arange(0, a.fmax + a.df / 2, a.df); k0 = int(round(a.flow / a.df)); fs = f[k0:]; nb = len(fs)
p = np.loadtxt(a.psd); asd = np.sqrt(np.interp(fs, p[:, 0], p[:, 1]))
nfft = 4 * nb                      # lag grid 0.25 ms: a 2x grid underestimates matches by up to ~1%
xp = np
if a.gpu:
    import cupy as xp


def tau03(m1, m2):
    from pycbc.conversions import tau0_from_mass1_mass2, tau3_from_mass1_mass2
    return tau0_from_mass1_mass2(m1, m2, a.flow), tau3_from_mass1_mass2(m1, m2, a.flow)


def family(seed):
    """One base solve -> whitened unit-norm candidates of every mass in the family that lands in the box."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(1, a.qmax); chi = rng.uniform(-a.chimax, a.chimax)
    # base mass: log-uniform so families tile the box; e_b drawn so the family's e20 range overlaps the target
    Mb = np.exp(rng.uniform(np.log(a.mtot[0] / a.fam_span), np.log(a.mtot[1])))
    eb_hi = a.e20[1] / et.e_at(20.0, 1.0, a.fstart_b)
    eb_lo = a.e20[0] / et.e_at(20.0 * a.fam_span, 1.0, a.fstart_b)
    eb = rng.uniform(eb_lo, min(eb_hi, 0.6))
    key = et.base_key(q, chi, eb, 0.0, Mb, a.fstart_b)
    M = Mb * (1 + a.dlnm) ** np.arange(int(np.log(a.fam_span) / np.log1p(a.dlnm)) + 1)
    e20 = et.e_at(20.0 * M / Mb, eb, a.fstart_b)
    m2 = M / (1 + q)
    ok = (M >= a.mtot[0]) & (M <= a.mtot[1]) & (m2 >= a.mmin) & (e20 >= a.e20[0]) & (e20 <= a.e20[1])
    M, e20 = M[ok], e20[ok]
    if not len(M):
        return None
    try:
        h = et.fd_on_grid([(key, m) for m in M], fs)
    except Exception:
        return None
    w = h / asd
    w /= np.sqrt(np.sum(np.abs(w) ** 2, 1))[:, None]
    return dict(q=q, chi=chi, eb=eb, Mb=Mb, M=M, e20=e20, W=w.astype(np.complex64))


Wacc = None   # allocated in main after the worker pool forks, so workers never inherit a CUDA context


def maxmatch(w, idx):
    best = 0.0
    for s in range(0, len(idx), 512):
        j = idx[s:s + 512]
        prod = xp.zeros((len(j), nfft), np.complex64)
        prod[:, :nb] = xp.conj(Wacc[xp.asarray(j)]) * xp.asarray(w)
        z = xp.fft.ifft(prod, axis=1) if a.gpu else sfft.ifft(prod, axis=1, workers=a.procs)
        best = max(best, float(xp.abs(z).max()) * nfft)
        if best >= a.mm:
            break
    return best


def save(rows, ncand, nsolve):
    R = np.array(rows)
    n = len(R)
    tmp = a.out + '.tmp'
    with h5py.File(tmp, 'w') as o:
        for j, k in enumerate(('mass1', 'mass2', 'spin1z', 'ecc_b', 'anom_b', 'mtot_b', 'fstart_b', 'ecc20_est')):
            o[k] = R[:, j]
        o['spin2z'] = R[:, 2]; o['f_lower'] = np.full(n, a.flow, np.float32)
        o.create_dataset('approximant', data=np.array(['SEOBNRv5EHM'] * n, dtype=object), dtype=h5py.string_dtype('utf-8'))
        o['template_hash'] = np.array([np.frombuffer(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), np.int64)[0] for r in R])
        o.attrs['parameters'] = np.array(['mass1', 'mass2', 'spin1z', 'spin2z', 'f_lower', 'approximant', 'template_hash'], dtype=object)
        o.attrs['args'] = repr(vars(a)); o.attrs['n_candidates'] = ncand; o.attrs['n_solves'] = nsolve
    os.replace(tmp, a.out)


if __name__ == '__main__':
    rows = []; t0s = []; t3s = []
    T0 = np.empty(a.max_templates); T3 = np.empty(a.max_templates)
    pool = Pool(a.procs)
    Wacc = xp.zeros((a.max_templates, nb), np.complex64)
    t = time.time(); ncand = 0; nsolve = 0
    for c in range(a.max_chunks):
        seeds = [(a.seed, c, i) for i in range(a.bases_per_chunk)]
        fams = [x for x in pool.map(family, seeds) if x is not None]
        nsolve += len(fams)
        cand = [(fi, k) for fi, F in enumerate(fams) for k in range(len(F['M']))]
        order = np.random.default_rng((a.seed, c, 99)).permutation(len(cand))
        na0 = len(rows); ncand += len(cand)
        for o in order:
            fi, k = cand[o]; F = fams[fi]
            m1, m2 = F['M'][k] * F['q'] / (1 + F['q']), F['M'][k] / (1 + F['q'])
            t0, t3 = tau03(m1, m2)
            na = len(rows)
            if na:
                near = np.flatnonzero((np.abs(T0[:na] - t0) < 0.1 + 0.08 * t0) & (np.abs(T3[:na] - t3) < 0.03 + 0.08 * t3))
                if len(near) and maxmatch(F['W'][k], near) >= a.mm:
                    continue
            Wacc[na] = xp.asarray(F['W'][k]); T0[na] = t0; T3[na] = t3
            rows.append((m1, m2, F['chi'], F['eb'], 0.0, F['Mb'], a.fstart_b, F['e20'][k]))
        acc = len(rows) - na0
        print(f'chunk {c}: {len(cand)} candidates from {len(fams)} solves, accepted {acc}, bank {len(rows)}, '
              f'{ncand} candidates, {time.time() - t:.0f}s', flush=True)
        if c % 5 == 4:
            save(rows, ncand, nsolve)
        if c >= 3 and acc < a.stop_frac * max(len(cand), 1):
            break
    save(rows, ncand, nsolve); n = len(rows)
    print(f'done: {n} templates, {ncand} candidates, {nsolve} solves, {time.time() - t:.0f}s')
