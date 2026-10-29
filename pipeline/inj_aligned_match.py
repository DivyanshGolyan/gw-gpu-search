#!/usr/bin/env python
"""For each injection: match between the signal each detector actually sees (full model, e.g. IMRPhenomXPHM with
precession and higher modes, projected with F+, Fx) and the best aligned-spin (2,2)-only IMRPhenomD waveform with
the same masses and spin z-components, maximised over time and phase. Low match = precession/higher modes matter.
Usage: inj_aligned_match.py INJ.hdf PSD.txt OUT.npz [--procs 4]"""
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument('inj'); ap.add_argument('psd'); ap.add_argument('out'); ap.add_argument('--procs', type=int, default=4)
a = ap.parse_args()
DF, FL = 1.0 / 32, 20.0


def work(rows):
    from pycbc.waveform import get_fd_waveform
    from pycbc.detector import Detector
    from pycbc.filter import match
    from pycbc.types import FrequencySeries
    f, p = np.loadtxt(a.psd, unpack=True)
    out = []
    for r in rows:
        try:
            kw = dict(mass1=r['mass1'], mass2=r['mass2'], spin1x=r['spin1x'], spin1y=r['spin1y'], spin1z=r['spin1z'],
                      spin2x=r['spin2x'], spin2y=r['spin2y'], spin2z=r['spin2z'], inclination=r['inclination'],
                      coa_phase=r['coa_phase'], f_ref=r['f_ref'], delta_f=DF, f_lower=FL)
            hp, hc = get_fd_waveform(approximant=r['approximant'], **kw)
            ha, _ = get_fd_waveform(approximant='IMRPhenomD', mass1=r['mass1'], mass2=r['mass2'],
                                    spin1z=r['spin1z'], spin2z=r['spin2z'], delta_f=DF, f_lower=FL)
            n = max(len(hp), len(ha)); hp.resize(n); hc.resize(n); ha.resize(n)
            psd = FrequencySeries(np.interp(np.arange(n) * DF, f, p, left=np.inf, right=np.inf), delta_f=DF)
            m = []
            for ifo in ('H1', 'L1'):
                fp, fc = Detector(ifo).antenna_pattern(r['ra'], r['dec'], r['polarization'], r['tc'])
                m.append(match(fp * hp + fc * hc, ha, psd=psd, low_frequency_cutoff=FL, high_frequency_cutoff=1000.0)[0])
            out.append(m)
        except Exception:
            out.append([np.nan, np.nan])
    return out


if __name__ == '__main__':
    import h5py
    with h5py.File(a.inj) as f:
        cols = ['tc', 'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z', 'inclination',
                'coa_phase', 'f_ref', 'ra', 'dec', 'polarization']
        D = {c: f[c][:] for c in cols}
        apx = f.attrs['static_args'] if 'static_args' in f.attrs else None
        approximant = f.attrs['approximant'] if 'approximant' in f.attrs else 'IMRPhenomXPHM'
    approximant = approximant.decode() if isinstance(approximant, bytes) else str(approximant)
    rows = [dict({c: float(D[c][i]) for c in cols}, approximant=approximant) for i in range(len(D['tc']))]
    chunks = [rows[i::a.procs * 8] for i in range(a.procs * 8)]
    with ProcessPoolExecutor(a.procs) as ex:
        res = list(ex.map(work, chunks))
    M = np.full((len(rows), 2), np.nan)
    for i, r in enumerate(res):
        M[i::a.procs * 8] = r
    np.savez(a.out, match_H1=M[:, 0], match_L1=M[:, 1], approximant=approximant)
    print('approximant', approximant, 'n', len(rows), 'nan', int(np.isnan(M).any(1).sum()),
          'median match', np.nanmedian(M, 0), 'frac min(H,L) < 0.95:', float(np.mean(np.nanmin(M, 1) < 0.95)))
