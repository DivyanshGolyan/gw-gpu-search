#!/usr/bin/env python
"""Q-transform images of H1 and L1 around a candidate, from GWOSC open strain, for eyeballing glitches.

For each detector: fetch the 4 kHz GWOSC file holding the time, take 64 s around it, whiten with a 4 s
Welch PSD from that stretch, and plot the normalised Q-transform energy for +-2 s and +-0.5 s (20-512 Hz),
with the candidate's template track time marked. Also records the largest normalised energy within
+-0.1 s and within +-2 s, and the data-quality bitmask over the 64 s.
Usage: REQUESTS_CA_BUNDLE=BUNDLE qscan.py GPS OUT_PREFIX [--title TEXT] [--cacert BUNDLE]  (fresh containers: emSign bundle)
Tested on GW230709_122727 (results in ~1 min, ~260 MB downloaded and deleted).
Writes OUT_PREFIX.png and OUT_PREFIX.json. Downloads go to a fresh temp dir, deleted afterwards.
"""
import argparse, json, os, subprocess, tempfile
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('gps', type=float); ap.add_argument('out')
ap.add_argument('--title', default=''); ap.add_argument('--cacert', default=None)
a = ap.parse_args()

from gwosc.locate import get_urls
from pycbc.types import TimeSeries

res = dict(gps=a.gps)
fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
with tempfile.TemporaryDirectory() as tmp:
    for row, ifo in enumerate(('H1', 'L1')):
        urls = [u for u in get_urls(ifo, a.gps - 32, a.gps + 32, sample_rate=4096) if u.endswith('.hdf5')]
        ts, dq = [], []
        for u in urls:
            p = os.path.join(tmp, os.path.basename(u))
            subprocess.run(['curl', '-sfL', '--retry', '3', '-o', p, u] + (['--cacert', a.cacert] if a.cacert else []), check=True)
            with h5py.File(p) as f:
                s = f['strain/Strain']; t0 = float(s.attrs['Xstart']); dt = float(s.attrs['Xspacing'])
                i0 = max(0, int((a.gps - 32 - t0) / dt)); i1 = min(len(s), int((a.gps + 32 - t0) / dt))
                if i1 > i0:
                    ts.append((t0 + i0 * dt, s[i0:i1]))
                m = f['quality/simple/DQmask']; mt0 = float(m.attrs['Xstart'])
                j0 = max(0, int(a.gps - 32 - mt0)); j1 = min(len(m), int(a.gps + 32 - mt0))
                dq.append(m[j0:j1])
            os.remove(p)
        ts.sort(key=lambda x: x[0])
        x = np.concatenate([v for _, v in ts]); start = ts[0][0]
        res[f'{ifo}_dq_mask_values'] = sorted(set(int(v) for v in np.concatenate(dq))) if dq else None
        if np.isnan(x).any():
            res[f'{ifo}_note'] = 'NaNs in strain (no science data)'; continue
        h = TimeSeries(x.astype(np.float64), delta_t=1 / 4096, epoch=start)
        w = h.whiten(4, 4, low_frequency_cutoff=15.0)
        for col, half in enumerate((2.0, 0.5)):
            seg = w.time_slice(a.gps - half - 1, a.gps + half + 1)
            tq, fq, pq = seg.qtransform(delta_t=half / 400, logfsteps=200, qrange=(4, 64), frange=(20, 512))
            pq = np.asarray(pq)
            m = (tq >= a.gps - half) & (tq <= a.gps + half)
            ax = axes[row, col]
            ax.pcolormesh(tq[m] - a.gps, fq, pq[:, m] if pq.shape[0] == len(fq) else pq[m].T, vmax=25, shading='auto')
            ax.set_yscale('log'); ax.set_title(f'{ifo}  +-{half:g} s'); ax.axvline(0, color='w', lw=0.5, ls=':')
            ax.set_xlabel(f's from {a.gps:.3f}'); ax.set_ylabel('Hz')
            P = pq if pq.shape[0] == len(fq) else pq.T
            near = np.abs(tq - a.gps) <= 0.1
            res[f'{ifo}_max_energy_pm{half:g}s'] = float(P[:, m].max())
            if col == 1:
                res[f'{ifo}_max_energy_pm0.1s'] = float(P[:, near].max())
fig.suptitle(a.title or f'GPS {a.gps:.3f}')
fig.savefig(a.out + '.png', dpi=90)
json.dump(res, open(a.out + '.json', 'w'), indent=1)
print(json.dumps(res))
