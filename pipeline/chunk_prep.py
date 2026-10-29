#!/usr/bin/env python
"""Download one GWOSC chunk for one detector and write a PyCBC hdf strain store.

Usage: chunk_prep.py IFO GPS_START GPS_END OUT_STORE [--dataset O4a] [--rate 4096] [--cache DIR]
Unlike prep_data.py it does not check data-quality flags or write PSDs: the caller picks
chunks inside science time. It stops if the span has NaNs (data gaps).
"""
import argparse, os, subprocess, sys
import h5py, numpy as np
from gwosc.locate import get_urls

ap = argparse.ArgumentParser()
ap.add_argument('ifo'); ap.add_argument('start', type=int); ap.add_argument('end', type=int); ap.add_argument('out')
ap.add_argument('--dataset', default='O4a'); ap.add_argument('--rate', type=int, default=4096)
ap.add_argument('--cache', default=os.path.expanduser('~/gwdata'))
a = ap.parse_args()
os.makedirs(a.cache, exist_ok=True)
ca = os.environ.get('REQUESTS_CA_BUNDLE')
files = []
for u in get_urls(a.ifo, a.start, a.end, sample_rate=a.rate, format='hdf5', dataset=a.dataset):
    fn = os.path.join(a.cache, os.path.basename(u))
    if not os.path.exists(fn):
        part = f'{fn}.{os.getpid()}.part'     # adjacent chunks may fetch the same file at the same time
        subprocess.check_call(['curl', '-sS', '-L', '--retry', '4', '-o', part, u] + (['--cacert', ca] if ca else []))
        os.replace(part, fn)
    files.append(fn)
chunks, t0, dt = [], None, None
for fn in sorted(files):
    with h5py.File(fn, 'r') as f:
        s = f['strain/Strain']
        x0 = int(s.attrs['Xstart']); dt = float(s.attrs['Xspacing'])
        if t0 is None: t0 = x0
        if x0 != t0 + round(sum(len(c) for c in chunks) * dt):
            sys.exit(f'{a.ifo}: files are not contiguous at {x0}')
        chunks.append(s[:])
x = np.concatenate(chunks)[int((a.start - t0) / dt):int((a.end - t0) / dt)]
if len(x) != int((a.end - a.start) / dt) or not np.all(np.isfinite(x)):
    sys.exit(f'{a.ifo} {a.start}-{a.end}: missing samples or NaNs (gap in the data)')
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with h5py.File(a.out + '.part', 'w') as f:
    g = f.create_group(f'{a.ifo}:GWOSC_STRAIN')
    g.create_dataset('0', data=x)
    sg = g.create_group('segments'); sg['start'] = np.array([a.start]); sg['end'] = np.array([a.end])
os.rename(a.out + '.part', a.out)
print(a.out)
