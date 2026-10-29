#!/usr/bin/env python
"""Concatenate per-chunk peak files of one detector into one file sorted by end_time, dropping duplicate
peaks from overlapping chunks (same template, |dt| < 1 ms). Usage: merge_peaks.py OUT.hdf IN1.hdf IN2.hdf ...
coinc_rank.py and fg_summary.py call merge() directly on the piece files, so no merged file is needed."""
import sys
import numpy as np
import h5py


def merge(ins, want=None):
    """Merged columns (all 1-d per-peak datasets the files share, or only those in `want`) as a dict."""
    keys = None; parts = {}
    for p in sorted(ins):
        with h5py.File(p) as f:
            k = [x for x in f.keys() if isinstance(f[x], h5py.Dataset) and f[x].ndim == 1 and len(f[x]) == len(f['end_time'])
                 and (want is None or x in want or x in ('end_time', 'template_id'))]
            keys = k if keys is None else [x for x in keys if x in k]
            for x in k:
                parts.setdefault(x, []).append(f[x][:])
    t, tid = np.concatenate(parts['end_time']), np.concatenate(parts['template_id'])
    o = np.lexsort((tid, t))
    t, tid = t[o], tid[o]
    keep = np.ones(len(o), bool)
    keep[1:] = ~((np.diff(t) < 1e-3) & (np.diff(tid) == 0))
    D = {}
    for x in keys:
        D[x] = t[keep] if x == 'end_time' else tid[keep] if x == 'template_id' else np.concatenate(parts.pop(x))[o][keep]
    return D, len(o) - int(keep.sum())


if __name__ == '__main__':
    out, ins = sys.argv[1], sys.argv[2:]
    D, ndup = merge(ins)
    with h5py.File(out, 'w') as f:
        for x, v in D.items():
            f[x] = v
    print(out, 'files', len(ins), 'peaks', len(D['end_time']), 'duplicates dropped', ndup, 'fields', list(D))
