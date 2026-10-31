"""Compare two fp_filter output trees (RUN/grp_NN/peaks*/*.hdf and timing.json): bitwise per dataset, then the
frozen-run statistics (compare_stats.py) if anything differs, and the work counters. Usage: cmp_runs.py A B"""
import glob, json, os, sys
import h5py, numpy as np

A, B = sys.argv[1:3]
COUNTERS = ('n_prefilter_in', 'n_prefilter_out', 'n_rebuilt_peaks', 'n_exact_candidates', 'n_candidates',
            'n_chisq_points', 'n_sg_chisq_points', 'n_consolidated_out')
files = sorted(glob.glob(f'{A}/grp_*/peaks*/*.hdf'))
same = diff = 0; ndiff = []; rs = []; nsd = []; nkeys = [0, 0, 0]
for fa in files:
    fb = os.path.join(B, os.path.relpath(fa, A))
    with h5py.File(fa) as a, h5py.File(fb) as b:
        ka = sorted(k for k in a.keys() if isinstance(a[k], h5py.Dataset))
        eq = ka == sorted(k for k in b.keys() if isinstance(b[k], h5py.Dataset)) and \
            all(a[k].shape == b[k].shape and np.array_equal(a[k][:], b[k][:], equal_nan=True) for k in ka)
        if eq:
            same += 1
            nkeys[0] += len(a['snr']); continue
        diff += 1; ndiff.append(os.path.relpath(fa, A))
        ta, tb_ = a['template_id'][:], b['template_id'][:]
        ea, eb = np.round(a['end_time'][:] * 2048).astype(np.int64), np.round(b['end_time'][:] * 2048).astype(np.int64)
        kA, kB = ta * (1 << 40) + ea, tb_ * (1 << 40) + eb
        com, ia, ib = np.intersect1d(kA, kB, return_indices=True)
        nkeys[0] += len(com); nkeys[1] += len(kA) - len(com); nkeys[2] += len(kB) - len(com)
        sa, sb = a['snr'][:][ia], b['snr'][:][ib]; rs.append(np.abs(sa - sb) / sa)
        na, nb = a['newsnr'][:][ia], b['newsnr'][:][ib]; nsd.append(np.abs(na - nb)[na > 6])
print(f'peak files: {len(files)} compared, {same} bitwise identical, {diff} differ', *ndiff[:6])
if diff:
    r = np.concatenate(rs); d = np.concatenate(nsd)
    print(f'keys matched {nkeys[0]}, only in A {nkeys[1]}, only in B {nkeys[2]}; '
          f'snr rel diff p99 {np.percentile(r, 99):.1e} p99.99 {np.percentile(r, 99.99):.1e} max {r.max():.1e}; '
          f'newsnr>6 diff max {d.max() if len(d) else 0:.2e}')
for ta in sorted(glob.glob(f'{A}/grp_*/timing.json')):
    g = os.path.basename(os.path.dirname(ta))
    x, y = json.load(open(ta)), json.load(open(os.path.join(B, g, 'timing.json')))
    c = {k: (x.get(k), y.get(k)) for k in COUNTERS if x.get(k) != y.get(k)}
    print(f'{g}: total_s {x["total_s"]:.1f} vs {y["total_s"]:.1f}; gpu_peak_gib {x.get("gpu_peak_gib", 0):.2f} vs '
          f'{y.get("gpu_peak_gib", 0):.2f}; counters ' + ('identical' if not c else f'DIFFER {c}'))
