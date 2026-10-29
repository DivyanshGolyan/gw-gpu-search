#!/usr/bin/env python
"""Gate for gpu_prep.py: condition datasets with PyCBC (fp_filter.prep_one) and with gpu_prep.prep, compare.

Usage: test_gpu_prep.py --argv inspiral_argv.txt --datasets datasets.txt [--injection-file F] [--xp cupy|numpy]
       [--reps 3] [--out result.json] [--ref-cache DIR]
datasets.txt lines: "IFO GPS_START GPS_END STORE [TRIG_START]" (as fp_filter). Writes per-dataset comparisons and
timings to --out; exit 1 if any dataset fails the gate.
Gate: same segments and gates (times within 1 us); per segment ||D_gpu - D_ref|| / ||D_ref|| and the same for IP
<= 1e-3; a chirp-weighted sigma^2 (sum f^-7/3 IP) within 1e-5; and both GPU-vs-PyCBC differences no more than 3x
PyCBC-vs-PyCBC (PyCBC is run twice; floor 1e-4). Why not 1e-7: PyCBC itself does not reproduce run to run beyond ~1e-4
in whitened high-frequency bins. Its float32 FFT of the raw strain rounds at ~1e-7 of the total power, which sits
below 20 Hz; whitening divides by a PSD 1e3-1e4 smaller at high frequency, so that rounding shows up at ~5e-5. FFTW
picks aligned or unaligned codelets per call, so two PyCBC runs differ by that much; cuFFT differs by the same.
"""
import argparse, json, os, shlex, sys, tempfile, time
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline')
sys.path.insert(0, HERE)
import gpu_prep

TOL_D, TOL_IP, TOL_SIG, TOL_T, REF_X, REF_FLOOR = 1e-3, 1e-3, 1e-5, 1e-6, 3.0, 1e-4


def rel_l2(x, y):
    return [float(np.linalg.norm(x[i] - y[i]) / np.linalg.norm(y[i])) for i in range(len(y))]


def sigma_rel(I1, I0, df, kmin):
    w = np.zeros(I0.shape[1]); f = np.arange(kmin, I0.shape[1]) * df; w[kmin:] = f ** (-7 / 3.)
    return float(np.abs((I1 * w).sum(1) / (I0 * w).sum(1) - 1).max())


def compare(ref, refz, ref2z, got, kmin):
    D0, I0 = refz['D'], refz['IP']
    D1, I1 = got['D'], got['IP']
    r = dict(nseg_ref=len(D0), nseg_gpu=len(D1), gates_ref=ref['gates'], gates_gpu=got['gates'])
    r['meta_ok'] = bool(len(D0) == len(D1) and ref['flen'] == got['flen'] and ref['tlen'] == got['tlen']
                        and [list(map(float, a)) for a in ref['analyze']] == [list(map(float, a)) for a in got['analyze']]
                        and list(map(float, ref['seg_start'])) == list(map(float, got['seg_start'])))
    r['gates_ok'] = bool(len(ref['gates']) == len(got['gates']) and all(
        abs(a[0] - b[0]) <= TOL_T and a[1:] == tuple(b[1:]) for a, b in zip(map(tuple, ref['gates']), map(tuple, got['gates']))))
    if not r['meta_ok']:
        return r
    r.update(D_rel_l2=rel_l2(D1, D0), IP_rel_l2=rel_l2(I1, I0), sigmasq_rel=sigma_rel(I1, I0, ref['df'], kmin),
             ref_D_rel_l2=rel_l2(ref2z['D'], D0), ref_IP_rel_l2=rel_l2(ref2z['IP'], I0),
             ref_sigmasq_rel=sigma_rel(ref2z['IP'], I0, ref['df'], kmin), IP_zero_below_flow=bool((I1[:, :kmin] == 0).all()))
    ipr = np.abs(I1[:, kmin:-1].astype(np.float64) - I0[:, kmin:-1]) / I0[:, kmin:-1]
    live = I0[:, kmin:-1] >= 1e-3 * np.median(I0[:, kmin:-1], axis=1)[:, None]
    r.update(IP_bin_rel_p99_live=float(np.percentile(ipr[live], 99)), IP_bin_rel_max_live=float(ipr[live].max()))
    lim_d = REF_X * max(max(r['ref_D_rel_l2']), REF_FLOOR); lim_i = REF_X * max(max(r['ref_IP_rel_l2']), REF_FLOOR)
    r['pass'] = bool(r['gates_ok'] and max(r['D_rel_l2']) <= min(TOL_D, lim_d) and max(r['IP_rel_l2']) <= min(TOL_IP, lim_i)
                     and r['sigmasq_rel'] <= TOL_SIG and r['IP_zero_below_flow'])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--argv', required=True); ap.add_argument('--datasets', required=True)
    ap.add_argument('--injection-file'); ap.add_argument('--xp', default='cupy'); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', default='test_gpu_prep.json'); ap.add_argument('--ref-cache')
    a = ap.parse_args()
    from multi_filter import dataset_argv
    import fp_filter
    base = shlex.split(open(a.argv).read())
    if a.injection_file:
        base += ['--injection-file', a.injection_file]
    xp = np if a.xp == 'numpy' else __import__('cupy')
    res = dict(xp=a.xp, injection_file=a.injection_file, datasets=[])
    ok = True
    for line in open(a.datasets):
        if not line.strip() or line.startswith('#'):
            continue
        f = line.split()
        argv = dataset_argv(base, f[0], int(f[1]), int(f[2]), *f[3:])
        tag = '-'.join(f[:3])
        cache = a.ref_cache or tempfile.mkdtemp()
        os.makedirs(cache, exist_ok=True)
        t = time.perf_counter()
        ref = fp_filter.prep_one((tag, argv, cache, cache, 0))
        t_ref = time.perf_counter() - t
        refz = np.load(ref['file'])
        if not a.ref_cache:
            os.remove(ref['file'])
        t2 = tempfile.mkdtemp()                 # second PyCBC run, uncached: PyCBC's own run-to-run spread
        ref2 = fp_filter.prep_one((tag, argv, t2, None, 0))
        ref2z = dict(np.load(ref2['file'])); os.remove(ref2['file'])
        if xp is not np:
            import pycbc
            assert pycbc.DYN_RANGE_FAC == gpu_prep.DYN_RANGE_FAC
        times, got = [], None
        inj = None
        if a.injection_file:   # injections are CPU work that production overlaps with the GPU: time them apart
            o, _ = gpu_prep.parser().parse_known_args(argv)
            t = time.perf_counter()
            n = int(round((o.gps_end_time - o.gps_start_time + 2 * o.pad_data) * o.sample_rate))
            inj = gpu_prep.injection_signal(o, o.gps_start_time - o.pad_data, n, o.sample_rate)
            t_inj = time.perf_counter() - t
        for rep in range(a.reps):
            t = time.perf_counter()
            g = gpu_prep.prep(argv, xp, inj=inj)
            times.append(dict(wall=time.perf_counter() - t, **g['timing']))
            if got is None:
                got = dict(g, D=g['D'].get() if xp is not np else g['D'], IP=g['IP'].get() if xp is not np else g['IP'])
        kmin = int(gpu_prep.parser().parse_known_args(argv)[0].low_frequency_cutoff / ref['df'])
        r = compare(ref, refz, ref2z, got, kmin)
        r.update(tag=tag, ref_prep_s=t_ref, gpu_prep_runs=times, inj_cpu_s=t_inj if a.injection_file else None)
        print(json.dumps({k: v for k, v in r.items() if k != 'gpu_prep_runs'}), flush=True)
        print('  gpu timings:', json.dumps(times), flush=True)
        ok &= r.get('pass', False)
        res['datasets'].append(r)
    res['pass'] = bool(ok)
    json.dump(res, open(a.out + '.part', 'w'), indent=1); os.replace(a.out + '.part', a.out)
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
