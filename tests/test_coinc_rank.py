#!/usr/bin/env python
"""Self-test for coinc_rank.py on fake noise triggers. Usage: test_coinc_rank.py [quadsum|lnl]

1. Calibration: in pure noise, the number of real-data (zero-lag) candidates with FAR <= f must be
   about f * live_time. Checked over several noise realisations.
2. Recovery: a fake signal, very loud in H1 (SNR 20) and weak in L1 (SNR 6), 5 ms apart, in a few
   neighbouring templates. Its H1 trigger pairs with L1 noise in the slides, so its inclusive FAR is
   inflated by itself; the exclusive FAR must remove that and sit at the floor.
"""
import json, subprocess, sys, os, tempfile
import numpy as np
import h5py

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline')
YEAR = 365.25 * 86400


def fake(path, rng, t0, P, ntmp, rate, signal=None, sig_phase=0.0):
    n = rng.poisson(rate * P * ntmp)
    tid = rng.integers(0, ntmp, n)
    t = t0 + rng.uniform(0, P, n)
    # steep noise tail, a little steeper in odd templates so per-template fits have something to find
    snr = 4.5 + rng.exponential(np.where(tid % 2, 0.35, 0.5), n)
    ph = rng.uniform(-np.pi, np.pi, n)
    if signal:
        ts, tids, s = signal
        tid = np.concatenate([tid, tids]); t = np.concatenate([t, np.full(len(tids), ts)])
        snr = np.concatenate([snr, s]); ph = np.concatenate([ph, np.full(len(tids), sig_phase)])
    with h5py.File(path, 'w') as f:
        f['template_id'] = tid; f['end_time'] = t; f['snr'] = snr
        f['coa_phase'] = ph; f['peak_sigmasq'] = np.full(len(t), 1e8)


def run(tmp, P, seed, signal=False, stat='quadsum'):
    rng = np.random.default_rng(seed)
    t0, ntmp, rate = 1e9, 50, 0.05
    sig_h = sig_l = None
    if signal:
        ts = t0 + 0.37 * P
        tids = np.arange(20, 25)
        sig_h = (ts, tids, np.array([18.0, 19.0, 20.0, 19.2, 18.3]))
        sig_l = (ts + 0.005, tids, np.array([5.0, 5.6, 6.0, 5.7, 5.2]))
    fake(f'{tmp}/H1.hdf', rng, t0, P, ntmp, rate, sig_h, 0.0)
    fake(f'{tmp}/L1.hdf', rng, t0, P, ntmp, rate, sig_l, 3.0)
    out = f'{tmp}/c{seed}'
    subprocess.run([sys.executable, f'{HERE}/coinc_rank.py', '--peaks', f'H1={tmp}/H1.hdf', '--peaks', f'L1={tmp}/L1.hdf',
                    '--span', str(t0), str(t0 + P), '--out', out + '.hdf', '--top', '3', '--min-rank', '4.5', '--stat', stat, '--fit-thr', '5.5', '--phasetd-cache', f'{tmp}/phasetd.npz'],
                   check=True, stdout=subprocess.DEVNULL if not signal else None)
    with h5py.File(out + '.hdf') as f:
        far = f['foreground/far_incl_per_yr'][:] / YEAR
        fe = f['foreground/far_excl_per_yr'][:] / YEAR
        h1t = f['foreground/H1_time'][:]
        floor = json.loads(open(out + '.json').read())['far_floor_per_yr'] / YEAR
    return far, fe, h1t, floor, (sig_h[0] if signal else None)


stat = sys.argv[1] if len(sys.argv) > 1 else 'quadsum'
print('statistic:', stat)
with tempfile.TemporaryDirectory() as tmp:
    P = 20000.0
    thresholds = np.array([1 / P, 3 / P, 10 / P, 30 / P])   # expect 1, 3, 10, 30 per realisation
    counts = []
    for seed in range(8):
        far, fe, *_ = run(tmp, P, seed, stat=stat)
        counts.append([(far <= f).sum() for f in thresholds])
    counts = np.array(counts)
    print('calibration: expected per run', thresholds * P)
    print('              observed mean ', counts.mean(0), ' (8 runs)')
    ok_cal = np.all(np.abs(counts.mean(0) - thresholds * P) <= 3 * np.sqrt(thresholds * P / 8) + 0.5)
    far, fe, h1t, floor, ts = run(tmp, P, 99, signal=True, stat=stat)
    ok_sig = abs(h1t[0] - ts) < 0.01 and fe[0] <= floor and far[0] > floor
    print('signal: top candidate at', h1t[0], 'injected', ts, 'FAR_excl', fe[0] * YEAR, 'FAR_incl', far[0] * YEAR, '/yr (floor', floor * YEAR, ')')
    print('PASS' if ok_cal and ok_sig else 'FAIL', 'calibration' if ok_cal else 'calibration-bad', 'signal' if ok_sig else 'signal-bad')
    sys.exit(0 if ok_cal and ok_sig else 1)
