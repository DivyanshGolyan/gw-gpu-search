"""GPU check: interpolated IMR templates vs PyCBC's direct fine-grid templates, PSD-weighted match.
Usage: test_interp.py BANK.hdf STORE.hdf IFO GPS_START GPS_END OUT.json [--n 50]
The PSD is a median Welch estimate (16 s) from the store, as in prep_data.py."""
import sys, json, time, argparse, os
import numpy as np, cupy as cp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline'))
from imr_interp import coarse_grid, coarse_generate, FineInterpolator
import pycbc.waveform
from pycbc.types import complex64
ap = argparse.ArgumentParser(); ap.add_argument('bank'); ap.add_argument('store'); ap.add_argument('ifo')
ap.add_argument('start', type=int); ap.add_argument('end', type=int); ap.add_argument('out')
ap.add_argument('--n', type=int, default=50); ap.add_argument('--n-chirp', type=int, default=6000); ap.add_argument('--n-log', type=int, default=1500)
a = ap.parse_args()
df = 1 / 1024; flen = 2 ** 20 + 1; flow = 30.0; kmin = int(flow / df); kend = flen - 1
bank = pycbc.waveform.FilterBank(a.bank, flen, df, complex64, low_frequency_cutoff=flow, approximant='IMRPhenomD')
m = np.array(bank.table.mass1) + np.array(bank.table.mass2)
rng = np.random.default_rng(0)
order = np.argsort(m)
pick = np.unique(np.r_[order[:10], order[-5:], rng.choice(len(m), a.n - 15, replace=False)])[:a.n]
import pycbc.psd, pycbc.filter
from pycbc.frame.store import read_store
ts = read_store(a.store, f'{a.ifo}:GWOSC_STRAIN', a.start, a.end)
ts = pycbc.filter.resample_to_delta_t(pycbc.filter.highpass(ts, 15.0), 1.0 / 2048)
pp = pycbc.psd.welch(ts, seg_len=16 * 2048, seg_stride=8 * 2048, avg_method='median')
p = np.c_[pp.sample_frequencies.numpy(), pp.numpy()]; fk = np.arange(flen) * df
w = np.zeros(flen); w[kmin:kend] = 1 / np.interp(fk[kmin:kend], p[:, 0], p[:, 1])
x = coarse_grid(flow, kend * df, a.n_chirp, a.n_log)
t0 = time.perf_counter(); amp, ph, step = coarse_generate([bank.table[i] for i in pick], x, 'IMRPhenomD', flow); tc = time.perf_counter() - t0
fi = FineInterpolator(x, df, kmin, kend, flen, cp)
H = fi(amp, ph); cp.cuda.Device().synchronize()
t0 = time.perf_counter(); H = fi(amp, ph); cp.cuda.Device().synchronize(); tf = time.perf_counter() - t0
H = H.get()
t0 = time.perf_counter(); D = np.stack([bank[i].numpy() for i in pick]); td = time.perf_counter() - t0
W = cp.asarray(w)
res = []
for j, i in enumerate(pick):
    a_, b_ = cp.asarray(D[j]), cp.asarray(H[j])
    na = float(cp.sqrt((cp.abs(a_) ** 2 * W).sum())); nb = float(cp.sqrt((cp.abs(b_) ** 2 * W).sum()))
    prod = cp.conj(a_) * b_ * W
    ov = float(cp.abs(prod.sum())) / na / nb
    z = cp.zeros(2 * (flen - 1), cp.complex128); z[:flen] = prod
    mt = float(cp.abs(cp.fft.ifft(z)).max()) * len(z) / na / nb
    res.append(dict(index=int(i), mtot=float(m[i]), overlap_no_time_shift=ov, match=mt, norm_ratio=nb / na))
mm = np.array([r['match'] for r in res]); ov = np.array([r['overlap_no_time_shift'] for r in res])
summ = dict(n=len(res), coarse_points=len(x), max_phase_step_rad=step, match_min=float(mm.min()), match_median=float(np.median(mm)),
            overlap_min=float(ov.min()), overlap_median=float(np.median(ov)),
            coarse_gen_s_per_template=tc / len(pick), gpu_interp_s_per_template=tf / len(pick), direct_gen_s_per_template=td / len(pick),
            lightest=min(res, key=lambda r: r['mtot']), worst=min(res, key=lambda r: r['match']))
print(json.dumps(summ, indent=1))
json.dump(dict(summary=summ, rows=res), open(a.out, 'w'), indent=1)
