"""Kill test for exact_stage on the CPU (numpy reference path), against today's dense definitions.

Real bank_v2 IMRPhenomD templates through imr_interp exactly as fp_filter makes them (flow 20 Hz, df 1/1024 Hz,
flen 2^20 + 1), a PSD conditioned like fp_filter's (aLIGO O4 curve plus narrow lines, 16 s inverse-spectrum
truncation), Gaussian noise with two injections, over-whitened into D. Checks:
  - edges from moments equal multi_filter.edges_per_template on the dense H within 1 bin;
  - z and pw from the regenerated binned sums equal a direct float64 sum over the dense H (snr, phase, chisq errors).
The design is killed if any edge differs by more than 1 bin, more than 1e-3 of edges differ (candidate-2 DESIGN),
or the chisq relative error p99 exceeds 1e-3.
With cupy present it also checks the CUDA kernels against the numpy reference.

Run: python tests/test_exact_stage.py [--n-edges 64] [--n-peak-rows 12] [--json OUT]
"""
import argparse, json, os, sys, time
from types import SimpleNamespace
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline')
sys.path.insert(0, HERE)

from exact_stage import (ExactStage, TemplateTile, Coarse, Grid, SegmentData, PeakRequest, compare_values,
                         _BACKENDS, _moments, _plan)
from imr_interp import coarse_grid, coarse_generate, FineInterpolator
from multi_filter import edges_per_template

BANK = os.environ.get('GW_BANK', os.path.join(os.path.dirname(HERE), 'data', 'banks', 'bank_v2.hdf'))
CHISQ_BINS = "0.72*get_freq('fSEOBNRv4Peak',params.mass1,params.mass2,params.spin1z,params.spin2z)**0.7"
FLOW, DF, FLEN = 20.0, 1.0 / 1024, 2 ** 20 + 1
TLEN, SR = 2 * (FLEN - 1), 2048.0
LINES = [(f, 1e3, 0.02) for f in (60.0, 120.0, 180.0)] + [(f, 1e4, 0.01) for f in (17.1, 35.9, 410.3, 434.9)] + \
        [(500.0 + 1.7 * i, 1e5, 0.05) for i in range(6)] + [(1000.0 + 2.1 * i, 1e5, 0.05) for i in range(4)] + \
        [(30.0 + i, 10.0, 0.005) for i in range(11)]


def templates(n):
    import pycbc.waveform, pycbc.conversions as cv
    from pycbc.types import complex64
    from pycbc.vetoes.chisq import SingleDetPowerChisq
    bank = pycbc.waveform.FilterBank(BANK, FLEN, DF, complex64, low_frequency_cutoff=FLOW, approximant='IMRPhenomD')
    kmin = int(FLOW / DF)
    f_end = bank.end_frequency(0)
    kend = int(f_end / DF) if f_end and f_end < (FLEN - 1) * DF else FLEN - 1
    m1, m2 = np.asarray(bank.table.mass1), np.asarray(bank.table.mass2)
    order = np.argsort(cv.tau0_from_mass1_mass2(m1, m2, FLOW), kind='stable')
    pick = order[np.unique(np.round(np.linspace(0, len(order) - 1, n)).astype(int))]
    rows = [bank.table[int(i)] for i in pick]
    nbins = np.array([int(SingleDetPowerChisq.parse_option(SimpleNamespace(params=r), CHISQ_BINS)) for r in rows])
    x = coarse_grid(FLOW, kend * DF)
    amp, coef, step = coarse_generate(rows, x, 'IMRPhenomD', FLOW)
    interp = FineInterpolator(x, DF, kmin, kend, FLEN, np)
    H = np.concatenate([interp(amp[c:c + 4], coef[c:c + 4]) for c in range(0, len(rows), 4)])
    return SimpleNamespace(pick=pick, mtot=(m1 + m2)[pick], nbins=nbins, kmin=kmin, kend=kend, x=x, amp=amp, coef=coef,
                           step=step, interp=interp, H=H)


def conditioned_psd(kmin):
    import pycbc.psd
    from pycbc import DYN_RANGE_FAC
    psd = pycbc.psd.from_string('aLIGOAdVO4T1800545', FLEN, DF, 15.0)
    f = psd.sample_frequencies.numpy()
    raw = psd.numpy() * DYN_RANGE_FAC ** 2          # pycbc's scaling, as fp_filter's dyn_range_factor
    for f0, height, width in LINES:
        raw *= 1 + height / (1 + ((f - f0) / width) ** 2)
    psd.data[:] = raw
    trunc = pycbc.psd.inverse_spectrum_truncation(psd, int(16 * SR), low_frequency_cutoff=FLOW, trunc_method='hann')
    p = trunc.numpy().real.astype(np.float64)
    ip = np.zeros(FLEN)
    ok = np.arange(FLEN) >= kmin
    ip[ok] = 1.0 / p[ok]
    return raw, ip.astype(np.float32)


def segment(T, raw_psd, IP, peak_rows, rng):
    """Over-whitened data: noise with the raw PSD (E|n_k|^2 = S_k / (2 df)), plus a mid-duration peak row 'loud' at
    SNR 100 and the longest peak row 'quiet' at SNR 20, whitened with the truncated PSD."""
    sd = np.sqrt(np.where(raw_psd > 0, raw_psd, 0) / (4 * DF))
    s = sd * (rng.standard_normal(FLEN) + 1j * rng.standard_normal(FLEN))
    k = np.arange(FLEN)
    sig = np.array([4 * DF * (np.abs(h.astype(np.complex128)) ** 2 @ IP.astype(np.float64)) for h in T.H])
    inj = dict(loud=(peak_rows[len(peak_rows) // 2], 100.0, int(0.40 * TLEN)), quiet=(peak_rows[-1], 20.0, int(0.75 * TLEN)))
    for row, rho, t0 in inj.values():
        s = s + rho / np.sqrt(sig[row]) * T.H[row].astype(np.complex128) * np.exp(-2j * np.pi * k * t0 / TLEN)
    return (s * IP).astype(np.complex64), sig, inj


def snr_series(H_row, D, sig):
    q = np.zeros(TLEN, np.complex128)
    q[:FLEN] = np.conj(H_row.astype(np.complex128)) * D
    return np.abs(np.fft.ifft(q)) * TLEN * 4 * DF / np.sqrt(sig)


def pick_peaks(T, D, sig, inj, peak_rows, rng):
    """Per row the loudest sample of some 1 s windows (as the filter's candidates): both injection windows first,
    then random ones, with 1 to 12 peaks per row so items of 1..8 peaks and rows split across items all occur. Also
    a duplicated peak and the first and last samples of the segment."""
    counts = [1, 2, 3, 4, 5, 8, 9, 12, 6, 7, 2, 3, 4, 5, 6, 7]
    rows, samples = [], []
    for i, r in enumerate(peak_rows):
        zt = snr_series(T.H[r], D, sig[r])
        cen = [t0 for _, _, t0 in inj.values()] + list(rng.integers(1024, TLEN - 1024, 32))
        for c in cen[:counts[i % len(counts)]]:
            rows.append(r); samples.append(c - 1024 + int(np.argmax(zt[c - 1024:c + 1024])))
    rows += [peak_rows[0], peak_rows[1], peak_rows[1]]; samples += [samples[0], 0, TLEN - 1]
    return np.array(rows), np.array(samples)


def dense_float64(H, D, rows, samples, E, nb):
    """The definition: Z_j = sum_{k in bin j} conj(H_k) D_k exp(2 pi i k t / N) in float64; z = sum_j Z_j and
    pw = sum_j |Z_j|^2."""
    z = np.empty(len(rows), np.complex128); pw = np.empty(len(rows))
    for r in np.unique(rows):
        for i in np.flatnonzero(rows == r):
            e = E[i, :max(nb[r], 1) + 1]
            k = np.arange(e[0], e[-1], dtype=np.int64)
            q = np.conj(H[r, e[0]:e[-1]].astype(np.complex128)) * D[e[0]:e[-1]].astype(np.complex128)
            x = q * np.exp(2j * np.pi * ((k * int(samples[i])) % TLEN) / TLEN)
            b = np.searchsorted(e, k, side='right') - 1
            Z = np.bincount(b, x.real, len(e) - 1) + 1j * np.bincount(b, x.imag, len(e) - 1)
            z[i] = Z.sum(); pw[i] = (np.abs(Z) ** 2).sum()
    return z, pw


def cuda_vs_numpy(T, grid, D, IP, req, E_np, out):
    """Kernel against reference, on a host with cupy: edges kernel against _edges_numpy (+-1 bin), and regen_bins +
    finalize against _binned_numpy on the same edges."""
    import cupy as cp
    tile_np = TemplateTile(T.H, T.nbins, T.kmin, np.full(len(T.pick), T.kend), Coarse(grid, T.amp, T.coef))
    tile_cp = TemplateTile(cp.asarray(T.H), T.nbins, T.kmin, np.full(len(T.pick), T.kend), Coarse(grid, T.amp, T.coef))
    st_np, st_cp = ExactStage(tile_np), ExactStage(tile_cp)
    seg_cp = SegmentData(cp.asarray(D), cp.asarray(IP), DF, TLEN)
    rows_u = np.arange(len(T.pick))
    E_cp = st_cp.edges(seg_cp, rows_u)
    d = np.abs(E_cp - E_np)
    plan = _plan(req, len(T.pick), TLEN)
    nbe = np.maximum(T.nbins[plan.rows_u], 1)
    E_u = E_np[plan.rows_u]
    zs_np, pw_np = _BACKENDS['numpy'].binned(st_np.state, SegmentData(D, IP, DF, TLEN), plan, E_u, nbe)
    zs_cp, pw_cp = _BACKENDS['cuda'].binned(st_cp.state, seg_cp, plan, E_u, nbe)
    out['cuda_vs_numpy'] = dict(edge_frac_diff=float((d > 0).mean()), edge_max_diff=int(d.max()),
                                **compare_values(zs_cp, pw_cp, zs_np, pw_np, T.nbins[plan.rows_u[plan.peak_u]]))
    pv = st_cp.evaluate(seg_cp, req)
    out['cuda_self_check'] = st_cp.check
    assert d.max() <= 1 and out['cuda_vs_numpy']['chisq_rel_max'] < 1e-4, out['cuda_vs_numpy']
    return pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-edges', type=int, default=64, help='bank_v2 templates for the edge test, spread over duration')
    ap.add_argument('--n-peak-rows', type=int, default=12, help='of those, templates that get peaks')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--json')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {}
    t0 = time.perf_counter()
    T = templates(a.n_edges)
    peak_rows = np.unique(np.round(np.linspace(0, len(T.pick) - 1, a.n_peak_rows)).astype(int))
    raw, IP = conditioned_psd(T.kmin)
    D, sig, inj = segment(T, raw, IP, peak_rows, rng)
    grid = Grid.from_interp(T.interp)
    tile = TemplateTile(T.H, T.nbins, T.kmin, np.full(len(T.pick), T.kend), Coarse(grid, T.amp, T.coef))
    stage = ExactStage(tile)
    seg = SegmentData(D, IP, DF, TLEN)
    out['setup'] = dict(n_templates=len(T.pick), mtot=[round(float(m), 2) for m in T.mtot], nbins=T.nbins.tolist(),
                        nx=len(T.x), kmin=T.kmin, kend=T.kend, max_phase_step_rad=T.step,
                        setup_s=round(time.perf_counter() - t0, 1))

    rows_u = np.arange(len(T.pick))
    t1 = time.perf_counter()
    E = stage.edges(seg, rows_u)
    t2 = time.perf_counter()
    E_ref = np.full(E.shape, T.kend, np.int64)
    for r in np.array_split(rows_u, -(-len(rows_u) // 16)):
        e = edges_per_template(np, r, np.maximum(T.nbins[r], 1), T.kmin, np.full(len(r), T.kend), T.H, IP)
        E_ref[r, :e.shape[1]] = e
    inside = np.arange(E.shape[1])[None, :] <= T.nbins[:, None]
    d = np.abs(E - E_ref)[inside]
    width = np.diff(E, axis=1)[np.arange(E.shape[1] - 1)[None, :] < T.nbins[:, None]]
    mom = _moments(np, grid, grid.kmin, grid.kend, IP)
    A = T.amp
    tot = (A[:, :-1] ** 2 * mom[1][:, 0] + 2 * A[:, :-1] * A[:, 1:] * mom[1][:, 1] + A[:, 1:] ** 2 * mom[1][:, 2]).sum(1)
    out['edges'] = dict(n=int(d.size), n_diff=int((d > 0).sum()), frac_diff=float((d > 0).mean()),
                        max_abs_diff=int(d.max()), min_bin_width=int(width.min()),
                        sigmasq_rel_max=float(np.max(np.abs(4 * DF * tot - sig) / sig)),
                        moments_edges_s=round(t2 - t1, 2))

    rows, samples = pick_peaks(T, D, sig, inj, peak_rows, rng)
    req = PeakRequest(rows, samples)
    t3 = time.perf_counter()
    pv = stage.evaluate(seg, req)
    t4 = time.perf_counter()
    norm = 4 * DF / np.sqrt(sig[rows])
    z_ref, pw_ref = dense_float64(T.H, D, rows, samples, E_ref[rows], T.nbins)
    z_own, pw_own = dense_float64(T.H, D, rows, samples, E[rows], T.nbins)
    snr = np.abs(z_ref) * norm
    chisq = (T.nbins[rows] * pw_ref - np.abs(z_ref) ** 2) * norm ** 2
    fpeak_ref = (2 * E_ref[rows_u, np.maximum(T.nbins - 1, 0)] - E_ref[rows_u, np.maximum(T.nbins - 2, 0)]) * DF
    out['peaks'] = dict(n=len(rows), n_rows=len(np.unique(rows)), snr_min=float(snr.min()), snr_max=float(snr.max()),
                        reduced_chisq_min=float((chisq / (2 * T.nbins[rows] - 2)).min()),
                        reduced_chisq_max=float((chisq / (2 * T.nbins[rows] - 2)).max()),
                        evaluate_s_with_self_check=round(t4 - t3, 1))
    out['vs_dense_todays_edges'] = compare_values(pv.z, pv.pw, z_ref, pw_ref, T.nbins[rows])
    out['vs_dense_same_edges'] = compare_values(pv.z, pv.pw, z_own, pw_own, T.nbins[rows])
    loud = snr > 50
    out['vs_dense_todays_edges_snr_gt_50'] = compare_values(pv.z[loud], pv.pw[loud], z_ref[loud], pw_ref[loud],
                                                            T.nbins[rows[loud]])
    out['fpeak_max_abs_diff_hz'] = float(np.abs(pv.fpeak - fpeak_ref[rows]).max())
    out['self_check'] = stage.check
    try:
        import cupy  # noqa: F401
        has_cupy = True
    except ImportError:
        has_cupy = False
    if has_cupy:
        cuda_vs_numpy(T, grid, D, IP, req, E, out)
    else:
        out['cuda_vs_numpy'] = 'skipped: no cupy on this host'
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)

    v = out['vs_dense_todays_edges']
    assert out['edges']['max_abs_diff'] <= 1, f"KILL: an edge differs by {out['edges']['max_abs_diff']} bins"
    assert out['edges']['frac_diff'] <= 1e-3, f"KILL: {out['edges']['frac_diff']:.3g} of edges differ"
    assert v['chisq_rel_p99'] <= 1e-3, f"KILL: chisq relative error p99 {v['chisq_rel_p99']:.3g} > 1e-3"
    assert out['vs_dense_same_edges']['chisq_rel_max'] <= 1e-3 and v['snr_rel_max'] <= 1e-4 and v['phase_max'] <= 1e-4
    print('PASS: edges within 1 bin and under 1e-3 differing, chisq p99 within 1e-3')


if __name__ == '__main__':
    main()
