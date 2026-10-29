#!/usr/bin/env python
"""H1+L1 coincidence, time-slide background and false-alarm rates for batched_filter.py peaks.

What it does, in order (PyCBC's standard offline method, 'quadsum' statistic):
  1. Read each detector's peak file (template_id, end_time, snr, optional chisq/chisq_dof/newsnr).
     Single-detector score = newsnr (SNR down-weighted when chi-squared says the shape is wrong);
     plain SNR if the file has no chi-squared.
  2. Keep only times both detectors analysed (--segments or --span), and squeeze the gaps out so
     that the analysed time is one continuous "live" clock of length P.
  3. Coincidence: same template in both detectors, end times within --window (10 ms light travel
     + slop). Time slides: shift L1 by k * --slide-step on the live clock, wrapping around P,
     for every k with |k| <= --max-slides. k = 0 is the real data ("foreground"), every other k is
     fake data that can only contain noise coincidences ("background").
  4. Score a coincidence. --stat quadsum: sqrt(newsnr_H1^2 + newsnr_L1^2). --stat lnl (default when
     the peak files carry coa_phase and peak_sigmasq): PyCBC-style log likelihood ratio from
     ranking_stat.py (per-template noise-tail fits + H1/L1 phase/time/amplitude consistency, optional
     sensitivity and local-rate terms). Within each slide, keep only the loudest coincidence per
     --cluster-window seconds (one signal rings up many templates).
  5. FAR of a foreground candidate = (background coincidences scoring >= it) / (background time),
     background time = (number of nonzero slides) * P. Reported two ways:
       far_incl: full background.
       far_excl: background with any coincidence that reuses a single-detector trigger of a
                 louder-or-equal foreground candidate removed (PyCBC's "exclusive" background;
                 stops a real loud signal in H1 from inflating its own background). Computed for
                 the --n-excl loudest candidates; the rest keep far_incl.

All pairs are found with sorted searches on (template, time) keys, no per-template Python loop.

Usage:
  coinc_rank.py --peaks H1=h1.hdf --peaks L1=l1.hdf --span START END --out cands.hdf [--bank bank.hdf]
"""
import argparse, json, sys, time
import numpy as np
import h5py

YEAR = 365.25 * 86400


def newsnr(snr, rchisq):
    """PyCBC newsnr: snr / ((1 + rchisq^3) / 2)^(1/6) where reduced chisq > 1, else snr."""
    ns = np.array(snr, dtype=np.float64)
    m = rchisq > 1
    ns[m] = snr[m] / ((1 + rchisq[m] ** 3) / 2.0) ** (1.0 / 6)
    return ns


PEAK_KEYS = ('template_id', 'end_time', 'snr', 'chisq', 'chisq_dof', 'sg_chisq', 'coa_phase', 'peak_sigmasq')


def load_peaks(path):
    """path: one (merged) peak file, or a comma-separated list of piece files to merge in memory."""
    if ',' in path:
        from merge_peaks import merge
        return _load(merge(path.split(','), PEAK_KEYS)[0])
    with h5py.File(path, 'r') as f:
        return _load(f)


def _load(f):
    d = dict(tid=f['template_id'][:].astype(np.int64), t=f['end_time'][:].astype(np.float64),
             snr=f['snr'][:].astype(np.float64))
    if 'chisq' in f:
        d['chisq'] = f['chisq'][:].astype(np.float64)
        d['dof'] = f['chisq_dof'][:].astype(np.float64)
        # chi-squared skipped below --chisq-snr-threshold is stored as 0: those keep plain snr
        d['rank'] = newsnr(d['snr'], d['chisq'] / np.maximum(d['dof'], 1))
        d['has_chisq'] = True
    else:
        d['rank'] = d['snr'].copy()
        d['has_chisq'] = False
    if 'sg_chisq' in f:
        d['sg_chisq'] = f['sg_chisq'][:].astype(np.float64)
    if 'coa_phase' in f and 'peak_sigmasq' in f:
        d['phase'] = f['coa_phase'][:].astype(np.float64); d['sigmasq'] = f['peak_sigmasq'][:].astype(np.float64)
    return d


class LiveClock:
    """Maps GPS times inside a list of disjoint segments onto one continuous clock [0, P)."""
    def __init__(self, segs):
        segs = sorted((float(a), float(b)) for a, b in segs if b > a)
        self.s0 = np.array([a for a, b in segs]); self.s1 = np.array([b for a, b in segs])
        self.off = np.concatenate([[0.0], np.cumsum(self.s1 - self.s0)])[:-1]
        self.P = float((self.s1 - self.s0).sum())

    def inside(self, t):
        i = np.searchsorted(self.s0, t, side='right') - 1
        return (i >= 0) & (t < self.s1[np.clip(i, 0, None)])

    def to_live(self, t):
        i = np.searchsorted(self.s0, t, side='right') - 1
        return self.off[i] + (t - self.s0[i])

    def to_gps(self, x):
        x = np.mod(x, self.P)
        i = np.searchsorted(self.off, x, side='right') - 1
        return self.s0[i] + (x - self.off[i])


def find_pairs(H, L, P, step, K, window, stat_fn, min_stat=-np.inf, budget=20_000_000):
    """All (H1 index, L1 index, slide k, stat) with same template and |tH - (tL + k*step) mod P| <= window,
    for |k| <= K, scored by stat_fn(h_idx, l_idx, dt) with dt = (tL + k*step) - tH; pairs scoring below
    min_stat are dropped (exact for every FAR above min_stat: a dropped pair can't out-cluster a louder one).
    Times are live-clock seconds in [0, P). Works in chunks of about `budget` candidate pairs."""
    reach = K * step + window
    if 2 * reach >= P:
        raise SystemExit(f'--max-slides {K} x --slide-step {step} covers the whole span ({P:.0f} s); lower it')
    # L1 copies at -P, 0, +P so the wrap-around is a plain range search
    lt = np.concatenate([L['x'] - P, L['x'], L['x'] + P])
    li = np.tile(np.arange(len(L['x'])), 3)
    lk = np.tile(L['tid'], 3)
    big = 4.0 * P + 10 * reach
    lkey = lk * big + lt
    o = np.argsort(lkey, kind='stable'); lkey, li, lt = lkey[o], li[o], lt[o]
    hkey = H['tid'] * big + H['x']
    lo_all = np.searchsorted(lkey, hkey - reach, 'left'); hi_all = np.searchsorted(lkey, hkey + reach, 'right')
    n_all = hi_all - lo_all
    cum = np.concatenate([[0], np.cumsum(n_all)])
    bounds = np.unique(np.concatenate([np.searchsorted(cum, np.arange(0, cum[-1], budget), 'right') - 1, [len(hkey)]]))
    out_h, out_l, out_k, out_s = [], [], [], []
    for c0, c1 in zip(bounds[:-1], bounds[1:]):
        lo, n = lo_all[c0:c1], n_all[c0:c1]
        tot = int(n.sum())
        if tot == 0:
            continue
        hidx = np.repeat(np.arange(c0, c1), n)
        # flat index into the L arrays: lo[i], lo[i]+1, ..., hi[i]-1
        lidx = np.repeat(lo - np.concatenate([[0], np.cumsum(n)[:-1]]), n) + np.arange(tot)
        dt = H['x'][hidx] - lt[lidx]                 # tH - tL (tL already unwrapped)
        k = np.rint(dt / step).astype(np.int64)
        ok = (np.abs(dt - k * step) <= window) & (np.abs(k) <= K)
        hh, ll, kk = hidx[ok], li[lidx[ok]], k[ok]
        st = stat_fn(hh, ll, k[ok] * step - dt[ok])
        m = st >= min_stat
        out_h.append(hh[m]); out_l.append(ll[m]); out_k.append(kk[m]); out_s.append(st[m])
    if not out_h:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(out_h), np.concatenate(out_l), np.concatenate(out_k), np.concatenate(out_s)


def prune_singles(H, L, uH, uL, cap, min_stat, reach, P, rounds=3):
    """Masks of H and L triggers that can still pair to stat >= min_stat, given stat <= uH[h] + uL[l] + cap.
    A trigger survives if its bound plus the best partner bound, same template, within +-reach (wrapping on
    [0, P)), reaches min_stat. Bins of width >= reach, so every partner sits in the same or a neighbouring bin."""
    ntmp = int(max(H['tid'].max(), L['tid'].max())) + 1
    nb = min(int(P // reach), 50_000_000 // ntmp)      # wider bins only loosen the bound
    if nb < 3:
        return np.ones(len(uH), bool), np.ones(len(uL), bool)
    B = P / nb
    bins = {id(d): np.minimum((d['x'] // B).astype(np.int64), nb - 1) for d in (H, L)}
    kH, kL = np.ones(len(uH), bool), np.ones(len(uL), bool)
    need = min_stat - cap - 1e-9
    for _ in range(rounds):
        for d, u, k, o, uo, ko in ((H, uH, kH, L, uL, kL), (L, uL, kL, H, uH, kH)):
            best = np.full(ntmp * nb, -np.inf)
            np.maximum.at(best, o['tid'][ko] * nb + bins[id(o)][ko], uo[ko])
            b, t = bins[id(d)], d['tid'] * nb
            bound = np.maximum(np.maximum(best[t + b], best[t + (b - 1) % nb]), best[t + (b + 1) % nb])
            k &= u + bound >= need
    return kH, kL


def keep_loudest(t, rank, interval, num):
    """Indices of the `num` highest-ranked triggers in each `interval` seconds (PyCBC's keep-loudest)."""
    b = np.floor(t / interval).astype(np.int64)
    o = np.lexsort((-rank, b))
    first = np.concatenate([[0], np.nonzero(np.diff(b[o]))[0] + 1])
    pos = np.arange(len(o)) - np.repeat(first, np.diff(np.concatenate([first, [len(o)]])))
    return np.sort(o[pos < num])


def cluster(k, t, stat, win, group=4_000_000):
    """Keep coincidences that are the loudest within +-win seconds in their own slide."""
    if len(k) == 0:
        return np.zeros(0, int)
    o = np.lexsort((t, k)); k_ = k[o]
    # split at slide boundaries into groups of about `group` events, so the range-max table stays small
    starts = np.concatenate([[0], np.nonzero(np.diff(k_))[0] + 1, [len(o)]])
    cuts = np.unique(starts[np.searchsorted(starts, np.arange(0, len(o), group))].tolist() + [len(o)])
    out = [o[c0 + _cluster_sorted(k_[c0:c1], t[o[c0:c1]], stat[o[c0:c1]], win)] for c0, c1 in zip(cuts[:-1], cuts[1:])]
    return np.concatenate(out)


def _cluster_sorted(k_, t_, s_, win):
    """cluster() on events already sorted by (slide, time); returns kept positions."""
    # slide offset must exceed any time difference yet keep float64 resolution (a 1e12 offset left ~8 s steps)
    span = (t_.max() - t_.min()) + 2 * win + 1.0
    key = (k_ - k_.min()) * span + (t_ - t_.min())
    lo = np.searchsorted(key, key - win, 'left'); hi = np.searchsorted(key, key + win, 'right')
    # sparse table for range-max over s_ (O(n log n))
    n = len(s_); levels = [s_]
    j = 1
    while (1 << j) <= n:
        prev = levels[-1]; h = 1 << (j - 1)
        levels.append(np.maximum(prev[:-h], prev[h:]))
        j += 1
    lg = np.floor(np.log2(np.maximum(hi - lo, 1))).astype(int)
    mx = np.empty(n)
    for L in np.unique(lg):
        m = lg == L; a = levels[L]
        mx[m] = np.maximum(a[lo[m]], a[hi[m] - (1 << L)])
    keep = s_ >= mx
    # exact ties inside a window: keep the first only
    idx = np.nonzero(keep)[0]
    if len(idx) > 1:
        same = (k_[idx[1:]] == k_[idx[:-1]]) & (t_[idx[1:]] - t_[idx[:-1]] <= win) & (s_[idx[1:]] == s_[idx[:-1]])
        keep[idx[1:][same]] = False
    return np.nonzero(keep)[0]


def make_stat(a, H, L, P):
    """Returns (stat_fn(h_idx, l_idx, dt), description)."""
    single = 'newsnr' if H['has_chisq'] and L['has_chisq'] else 'snr (no chisq)'
    use = a.stat or ('lnl' if 'phase' in H and 'phase' in L else 'quadsum')
    if use == 'quadsum':
        return (lambda h, l, dt: np.sqrt(H['rank'][h] ** 2 + L['rank'][l] ** 2)), f'quadsum {single}'
    import ranking_stat as rs
    terms = set(a.terms.split(',')) - {''}
    bad = terms - {'fits', 'phasetd', 'sens', 'lrate'}
    if bad:
        sys.exit(f'unknown --terms {bad}')
    ntmp = int(max(H['tid'].max(), L['tid'].max())) + 1
    order = None
    if a.bank:
        dur = rs.template_durations(a.bank)
        ntmp = max(ntmp, len(dur))
        order = np.argsort(dur)
        if a.templates:     # pool the noise fits only over the ranked templates
            order = order[np.isin(order, np.loadtxt(a.templates, dtype=np.int64, ndmin=1))]
        H['dur'] = L['dur'] = dur
    if a.short_dur > 0:
        if not a.bank or not H['has_chisq']:
            sys.exit('--short-dur needs --bank and chi-squared')
        short = H['dur'] < a.short_dur
        for d in (H, L):
            m = short[d['tid']]
            d['rank'][m] /= np.sqrt(np.maximum(1.0, d['chisq'][m] / np.maximum(d['dof'][m], 1)))
    saved = np.load(a.fits_from) if a.fits_from else None
    for name, d in (('H', H), ('L', L)):
        if 'fits' in terms:
            if saved is not None:      # noise model from the search run, so injection runs share its statistic
                fit = (saved[name + '_lra'], saved[name + '_alpha'])
            else:
                fit = rs.fit_noise(d['tid'], d['rank'], ntmp, P, a.fit_thr, order, a.fit_pool)
            d['lnn'] = rs.ln_noise(d['tid'], d['rank'], fit, a.fit_thr)
            d['fit'] = fit
        else:      # no fits: a single exponential with alpha = 6 per detector, i.e. close to quadsum newsnr^2/2
            d['lnn'] = -6.0 * (d['rank'] - a.fit_thr)
        if 'lrate' in terms:
            d['lnn'] = d['lnn'] + np.log(np.clip(rs.local_rate(d['t'], d['rank']), 0.5, 50))
    ptd = rs.PhaseTD(window=a.window, path=a.phasetd_cache) if 'phasetd' in terms else None
    if (ptd is not None or 'sens' in terms) and not ('phase' in H and 'phase' in L):
        sys.exit('phasetd and sens need coa_phase and peak_sigmasq in both peak files')
    sref = np.median(np.sqrt(H['sigmasq'])) if 'sens' in terms else 1.0

    def fn(h, l, dt):
        s = -H['lnn'][h] - L['lnn'][l]
        if ptd is not None:
            r = np.log(L['snr'][l] / H['snr'][h]) - 0.5 * np.log(L['sigmasq'][l] / H['sigmasq'][h])
            s = s + ptd(dt, L['phase'][l] - H['phase'][h], r)
        if 'sens' in terms:
            s = s + 1.5 * np.log(np.sqrt(H['sigmasq'][h] * L['sigmasq'][l]) / sref ** 2)
        return s
    # stat <= -lnn_H - lnn_L + max(phasetd), so single triggers too quiet for --min-stat can be dropped early
    if 'sens' not in terms:
        fn.bound = (-H['lnn'], -L['lnn'], float(ptd.lnp.max()) if ptd is not None else 0.0)
    return fn, f"lnl[{','.join(sorted(terms))}] on {single}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--peaks', action='append', required=True, help='IFO=path, exactly two (e.g. H1=h1.hdf)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--span', nargs=2, type=float, metavar=('START', 'END'), help='one analysed GPS interval')
    g.add_argument('--segments', help='text file, "start end" per line: times both detectors analysed')
    ap.add_argument('--window', type=float, default=0.012, help='coincidence window, s (10 ms light travel + 2 ms)')
    ap.add_argument('--slide-step', type=float, default=0.1, help='time-slide step, s')
    ap.add_argument('--max-slides', type=int, default=None,
                    help='use slides k = -K..K (default: as many as fit in half the span)')
    ap.add_argument('--cluster-window', type=float, default=10.0, help='s; keep loudest per window per slide')
    ap.add_argument('--min-rank', type=float, default=4.0, help='drop single triggers with newsnr below this')
    ap.add_argument('--excl-window', type=float, default=0.1, help='s; exclusive background veto window')
    ap.add_argument('--stat', choices=['quadsum', 'lnl'], default=None,
                    help='coincidence score (default: lnl if peak files have coa_phase and peak_sigmasq)')
    ap.add_argument('--terms', default='fits,phasetd',
                    help='lnl terms, comma list of: fits, phasetd, sens, lrate')
    ap.add_argument('--short-dur', type=float, default=0.0,
                    help='templates shorter than this (s, needs --bank) get an extra chi-squared penalty: '
                         'rank /= sqrt(max(1, reduced chisq)); aimed at glitches in high-mass templates')
    ap.add_argument('--fit-thr', type=float, default=6.0, help='newsnr threshold of the noise-tail fits')
    ap.add_argument('--fit-pool', type=int, default=50, help='templates pooled (nearest in duration) per fit')
    ap.add_argument('--phasetd-cache', default=None, help='.npz cache of the phase-time histogram')
    ap.add_argument('--min-stat', type=float, default=-np.inf, help='drop coincidences scoring below this')
    ap.add_argument('--sgveto', action='store_true', help="apply PyCBC's sine-Gaussian veto to newsnr (needs sg_chisq)")
    ap.add_argument('--save-fits', help='write the noise-tail fits (.npz) for reuse by an injection run')
    ap.add_argument('--fits-from', help='use noise-tail fits saved by a search run (--save-fits)')
    ap.add_argument('--background-from', help='coinc_rank .hdf of the search run: rank zero lag only, using its '
                    'background (use with --fits-from and the same --terms/settings; FAR is a lower bound '
                    'below that run\'s --min-stat; no exclusive FAR)')
    ap.add_argument('--n-excl', type=int, default=200, help='compute exclusive FAR for this many loudest candidates')
    ap.add_argument('--keep-loudest', nargs=2, type=float, metavar=('INTERVAL', 'NUM'),
                    help='per detector, keep only the NUM loudest triggers per INTERVAL s (PyCBC uses 1.072 100)')
    ap.add_argument('--bank', help='template bank hdf, to attach masses and spins to candidates')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--pseudo-shift', type=float, default=0.0,
                    help='s; shift the second detector on the live clock (noise-only pseudo-experiments, R7)')
    ap.add_argument('--templates', help='text file of template ids: rank only these (mass-split ranking)')
    ap.add_argument('--out', required=True, help='.hdf (all foreground + background stats) and .json summary')
    a = ap.parse_args()
    t_start = time.perf_counter()

    ifos = dict(p.split('=', 1) for p in a.peaks)
    if len(ifos) != 2:
        sys.exit('need exactly two --peaks IFO=path')
    (n1, f1), (n2, f2) = sorted(ifos.items())
    segs = [tuple(a.span)] if a.span else [tuple(map(float, l.split()[:2])) for l in open(a.segments) if l.strip() and not l.startswith('#')]
    clock = LiveClock(segs)
    P = clock.P

    D = {}
    for name, path in ((n1, f1), (n2, f2)):
        d = load_peaks(path)
        if a.sgveto:
            if 'sg_chisq' not in d:
                sys.exit(f'--sgveto: {path} has no sg_chisq')
            # PyCBC newsnr_sgveto: rank /= sqrt(sg_chisq / 4) where sg_chisq > 4
            d['rank'] = d['rank'] / np.sqrt(np.maximum(1.0, d['sg_chisq'] / 4.0))
        m = clock.inside(d['t']) & (d['rank'] >= a.min_rank)
        if a.templates:
            m &= np.isin(d['tid'], np.loadtxt(a.templates, dtype=np.int64, ndmin=1))
        if a.keep_loudest:
            w = np.nonzero(m)[0]
            m = np.zeros_like(m); m[w[keep_loudest(d['t'][w], d['rank'][w], a.keep_loudest[0], int(a.keep_loudest[1]))]] = True
        for key in ('tid', 't', 'snr', 'rank') + (('chisq', 'dof') if d['has_chisq'] else ()) + (('phase', 'sigmasq') if 'phase' in d else ()) + (('sg_chisq',) if 'sg_chisq' in d else ()):
            d[key] = d[key][m]
        d['x'] = clock.to_live(d['t'])
        if a.pseudo_shift and name == n2:
            # noise-only pseudo-experiment: move this detector's triggers by a fixed offset on the live clock,
            # far outside the slides, so the zero lag holds no real coincidences; everything else is unchanged
            d['x'] = np.mod(d['x'] + a.pseudo_shift, P)
            d['t'] = clock.to_gps(d['x'])
            o = np.argsort(d['t'], kind='stable')   # keep the time order the loaders give
            for key in [k for k, v in d.items() if isinstance(v, np.ndarray) and len(v) == len(o)]:
                d[key] = d[key][o]
        d['n_in'] = int(m.sum())
        D[name] = d
        if not d['has_chisq']:
            print(f'WARNING: {name} has no chi-squared; ranking on plain SNR (background will be much worse)')
    H, L = D[n1], D[n2]

    K = a.max_slides if a.max_slides is not None else int((P / 2 - a.window) / a.slide_step) - 1
    if a.background_from:
        K = 0          # zero lag only; the background comes from the search run
    stat_fn, stat_name = make_stat(a, H, L, P)
    if a.save_fits and 'fit' in H:
        np.savez(a.save_fits, H_lra=H['fit'][0], H_alpha=H['fit'][1], L_lra=L['fit'][0], L_alpha=L['fit'][1])
    if hasattr(stat_fn, 'bound') and np.isfinite(a.min_stat):
        masks = prune_singles(H, L, *stat_fn.bound, a.min_stat, K * a.slide_step + a.window, P)
        for d, m in zip((H, L), masks):
            for key in [k for k, v in d.items() if k != 'dur' and isinstance(v, np.ndarray) and len(v) == len(m)]:
                d[key] = d[key][m]
    t_pairs = time.perf_counter()
    hi, li, k, stat = find_pairs(H, L, P, a.slide_step, K, a.window, stat_fn, a.min_stat)
    t_pairs = time.perf_counter() - t_pairs
    keep = cluster(k, H['x'][hi], stat, a.cluster_window)
    hi, li, k, stat = hi[keep], li[keep], k[keep], stat[keep]
    fg = k == 0; bg = ~fg
    n_slides = 2 * K
    T_bg = n_slides * P
    bstat = np.sort(stat[bg])
    if a.background_from:
        with h5py.File(a.background_from, 'r') as f:
            bstat = np.sort(f['background/stat'][:]); T_bg = float(f.attrs['background_time_yr']) * YEAR
            n_slides = int(f.attrs['n_slides'])

    # inclusive FAR
    def n_louder(s, arr):
        return len(arr) - np.searchsorted(arr, s, 'left')
    fstat = stat[fg]; fo = np.argsort(-fstat)
    fh, fl, fstat = hi[fg][fo], li[fg][fo], fstat[fo]
    far_incl = n_louder(fstat, bstat) / T_bg

    # exclusive FAR, for the loudest --n-excl candidates: for candidate j, drop background coincs that
    # reuse a single trigger (same detector, within excl-window, any template) of any foreground
    # candidate ranked at or above j.
    bh, bl, bs = hi[bg], li[bg], stat[bg]
    nx = min(a.n_excl, len(fstat))
    bad_rank = np.full(len(bs), np.iinfo(np.int64).max)   # best rank of a foreground candidate vetoing each bg coinc
    for det_bg, det_fg, X in ((bh, fh, H['x']), (bl, fl, L['x'])):
        bt = X[det_bg]; ob = np.argsort(bt); bts = bt[ob]
        for j in range(nx):
            ft = X[det_fg[j]]
            sl = ob[np.searchsorted(bts, ft - a.excl_window, 'left'):np.searchsorted(bts, ft + a.excl_window, 'right')]
            bad_rank[sl] = np.minimum(bad_rank[sl], j)
    far_excl = far_incl.copy()
    v = bad_rank < nx
    vr, vs = bad_rank[v], bs[v]
    for j in range(nx):
        far_excl[j] = far_incl[j] - np.count_nonzero((vr <= j) & (vs >= fstat[j])) / T_bg
    far_floor = 1.0 / T_bg

    bank = None
    if a.bank:
        with h5py.File(a.bank, 'r') as f:
            bank = {c: f[c][:] for c in ('mass1', 'mass2', 'spin1z', 'spin2z') if c in f}

    cands = []
    for j in range(len(fstat)):
        h, l = fh[j], fl[j]
        c = dict(rank=j + 1, stat=float(fstat[j]),
                 far_incl_per_yr=float(far_incl[j] * YEAR), far_excl_per_yr=float(far_excl[j] * YEAR),
                 far_is_upper_limit=bool(far_excl[j] == 0),
                 template_id=int(H['tid'][h]),
                 **{f'{n1}_time': float(H['t'][h]), f'{n2}_time': float(L['t'][l]),
                    f'{n1}_snr': float(H['snr'][h]), f'{n2}_snr': float(L['snr'][l]),
                    f'{n1}_newsnr': float(H['rank'][h]), f'{n2}_newsnr': float(L['rank'][l])},
                 network_snr=float(np.hypot(H['snr'][h], L['snr'][l])))
        if bank is not None:
            c.update({p: float(v[c['template_id']]) for p, v in bank.items()})
        cands.append(c)

    summary = dict(ifos=[n1, n2], live_time_s=P, n_segments=len(clock.s0), slide_step_s=a.slide_step,
                   n_slides=n_slides, background_time_yr=T_bg / YEAR,
                   far_floor_per_yr=far_floor * YEAR,
                   n_singles={n1: H['n_in'], n2: L['n_in']}, n_foreground=int(fg.sum()), n_background=int(bg.sum()),
                   ranking=stat_name, min_stat=a.min_stat, pair_time_s=t_pairs,
                   runtime_s=time.perf_counter() - t_start, top=cands[:a.top])
    out = a.out[:-4] if a.out.endswith('.hdf') else a.out
    with h5py.File(out + '.hdf', 'w') as f:
        keys = list(cands[0]) if cands else ['rank', 'stat', 'far_incl_per_yr', 'far_excl_per_yr', f'{n1}_time', f'{n2}_time']
        for key in keys:
            f[f'foreground/{key}'] = np.array([c[key] for c in cands], dtype=float if not cands else None)
        f['background/stat'] = bs; f['background/slide'] = k[bg]; f['background/template_id'] = H['tid'][bh]
        f['background/' + n1 + '_time'] = H['t'][bh]; f['background/' + n2 + '_time'] = L['t'][bl]
        for kk, v in summary.items():
            if kk != 'top':
                f.attrs[kk] = json.dumps(v) if isinstance(v, (dict, list)) else v
    json.dump(summary, open(out + '.json', 'w'), indent=1)

    print(f"live {P:.0f} s, {n_slides} slides -> background {T_bg / YEAR:.2f} yr, FAR floor {far_floor * YEAR:.3g}/yr, "
          f"{fg.sum()} foreground / {bg.sum()} background clustered coincs, {summary['runtime_s']:.1f} s")
    print(f"{'#':>3} {'stat':>6} {'FARexcl/yr':>11} {'FARincl/yr':>11} {n1 + ' time':>15} {n1 + ' snr':>7} {n2 + ' snr':>7} {'tid':>6}")
    for c in cands[:a.top]:
        lt = '<' if c['far_is_upper_limit'] else ' '
        fe = c['far_excl_per_yr'] if not c['far_is_upper_limit'] else far_floor * YEAR
        print(f"{c['rank']:>3} {c['stat']:6.2f} {lt}{fe:10.3g} {c['far_incl_per_yr']:11.3g} {c[n1 + '_time']:15.3f} "
              f"{c[n1 + '_snr']:7.2f} {c[n2 + '_snr']:7.2f} {c['template_id']:>6}")


if __name__ == '__main__':
    main()
