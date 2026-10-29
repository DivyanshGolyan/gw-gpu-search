"""Exact complex SNR numerator z and chi-squared bin power sum_j |Z_j|^2 at candidate peaks of one template tile.

In --templates interp mode the dense row H[i] is defined as imr_interp.FineInterpolator's evaluation of ~7,500 coarse
nodes that every template shares: amplitude linear in t, phase a cubic per interval. So this stage never reads H:
  - Edges from moments. With u = 1 - t, sum over interval j of |h_k|^2 IP_k is A_j^2 Muu_j + 2 A_j A_j+1 Mut_j +
    A_j+1^2 Mtt_j, where M are per-segment moments of IP. The edge search is an interval cumsum, then a binary search
    inside the target interval through within-interval prefix sums. Same edges as the float64 cumsum of |H|^2 IP up
    to summation order (+-1 bin at near-ties).
  - Regenerated binned sums. Z_j(t) = sum_{k in bin j} conj(h_k) D_k exp(2 pi i k t / N), with h_k rebuilt in
    registers (fp32 cubic on the 2 pi-reduced node phase, one __sincosf, a lerp) and the time phasor advanced by a
    recurrence that is reseeded exactly from the integer phase every R = 32 bins.

Usage (fp_filter):
    grid = Grid.from_interp(interp)                                                    # once per run
    stage = ExactStage(TemplateTile(Hc, nbins_all[ids], kmin, kmax_all[ids], Coarse(grid, amp, coef)))   # per tile
    pv = stage.evaluate(SegmentData(Dc, IPc, df, tlen), PeakRequest(rows, samples))    # per (tile, segment)
    snr = |pv.z| norm;  chisq = (nb pv.pw - |pv.z|^2) norm^2;  sg reference frequency = pv.fpeak
--templates pycbc passes coarse=None and gets today's dense path (edges_per_template, bin_power_sums) behind the same
object. The first evaluate of a coarse tile checks itself against that dense path (stage.check) and raises on
disagreement. The backend follows the tile's H: numpy arrays run the numpy reference, cupy arrays the CUDA kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np

R = 32                 # consecutive fine bins per thread: phasor reseed interval and minimum chi-squared bin width
NT = 256               # threads per block
KC = R * NT            # fine bins per k-chunk; chunk c covers [kmin + c KC, kmin + (c + 1) KC)
MAX_PEAKS = 8          # peaks per item (one row's peaks share the regenerated template)
CHECK_ROWS = 16
CHECK_PEAKS = 64
CHECK_TOL = dict(edge_bins=1, edge_frac=1e-2, snr_rel=1e-4, phase_rad=1e-4, chisq_rel=1e-3)   # snr_rel, phase_rad 1e-5 -> 1e-4 2026-10-28 (float32 rounding tripped them; FREEZE.md change log)

_TWO_PI32 = np.float32(2 * np.pi)
_INV_TWO_PI32 = np.float32(1 / (2 * np.pi))


@dataclass(frozen=True, eq=False)
class Grid:
    """Fine bins k in [kmin, kend) -> coarse interval, exactly as FineInterpolator evaluates them. One per run, shared
    by every template. node[k - kmin] = j, d = f - x_j (Hz), t = d / (x_j+1 - x_j), kb[j] = first k of interval j,
    kb[J] = kend. An interval may hold no fine bin."""
    kmin: int
    kend: int
    node: np.ndarray
    d: np.ndarray
    t: np.ndarray
    kb: np.ndarray

    @classmethod
    def from_interp(cls, interp) -> Grid:
        node = _host(interp.i).astype(np.int32)
        kb = interp.kmin + np.searchsorted(node, np.arange(interp.nx), side='left')
        return cls(int(interp.kmin), int(interp.kend), node, _host(interp.dx).astype(np.float64),
                   _host(interp.t).astype(np.float64), kb.astype(np.int64))

    @property
    def J(self) -> int:
        return len(self.kb) - 1

    @property
    def nchunk(self) -> int:
        return -(-(self.kend - self.kmin) // KC)

    @cached_property
    def start(self) -> np.ndarray:
        return self.kb[:-1] - self.kmin

    @cached_property
    def last(self) -> np.ndarray:
        """Local index of each interval's last bin (below start for an empty interval)."""
        return self.kb[1:] - self.kmin - 1

    @cached_property
    def padded(self) -> np.ndarray:
        """Local index of every bin of the chunk-padded band, natural order. Padding repeats the last bin: it lands in
        bin nb (k >= kend), which no sum keeps."""
        return np.minimum(np.arange(self.nchunk * KC), self.kend - self.kmin - 1)

    @cached_property
    def device(self) -> _GridDev:
        return _GridDev.build(self)


@dataclass(frozen=True, eq=False)
class Coarse:
    """The interp nodes a tile's H rows were generated from: imr_interp.coarse_generate's amp (n, nx) and coef
    (n, 4, nx - 1), highest order first, for the tile's rows in tile order."""
    grid: Grid
    amp: np.ndarray
    coef: np.ndarray


@dataclass(frozen=True, eq=False)
class TemplateTile:
    """One tile. H (n, flen) complex64, numpy or cupy, holds the dense rows: the self-check oracle and, with
    coarse=None (--templates pycbc), the exact path itself. nbins per row (0 = no chi-squared, one bin). Bins cover
    [kmin, kmax[row])."""
    H: object
    nbins: np.ndarray
    kmin: int
    kmax: np.ndarray
    coarse: Coarse | None = None

    def __post_init__(self):
        n = self.H.shape[0]
        object.__setattr__(self, 'nbins', np.asarray(self.nbins, np.int64))
        object.__setattr__(self, 'kmax', np.asarray(self.kmax, np.int64))
        assert self.H.ndim == 2 and self.H.dtype == np.complex64 and self.H.flags.c_contiguous
        assert self.nbins.shape == (n,) and self.kmax.shape == (n,)
        if self.coarse is not None:
            g = self.coarse.grid
            assert self.coarse.amp.shape == (n, g.J + 1) and self.coarse.coef.shape == (n, 4, g.J)
            assert g.kmin == self.kmin and np.all(self.kmax == g.kend), 'interp bins cover [kmin, kend)'


@dataclass(frozen=True, eq=False)
class SegmentData:
    """One segment on the tile's backend: D = stilde / psd (complex64, flen) and IP = 1 / psd (float32, flen), both zero
    below kmin. tlen = 2 (flen - 1) is a power of two, so k t mod tlen can use 32-bit wraparound."""
    D: object
    IP: object
    df: float
    tlen: int

    def __post_init__(self):
        flen = self.tlen // 2 + 1
        assert self.tlen & (self.tlen - 1) == 0
        assert self.D.shape == (flen,) and self.D.dtype == np.complex64
        assert self.IP.shape == (flen,) and self.IP.dtype == np.float32


@dataclass(frozen=True, eq=False)
class PeakRequest:
    """Peaks of one (tile, segment): tile row and sample index. Duplicates are allowed; results keep this order."""
    rows: np.ndarray
    samples: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'rows', np.asarray(self.rows, np.int64))
        object.__setattr__(self, 'samples', np.asarray(self.samples, np.int64))
        assert self.rows.ndim == 1 and self.rows.shape == self.samples.shape


@dataclass(frozen=True, eq=False)
class PeakValues:
    """Per requested peak, in request order. z = sum_j Z_j (complex64), pw = sum_j |Z_j|^2 (float32), fpeak = the
    sine-Gaussian reference frequency (2 e[nb-1] - e[nb-2]) df from this stage's own edges (Hz)."""
    z: np.ndarray
    pw: np.ndarray
    fpeak: np.ndarray


class ExactStage:
    """Exact z, pw and fpeak at requested peaks of one tile, segment after segment. Built once per tile."""

    def __init__(self, tile: TemplateTile):
        self.tile = tile
        self.backend = _backend_of(tile.H)
        self.ops = _BACKENDS[self.backend]
        self.state = self.ops.prepare(tile.coarse) if tile.coarse is not None else None
        self.check: dict | None = None

    def edges(self, seg: SegmentData, rows: np.ndarray) -> np.ndarray:
        """The chi-squared bin edges this stage uses for the given tile rows in this segment: (len(rows), B + 1),
        pycbc's rule as multi_filter.edges_per_template, padded with kmax beyond each row's own bin count."""
        rows_u, inv = np.unique(np.asarray(rows, np.int64), return_inverse=True)
        return self._edges_u(seg, rows_u)[inv]

    def evaluate(self, seg: SegmentData, req: PeakRequest) -> PeakValues:
        assert _backend_of(seg.D) == self.backend and _backend_of(seg.IP) == self.backend
        plan = _plan(req, self.tile.H.shape[0], seg.tlen)
        npk = len(plan.order)
        if npk == 0:
            return PeakValues(np.zeros(0, np.complex64), np.zeros(0, np.float32), np.zeros(0))
        nbe_u = self._nbe(plan.rows_u)
        E_u = self._edges_u(seg, plan.rows_u)
        if self.state is None:
            zs, pws = self.ops.dense_sums(self.tile, seg, plan, E_u, nbe_u)
        else:
            _assert_bin_width(E_u, nbe_u)
            zs, pws = self.ops.binned(self.state, seg, plan, E_u, nbe_u)
        z = np.empty(npk, np.complex64); pw = np.empty(npk, np.float32); u_req = np.empty(npk, np.int64)
        z[plan.order], pw[plan.order], u_req[plan.order] = zs, pws, plan.peak_u
        out = PeakValues(z, pw, _fpeak(E_u[u_req], self.tile.nbins[req.rows], seg.df))
        if self.state is not None and self.check is None:
            self.check = _self_check(self, seg, req, out, plan, E_u)
        return out

    def _nbe(self, rows_u):
        return np.maximum(self.tile.nbins[rows_u], 1)

    def _edges_u(self, seg, rows_u):
        nbe = self._nbe(rows_u)
        if self.state is None:
            return _dense_edges(self.ops.xp(), self.tile, seg, rows_u, nbe)
        st = self.state
        E = self.ops.edges(st, _moments(self.ops.xp(), st.gv, st.grid.kmin, st.grid.kend, seg.IP), rows_u, nbe)
        E[self.tile.nbins[rows_u] == 0, 0] = self.tile.kmin
        return E


# ---------------------------------------------------------------------------------------------------------------------
# Shared host planning and checks
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class _Plan:
    """Peaks sorted by (row, sample), cut into items of at most MAX_PEAKS peaks of one row."""
    order: np.ndarray        # sorted position -> request index
    rows_u: np.ndarray       # distinct requested rows, ascending
    peak_u: np.ndarray       # per sorted peak: index into rows_u
    peak_t: np.ndarray       # per sorted peak: sample
    item_peak0: np.ndarray   # per item: first sorted peak
    item_np: np.ndarray      # per item: peak count, 1..MAX_PEAKS


def _plan(req: PeakRequest, n: int, tlen: int) -> _Plan:
    rows, samples = req.rows, req.samples
    assert len(rows) == 0 or (0 <= rows.min() and rows.max() < n and 0 <= samples.min() and samples.max() < tlen)
    order = np.lexsort((samples, rows))
    rs = rows[order]
    rows_u, peak_u = np.unique(rs, return_inverse=True)
    run_end = np.r_[np.flatnonzero(np.diff(rs)) + 1, len(rs)]
    run_start = np.r_[0, run_end[:-1]] if len(rs) else np.zeros(0, np.int64)
    item_peak0 = np.concatenate([np.arange(a, b, MAX_PEAKS) for a, b in zip(run_start, run_end)] or [np.zeros(0, int)])
    item_end = run_end[np.searchsorted(run_end, item_peak0, side='right')] if len(rs) else item_peak0
    item_np = np.minimum(item_end - item_peak0, MAX_PEAKS)
    return _Plan(order, rows_u, peak_u.astype(np.int64), samples[order], item_peak0.astype(np.int64),
                 item_np.astype(np.int64))


def _fpeak(E, nb, df):
    r = np.arange(len(E))
    return (2 * E[r, np.maximum(nb - 1, 0)] - E[r, np.maximum(nb - 2, 0)]) * df


def _assert_bin_width(E_u, nbe_u):
    w = np.diff(E_u, axis=1)
    inside = np.arange(w.shape[1])[None, :] < nbe_u[:, None]
    assert (w[inside] >= R).all(), f'a chi-squared bin is narrower than {R} bins (min {w[inside].min()})'


def _moments(xp, gv, kmin, kend, IP):
    """Per-segment moments of IP in the (1-t)^2, t(1-t), t^2 basis. P (K, 3): within-interval inclusive prefix at
    each bin. M (J, 3): interval totals, bit-identical to P at each interval's last bin (the edge search relies on
    that). The prefix is a global float64 cumsum minus its value at the interval start; its absolute error is about
    1e-16 of sum IP, far below one bin's weight."""
    w = IP[kmin:kend].astype(xp.float64)
    u = 1.0 - gv.t
    G = xp.cumsum(xp.stack([u * u * w, u * gv.t * w, gv.t * gv.t * w], axis=1), axis=0)
    Gb = xp.concatenate([xp.zeros((1, 3)), G], axis=0)
    P = xp.ascontiguousarray(G - Gb[gv.start[gv.node]])
    M = xp.ascontiguousarray(xp.where((gv.last >= gv.start)[:, None], P[xp.maximum(gv.last, 0)], 0.0))
    return P, M


def _dense_edges(xp, tile, seg, rows_u, nbe):
    """Today's edges: multi_filter.edges_per_template, the float64 cumsum of |H|^2 IP over each row."""
    from multi_filter import edges_per_template
    E = np.asarray(edges_per_template(xp, rows_u, nbe, tile.kmin, tile.kmax[rows_u], tile.H, seg.IP), np.int64)
    E[tile.nbins[rows_u] == 0, 0] = tile.kmin
    return E


def _self_check(stage, seg, req, out, plan, E_u):
    """Up to CHECK_PEAKS peaks of up to CHECK_ROWS rows against today's dense code: edges against edges_per_template,
    and z, pw against the dense sums over the same (this stage's) edges, so edge rounding and regeneration error are
    measured apart. Raises if a tolerance in CHECK_TOL fails."""
    pick = plan.rows_u[np.unique(np.linspace(0, len(plan.rows_u) - 1, CHECK_ROWS).astype(int))]
    rows = req.rows
    sel = np.flatnonzero(np.isin(rows, pick))[:CHECK_PEAKS]
    sub = _plan(PeakRequest(rows[sel], req.samples[sel]), stage.tile.H.shape[0], seg.tlen)
    nbe = stage._nbe(sub.rows_u)
    E_ref = _dense_edges(stage.ops.xp(), stage.tile, seg, sub.rows_u, nbe)
    E_sub = E_u[np.searchsorted(plan.rows_u, sub.rows_u)][:, :E_ref.shape[1]]
    zs, pws = stage.ops.dense_sums(stage.tile, seg, sub, E_sub, nbe)
    z_ref = np.empty(len(sel), np.complex128); pw_ref = np.empty(len(sel)); z_ref[sub.order] = zs; pw_ref[sub.order] = pws
    rep = dict(n_peaks=int(len(sel)), n_rows=int(len(sub.rows_u)), **_edge_stats(E_sub, E_ref, nbe),
               **compare_values(out.z[sel], out.pw[sel], z_ref, pw_ref, stage.tile.nbins[rows[sel]]))
    t = CHECK_TOL
    ok = (rep['edge_max_diff'] <= t['edge_bins'] and rep['edge_frac_diff'] <= t['edge_frac']
          and rep['snr_rel_max'] <= t['snr_rel'] and rep['phase_max'] <= t['phase_rad']
          and rep['chisq_rel_max'] <= t['chisq_rel'])
    assert ok, f'exact stage self-check failed: {rep}'
    return rep


def _edge_stats(E, E_ref, nbe):
    inside = np.arange(E.shape[1])[None, :] <= nbe[:, None]
    d = np.abs(E - E_ref)[inside]
    return dict(edge_n=int(d.size), edge_n_diff=int((d > 0).sum()), edge_frac_diff=float((d > 0).mean()),
                edge_max_diff=int(d.max()))


def compare_values(z, pw, z_ref, pw_ref, nbins) -> dict:
    """Relative |z| error, phase error (rad) and relative chi-squared error, max and p99. Chi-squared is
    (nb pw - |z|^2) norm^2, so its relative error needs no norm; rows without chi-squared bins are left out."""
    z = np.asarray(z, np.complex128); z_ref = np.asarray(z_ref, np.complex128)
    snr = np.abs(np.abs(z) - np.abs(z_ref)) / np.abs(z_ref)
    ph = np.abs(np.angle(z * np.conj(z_ref)))
    nb = np.asarray(nbins); m = nb > 0
    q = nb[m] * np.asarray(pw, np.float64)[m] - np.abs(z[m]) ** 2
    q_ref = nb[m] * np.asarray(pw_ref, np.float64)[m] - np.abs(z_ref[m]) ** 2
    chi = np.abs(q - q_ref) / np.abs(q_ref) if m.any() else np.zeros(1)
    return dict(snr_rel_max=float(snr.max()), snr_rel_p99=float(np.percentile(snr, 99)),
                phase_max=float(ph.max()), phase_p99=float(np.percentile(ph, 99)),
                chisq_rel_max=float(chi.max()), chisq_rel_p99=float(np.percentile(chi, 99)))


def _coarse_tables(coarse: Coarse):
    """Host tables the binned sums read per (row, interval): c4 = (c0, c1, c2, c3 reduced to [-pi, pi) in float64)
    and a2 = (A_j, A_j+1), both float32; plus float64 amplitudes for the edges."""
    coef, amp = coarse.coef, coarse.amp
    c4 = np.empty((coef.shape[0], coef.shape[2], 4), np.float32)
    c4[..., :3] = coef[:, :3, :].transpose(0, 2, 1)
    c4[..., 3] = np.mod(coef[:, 3, :] + np.pi, 2 * np.pi) - np.pi
    a2 = np.ascontiguousarray(np.stack([amp[:, :-1], amp[:, 1:]], -1).astype(np.float32))
    return np.ascontiguousarray(amp, np.float64), c4, a2


@dataclass(frozen=True, eq=False)
class _TileState:
    grid: Grid
    gv: object               # grid arrays on the backend (Grid itself on the host)
    amp64: object            # (n, nx) float64
    c4: object               # (n, J, 4) float32
    a2: object               # (n, J, 2) float32


# ---------------------------------------------------------------------------------------------------------------------
# numpy reference: the kernels' arithmetic, step for step (fp32 where the kernels are fp32)
# ---------------------------------------------------------------------------------------------------------------------

def _prepare_numpy(coarse):
    return _TileState(coarse.grid, coarse.grid, *_coarse_tables(coarse))


def _quad(a, b, p0, p1, p2):
    """Interval weight from node amplitudes a, b and moments p. The kernel's quad() keeps this operation order."""
    return a * a * p0 + 2 * a * b * p1 + b * b * p2


def _edges_numpy(st, mom, rows_u, nbe):
    """_edges_from_moments, reference. Interval weights W from the moments, their cumsum C, the interval J holding each
    target (first C > target), then a binary search in J for the first k with C[J-1] + quad(P[k]) > target. The last
    bin of J gives exactly C[J], so the search always lands inside J."""
    P, M = mom
    g = st.grid
    A = st.amp64[rows_u]
    a0, a1 = A[:, :-1], A[:, 1:]
    C = np.cumsum(_quad(a0, a1, M[:, 0], M[:, 1], M[:, 2]), axis=1)
    B = int(nbe.max())
    E = np.full((len(rows_u), B + 1), g.kend, np.int64)
    for r in range(len(rows_u)):
        T = np.arange(nbe[r]) / nbe[r] * C[r, -1]
        J = np.minimum(np.searchsorted(C[r], T, side='right'), g.J - 1)
        base = np.where(J > 0, C[r, np.maximum(J - 1, 0)], 0.0)
        a, b = a0[r, J], a1[r, J]
        lo, hi = g.start[J].copy(), g.last[J].copy()
        while (lo < hi).any():
            mid = (lo + hi) // 2
            above = base + _quad(a, b, P[mid, 0], P[mid, 1], P[mid, 2]) > T
            act = lo < hi
            hi = np.where(act & above, mid, hi)
            lo = np.where(act & ~above, mid + 1, lo)
        E[r, :nbe[r]] = np.clip(lo + g.kmin, g.kmin, g.kend)
    return E


def _regen_conj_h_d(c4r, a2r, node, d32, t32, Dp):
    """conj(h_k) D_k with h_k rebuilt in fp32 as the kernel does."""
    cf = c4r[node]
    x = ((cf[:, 0] * d32 + cf[:, 1]) * d32 + cf[:, 2]) * d32 + cf[:, 3]
    x = x - _TWO_PI32 * np.rint(x * _INV_TWO_PI32)
    am = a2r[node]
    a = (am[:, 1] - am[:, 0]) * t32 + am[:, 0]
    hr, hi = a * np.cos(x), a * np.sin(x)
    v = np.empty(len(x), np.complex64)
    v.real = hr * Dp.real + hi * Dp.imag
    v.imag = hr * Dp.imag - hi * Dp.real
    return v


def _binned_numpy(st, seg, plan, E_u, nbe_u):
    """The regenerated binned sums, reference for regen_bins + finalize. Thread = R consecutive bins with an fp32
    partial per (thread, bin) and a phasor reseeded at the thread's first bin; block = one KC chunk summing its
    threads per bin; finalize = fp32 sum of a bin's chunk sums, then z and pw in float64. Returns sorted-peak order."""
    g = st.grid
    loc = g.padded
    node, d32, t32 = g.node[loc], g.d[loc].astype(np.float32), g.t[loc].astype(np.float32)
    Dp = seg.D[g.kmin + loc]
    k = g.kmin + np.arange(len(loc), dtype=np.int64)
    run = np.arange(len(loc)) // R
    ka = k[::R]
    N = seg.tlen
    z = np.empty(len(plan.peak_t), np.complex64); pw = np.empty(len(plan.peak_t), np.float32)
    for p0, npi in zip(plan.item_peak0, plan.item_np):
        u = plan.peak_u[p0]; row = plan.rows_u[u]; nb = int(nbe_u[u])
        v = _regen_conj_h_d(st.c4[row], st.a2[row], node, d32, t32, Dp)
        b = np.searchsorted(E_u[u, :nb + 1], k, side='right') - 1
        s1 = np.r_[0, np.flatnonzero((np.diff(run) != 0) | (np.diff(b) != 0)) + 1]
        c1, b1 = run[s1] // NT, b[s1]
        s2 = np.r_[0, np.flatnonzero((np.diff(c1) != 0) | (np.diff(b1) != 0)) + 1]
        keep = (b1[s2] >= 0) & (b1[s2] < nb)
        bk = b1[s2][keep]
        s3 = np.r_[0, np.flatnonzero(np.diff(bk)) + 1]
        assert len(s3) == nb
        for l in range(p0, p0 + npi):
            t = int(plan.peak_t[l])
            w = np.empty((R, len(ka)), np.complex64)
            w[0] = np.exp(2j * np.pi * ((ka * t) % N) / N)
            rot = np.complex64(np.exp(2j * np.pi * (t % N) / N))
            for s in range(1, R):
                w[s] = w[s - 1] * rot
            part = np.add.reduceat(v * w.T.ravel(), s1)
            Zj = np.add.reduceat(np.add.reduceat(part, s2)[keep], s3).astype(np.complex128)
            z[l] = Zj.sum(); pw[l] = (Zj.real ** 2 + Zj.imag ** 2).sum()
    return z, pw


def _dense_sums_numpy(tile, seg, plan, E_u, nbe_u):
    """The dense oracle on the host: float64 sums of conj(H) D exp(2 pi i ((k t) mod N) / N) per bin."""
    N = seg.tlen
    z = np.empty(len(plan.peak_t), np.complex128); pw = np.empty(len(plan.peak_t))
    D = seg.D
    for u, row in enumerate(plan.rows_u):
        e = E_u[u, :nbe_u[u] + 1]
        k = np.arange(e[0], e[-1], dtype=np.int64)
        q = np.conj(tile.H[row, e[0]:e[-1]].astype(np.complex128)) * D[e[0]:e[-1]].astype(np.complex128)
        b = np.searchsorted(e, k, side='right') - 1
        for l in np.flatnonzero(plan.peak_u == u):
            x = q * np.exp(2j * np.pi * ((k * int(plan.peak_t[l])) % N) / N)
            Zj = np.bincount(b, x.real, nbe_u[u]) + 1j * np.bincount(b, x.imag, nbe_u[u])
            z[l] = Zj.sum(); pw[l] = (np.abs(Zj) ** 2).sum()
    return z, pw


# ---------------------------------------------------------------------------------------------------------------------
# CUDA (cupy RawModule). Not run on this host; matches the numpy reference above.
# ---------------------------------------------------------------------------------------------------------------------

_CUDA_SRC = r'''
#define NT %(NT)d
#define R %(R)d
#define KC (NT * R)
#define TWO_PI 6.283185307179586f
#define INV_TWO_PI 0.15915494309189535f

// a^2 p0 + 2 a b p1 + b^2 p2 in numpy's operation order with no FMA contraction, so the interval weight and the
// in-interval prefix at the interval's last bin are bit-identical
__device__ __forceinline__ double quad(const double a, const double b, const double* p)
{
    return __dadd_rn(__dadd_rn(__dmul_rn(__dmul_rn(a, a), p[0]), __dmul_rn(__dmul_rn(__dmul_rn(2.0, a), b), p[1])),
                     __dmul_rn(__dmul_rn(b, b), p[2]));
}

// bin holding k: (number of edges e[0..nb] <= k) - 1, so -1 below e[0] and nb at or above e[nb]
__device__ __forceinline__ int bin_of(const int* e, const int nb, const int k)
{
    int lo = 0, hi = nb + 1;
    while (lo < hi) { const int m = (lo + hi) >> 1; if (e[m] > k) hi = m; else lo = m + 1; }
    return lo - 1;
}

// One block per distinct row. Each thread sums the weights of a contiguous run of intervals, thread 0 scans the run
// sums serially, then each thread solves edges j = tid, tid + NT, ... : the run, the interval and the bin holding
// target j / nb * total. The walk repeats the run sum's association, so the last interval of the run always passes.
extern "C" __global__ void edges_moments(const double* __restrict__ amp, const int nx, const double* __restrict__ M,
        const double* __restrict__ P, const int* __restrict__ kb, const int* __restrict__ rows,
        const int* __restrict__ nbs, const int nedge, const int kmin, const int kend, int* __restrict__ out,
        double* __restrict__ tot_out)
{
    const int u = blockIdx.x, tid = threadIdx.x, J = nx - 1;
    const double* A = amp + (size_t)rows[u] * nx;
    const int nb = nbs[u];
    const int per = (J + NT - 1) / NT, j0 = min(tid * per, J), j1 = min(j0 + per, J);
    double s = 0.0;
    for (int j = j0; j < j1; j++) s = __dadd_rn(s, quad(A[j], A[j + 1], M + 3 * j));
    __shared__ double pre[NT + 1];
    pre[tid + 1] = s;
    if (tid == 0) pre[0] = 0.0;
    __syncthreads();
    if (tid == 0) for (int i = 1; i <= NT; i++) pre[i] = __dadd_rn(pre[i], pre[i - 1]);
    __syncthreads();
    const double tot = pre[NT];
    if (tid == 0) tot_out[u] = tot;
    int* e = out + (size_t)u * nedge;
    for (int i = tid; i < nedge; i += NT) {
        if (i >= nb) { e[i] = kend; continue; }
        const double T = ((double)i / (double)nb) * tot;
        int lo = 1, hi = NT;
        while (lo < hi) { const int m = (lo + hi) >> 1; if (pre[m] > T) hi = m; else lo = m + 1; }
        const int q = lo - 1, qa = min(q * per, J), qb = min(qa + per, J);
        double r = 0.0;
        int j = max(qb - 1, 0);
        for (int jj = qa; jj < qb; jj++) {
            const double w = quad(A[jj], A[jj + 1], M + 3 * jj);
            if (__dadd_rn(pre[q], __dadd_rn(r, w)) > T) { j = jj; break; }
            r = __dadd_rn(r, w);
        }
        const double a = A[j], b = A[j + 1];
        int klo = kb[j] - kmin, khi = kb[j + 1] - kmin - 1;
        while (klo < khi) {
            const int m = (klo + khi) >> 1;
            if (__dadd_rn(pre[q], __dadd_rn(r, quad(a, b, P + 3 * (size_t)m))) > T) khi = m; else klo = m + 1;
        }
        e[i] = min(max(klo + kmin, kmin), kend);
    }
}

// grid (item groups, k-chunks): blockIdx.x varies fastest, so co-resident blocks share one chunk of D in L2.
// Thread tid owns bins ka .. ka + R - 1 (ka = chunk start + R tid). Inputs indexed by o = c KC + s NT + tid are the
// run-major (transposed) layout of the band, so step s of all threads is one coalesced load. For each of the block's
// G items it regenerates conj(h_k) D_k once and applies it to the item's P peak phasors. Bins are >= R wide, so a
// thread crosses at most one edge per item: hd holds the part before it (head), acc the part after (tail).
template <int G, int P>
__global__ void __launch_bounds__(NT, 2) regen_bins(const float2* __restrict__ Dt, const int* __restrict__ node_t,
        const float* __restrict__ d_t, const float* __restrict__ t_t, const float4* __restrict__ coef,
        const float2* __restrict__ amp2, const int J, const int kmin, const int* __restrict__ grp_item,
        const int* __restrict__ item_row, const int* __restrict__ item_u, const int* __restrict__ item_peak0,
        const int* __restrict__ item_np, const int* __restrict__ nbs, const int* __restrict__ edges, const int nedge,
        const unsigned int* __restrict__ peak_t, const unsigned int nmask, const float inv_half_n, const int NS,
        float2* __restrict__ slots)
{
    const int c = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kc0 = kmin + c * KC, ka = kc0 + R * tid;
    const int* gi = grp_item + (size_t)blockIdx.x * G;
    int row[G], cut[G], bh[G];
    float2 acc[G][P], hd[G][P], ph[G][P], rot[G][P];
#pragma unroll
    for (int g = 0; g < G; g++) {
        const int it = gi[g];
        int np_ = 0, pk0 = 0;
        row[g] = -1; cut[g] = R; bh[g] = -1;
        if (it >= 0) {
            const int u = item_u[it], nb = nbs[u];
            const int* e = edges + (size_t)u * nedge;
            row[g] = item_row[it]; np_ = item_np[it]; pk0 = item_peak0[it];
            bh[g] = bin_of(e, nb, ka);
            if (bh[g] < nb) cut[g] = min(e[bh[g] + 1] - ka, R);
        }
#pragma unroll
        for (int p = 0; p < P; p++) {
            acc[g][p] = hd[g][p] = ph[g][p] = rot[g][p] = make_float2(0.f, 0.f);
            if (p < np_) {
                const unsigned int t = peak_t[pk0 + p];
                float sn, cs;
                sincospif((float)(((unsigned int)ka * t) & nmask) * inv_half_n, &sn, &cs);   // exact seed
                ph[g][p] = make_float2(cs, sn);
                sincospif((float)(t & nmask) * inv_half_n, &sn, &cs);
                rot[g][p] = make_float2(cs, sn);
            }
        }
    }
    for (int s = 0; s < R; s++) {
        const int o = c * KC + s * NT + tid;
        const float2 dk = Dt[o];
        const int j = node_t[o];
        const float dd = d_t[o], tt = t_t[o];
#pragma unroll
        for (int g = 0; g < G; g++) {
            if (row[g] < 0) continue;
            if (s == cut[g]) {
#pragma unroll
                for (int p = 0; p < P; p++) { hd[g][p] = acc[g][p]; acc[g][p] = make_float2(0.f, 0.f); }
            }
            const float4 cf = coef[(size_t)row[g] * J + j];
            const float2 am = amp2[(size_t)row[g] * J + j];
            float x = fmaf(fmaf(fmaf(cf.x, dd, cf.y), dd, cf.z), dd, cf.w);
            x = fmaf(-TWO_PI, rintf(x * INV_TWO_PI), x);
            const float a = fmaf(am.y - am.x, tt, am.x);
            float sn, cs;
            __sincosf(x, &sn, &cs);
            const float hr = a * cs, hi = a * sn;
            const float vx = fmaf(hr, dk.x, hi * dk.y), vy = fmaf(hr, dk.y, -hi * dk.x);   // conj(h) D
#pragma unroll
            for (int p = 0; p < P; p++) {
                const float2 w = ph[g][p], rr = rot[g][p];
                acc[g][p].x = fmaf(vx, w.x, fmaf(-vy, w.y, acc[g][p].x));
                acc[g][p].y = fmaf(vx, w.y, fmaf(vy, w.x, acc[g][p].y));
                ph[g][p] = make_float2(fmaf(w.x, rr.x, -w.y * rr.y), fmaf(w.x, rr.y, w.y * rr.x));
            }
        }
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
        if (cut[g] >= R) {
#pragma unroll
            for (int p = 0; p < P; p++) { hd[g][p] = acc[g][p]; acc[g][p] = make_float2(0.f, 0.f); }
        }
    }
    __shared__ float2 red[NT / 32][P];
#pragma unroll
    for (int g = 0; g < G; g++) {
        const int it = gi[g];
        if (it < 0) continue;
        const int u = item_u[it], nb = nbs[u], np_ = item_np[it], pk0 = item_peak0[it];
        const int* e = edges + (size_t)u * nedge;
        const int b0 = max(bin_of(e, nb, kc0), 0), b1 = min(bin_of(e, nb, kc0 + KC - 1), nb - 1);
        for (int b = b0; b <= b1; b++) {   // uniform over the block: deterministic tree, then warps in order
#pragma unroll
            for (int p = 0; p < P; p++) {
                float x = 0.f, y = 0.f;
                if (bh[g] == b) { x = hd[g][p].x; y = hd[g][p].y; }
                else if (bh[g] + 1 == b) { x = acc[g][p].x; y = acc[g][p].y; }
                for (int off = 16; off > 0; off >>= 1) {
                    x += __shfl_down_sync(0xffffffffu, x, off);
                    y += __shfl_down_sync(0xffffffffu, y, off);
                }
                if (lane == 0) red[warp][p] = make_float2(x, y);
            }
            __syncthreads();
            if (tid < np_) {
                float x = 0.f, y = 0.f;
                for (int w = 0; w < NT / 32; w++) { x += red[w][tid].x; y += red[w][tid].y; }
                slots[(size_t)(pk0 + tid) * NS + c + b] = make_float2(x, y);
            }
            __syncthreads();
        }
    }
}

// One thread per sorted peak. Slot c + j holds chunk c's sum for bin j, so bin j is slots c(e_j) + j ..
// c(e_j+1 - 1) + j, summed in chunk order; z and pw accumulate in float64.
extern "C" __global__ void finalize(const float2* __restrict__ slots, const int NS, const int npk,
        const int* __restrict__ peak_u, const int* __restrict__ nbs, const int* __restrict__ edges, const int nedge,
        const int kmin, float2* __restrict__ z, float* __restrict__ pw)
{
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= npk) return;
    const int u = peak_u[l], nb = nbs[u];
    const int* e = edges + (size_t)u * nedge;
    const float2* sl = slots + (size_t)l * NS;
    double zx = 0.0, zy = 0.0, p = 0.0;
    for (int j = 0; j < nb; j++) {
        const int c0 = (e[j] - kmin) / KC, c1 = (e[j + 1] - 1 - kmin) / KC;
        float x = 0.f, y = 0.f;
        for (int c = c0; c <= c1; c++) { x += sl[c + j].x; y += sl[c + j].y; }
        zx += x; zy += y; p += (double)x * x + (double)y * y;
    }
    z[l] = make_float2((float)zx, (float)zy);
    pw[l] = (float)p;
}
''' % dict(NT=NT, R=R)

_REGEN_NAMES = {P: f'regen_bins<{MAX_PEAKS // P}, {P}>' for P in (1, 2, 4, 8)}


@lru_cache(maxsize=None)
def _cuda_module():
    import cupy as cp
    return cp.RawModule(code=_CUDA_SRC, options=('--std=c++14',), name_expressions=list(_REGEN_NAMES.values()))


@dataclass(frozen=True, eq=False)
class _GridDev:
    """The grid on the device: natural order for the moments, run-major (o = c KC + s NT + i holds local bin
    c KC + R i + s) for the binned sums, and perm, which gathers a segment's D into that layout."""
    node: object
    t: object
    start: object
    last: object
    kb32: object
    perm: object
    node_t: object
    d_t: object
    t_t: object

    @classmethod
    def build(cls, g: Grid) -> _GridDev:
        import cupy as cp
        o = np.arange(g.nchunk * KC)
        rem = o % KC
        loc = g.padded[(o // KC) * KC + R * (rem % NT) + rem // NT]
        return cls(cp.asarray(g.node), cp.asarray(g.t), cp.asarray(g.start), cp.asarray(g.last),
                   cp.asarray(g.kb.astype(np.int32)), cp.asarray((g.kmin + loc).astype(np.int32)),
                   cp.asarray(g.node[loc]), cp.asarray(g.d[loc].astype(np.float32)),
                   cp.asarray(g.t[loc].astype(np.float32)))


def _prepare_cuda(coarse):
    import cupy as cp
    amp64, c4, a2 = _coarse_tables(coarse)
    return _TileState(coarse.grid, coarse.grid.device, cp.asarray(amp64), cp.asarray(c4), cp.asarray(a2))


def _edges_cuda(st, mom, rows_u, nbe):
    import cupy as cp
    P, M = mom
    g = st.grid
    B = int(nbe.max())
    out = cp.empty((len(rows_u), B + 1), cp.int32); tot = cp.empty(len(rows_u), cp.float64)
    _cuda_module().get_function('edges_moments')(
        (len(rows_u),), (NT,),
        (st.amp64, np.int32(g.J + 1), M, P, st.gv.kb32, cp.asarray(rows_u, cp.int32), cp.asarray(nbe, cp.int32),
         np.int32(B + 1), np.int32(g.kmin), np.int32(g.kend), out, tot))
    return out.get().astype(np.int64)


def _binned_cuda(st, seg, plan, E_u, nbe_u):
    import cupy as cp
    g = st.grid
    mod = _cuda_module()
    npk = len(plan.peak_t); nedge = E_u.shape[1]; NS = g.nchunk + nedge
    i32 = lambda a: cp.asarray(np.asarray(a, np.int32))
    E32, nb32, peak_u = i32(E_u), i32(nbe_u), i32(plan.peak_u)
    item_u = plan.peak_u[plan.item_peak0]
    it_row, it_u, it_p0, it_np = i32(plan.rows_u[item_u]), i32(item_u), i32(plan.item_peak0), i32(plan.item_np)
    peak_t = cp.asarray(plan.peak_t.astype(np.uint32))
    Dt = seg.D[st.gv.perm]
    slots = cp.zeros((npk, NS), cp.complex64)
    pclass = 1 << np.ceil(np.log2(plan.item_np)).astype(np.int64)
    for P, name in _REGEN_NAMES.items():
        its = np.flatnonzero(pclass == P)
        if not len(its):
            continue
        G = MAX_PEAKS // P
        grp = np.full(-(-len(its) // G) * G, -1, np.int32); grp[:len(its)] = its
        mod.get_function(name)(
            (len(grp) // G, g.nchunk), (NT,),
            (Dt, st.gv.node_t, st.gv.d_t, st.gv.t_t, st.c4, st.a2, np.int32(g.J), np.int32(g.kmin), cp.asarray(grp),
             it_row, it_u, it_p0, it_np, nb32, E32, np.int32(nedge), peak_t, np.uint32(seg.tlen - 1),
             np.float32(2.0 / seg.tlen), np.int32(NS), slots))
    z = cp.empty(npk, cp.complex64); pw = cp.empty(npk, cp.float32)
    mod.get_function('finalize')((-(-npk // 128),), (128,), (slots, np.int32(NS), np.int32(npk), peak_u, nb32, E32,
                                                            np.int32(nedge), np.int32(g.kmin), z, pw))
    return z.get(), pw.get()


def _dense_sums_cuda(tile, seg, plan, E_u, nbe_u):
    """Today's exact kernel on the dense rows: chisq_kernel.bin_power_sums, fused mode."""
    from chisq_kernel import bin_power_sums
    pw, z = bin_power_sums(tile.H, plan.rows_u[plan.peak_u], plan.peak_t, E_u[plan.peak_u], want_z=True, d=seg.D,
                           n_time=seg.tlen)
    return z, pw


# ---------------------------------------------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Ops:
    xp: object
    prepare: object
    edges: object
    binned: object
    dense_sums: object


def _cupy():
    import cupy
    return cupy


_BACKENDS = {
    'numpy': _Ops(lambda: np, _prepare_numpy, _edges_numpy, _binned_numpy, _dense_sums_numpy),
    'cuda': _Ops(_cupy, _prepare_cuda, _edges_cuda, _binned_cuda, _dense_sums_cuda),
}


def _backend_of(a) -> str:
    return 'numpy' if isinstance(a, np.ndarray) else 'cuda'


def _host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)
