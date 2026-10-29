"""Cut conditioned segments into short overlap-save chunks, so a follow-up bank (eccentric) can be filtered only on the
chunks and template tiles that the base search's triggers select, with the same SNR as filtering whole segments.

The base pass holds each segment as D = stilde / psd (over-whitened, one-sided, PyCBC normalisation D = dt rfft(d)).
Its SNR series z(t) = 4 df sum_k conj(h_k) D_k e^{2 pi i k t / N} correlates the template h(tau) with the over-whitened
series d. A template is supported on tau in [-hist, fut] (inspiral before merger, ringdown after), so z on
[t, t + V) only needs d on [t - hist, t + V + fut). A chunk of Lc = hist + V + fut samples of d, transformed at
df_c = 1 / (Lc dt), times the template sampled at df_c, gives the same z on its V valid samples. Chunks step by V,
so their valid ranges tile the segment like the base pass's analyze range.

Arrays are numpy or cupy (xp); nothing here depends on PyCBC.
"""
import numpy as np


def chunk_starts(tlen, Lc, hist, fut, lo=None, hi=None):
    """Chunk starts (samples in the segment) whose valid ranges [s + hist, s + Lc - fut) tile [lo, hi) (default: every
    sample that has hist samples of history and fut of future). The last chunk is moved back to end at the segment
    end, so it may overlap the previous one; valid() trims the overlap."""
    lo = hist if lo is None else max(lo, hist)
    hi = tlen - fut if hi is None else min(hi, tlen - fut)
    V = Lc - hist - fut
    assert V > 0 and hi > lo
    s = np.arange(lo - hist, hi - hist, V)
    s[-1] = min(s[-1], tlen - Lc)
    return s


def valid(starts, Lc, hist, fut, hi=None):
    """Per chunk, the half-open sample range [a, b) in the segment that it alone reports (no overlaps, nothing at or
    after hi)."""
    a = starts + hist
    b = np.append(a[1:], starts[-1] + Lc - fut)
    if hi is not None:
        b = np.minimum(b, hi)
    return a, np.maximum(b, a)


def chunk_segment(D, IP, tlen, dt, starts, Lc, xp=np):
    """Over-whitened chunks Dc = dt rfft(d[s : s + Lc]) and 1/psd sampled at the chunk resolution, (nch, Lc//2 + 1)."""
    step = tlen // Lc
    assert step * Lc == tlen, 'chunk length must divide the segment length'
    d = xp.fft.irfft(D, n=tlen) / dt
    idx = xp.asarray(starts)[:, None] + xp.arange(Lc)[None, :]
    Dc = (dt * xp.fft.rfft(d[idx], axis=1)).astype(xp.complex64)
    IPc = xp.broadcast_to(IP[::step][: Lc // 2 + 1], Dc.shape)
    return Dc, IPc
