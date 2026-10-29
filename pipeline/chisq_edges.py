"""Equal-power chi-squared bin edges per template (pycbc's non-SPA rule, as multi_filter.edges_per_template), computed
without a float64 cumulative sum over every frequency bin.

|h|^2/psd is summed in float64 over blocks of BL bins (one GPU pass over h), a float64 prefix over the blocks finds the
block that holds each edge, and a float64 prefix inside only that block finds the bin. Same edges as the full cumulative
sum up to rounding order. Works with numpy (tests) or cupy."""
import numpy as np

BL = 256

_src = r'''
extern "C" __global__ void blocksum(const float2* __restrict__ h, const float* __restrict__ ips, const int F,
                                    const int nblk, const int nw, double* __restrict__ S)
{
    // one warp per (row, block of BL bins); grid.x = rows * nblk / (blockDim.x / 32)
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int row = warp / nblk, b = warp % nblk;
    if (warp >= nw) return;
    double acc = 0.0;
    for (int i = lane; i < BL_; i += 32) {
        const int k = b * BL_ + i;
        if (k < F) { const float2 v = h[(size_t)row * F + k]; acc += ((double)v.x * v.x + (double)v.y * v.y) * (double)ips[k]; }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) S[warp] = acc;
}
'''.replace('BL_', str(BL))
_fn = None


def _block_sums(xp, h, ips, nblk):
    F = h.shape[1]
    if xp is np:
        w = (h.real.astype(np.float64) ** 2 + h.imag.astype(np.float64) ** 2) * ips.astype(np.float64)[None, :]
        w = np.concatenate([w, np.zeros((h.shape[0], nblk * BL - F))], axis=1)
        return w.reshape(h.shape[0], nblk, BL).sum(2)
    global _fn
    if _fn is None:
        _fn = xp.RawKernel(_src, 'blocksum')
    h = xp.ascontiguousarray(h); ips = xp.ascontiguousarray(ips.astype(xp.float32))
    S = xp.empty((h.shape[0], nblk), xp.float64)
    nw = h.shape[0] * nblk
    _fn((-(-nw // 8),), (256,), (h, ips, np.int32(F), np.int32(nblk), np.int32(nw), S))
    return S


def edges_blocked(xp, rows, nb_, kmin, k1, Hd, ips):
    """rows: per-peak row index into Hd; nb_, k1: per-peak bin count and end index (functions of the row); ips: 1/psd.
    Returns (npk, B + 1) int edges, padded with k1 beyond each peak's own bin count (as edges_per_template). The work is
    done once per distinct row, then copied to that row's peaks."""
    ur, first, inv = np.unique(rows, return_index=True, return_inverse=True)
    nbu, k1u = nb_[first], k1[first]
    u = len(ur)
    B = int(nb_.max())
    h = Hd[xp.asarray(ur)]
    F = h.shape[1]
    nblk = -(-F // BL)
    S = _block_sums(xp, h, ips, nblk)
    C = xp.cumsum(S, axis=1)                                               # C[b] = sum over bins < (b + 1) BL
    ipsd = ips.astype(xp.float64)
    ar = xp.arange(u)

    # total weight below k1 for each row
    kend = xp.asarray(k1u)
    b = kend // BL
    base = xp.where(b > 0, C[ar, xp.maximum(b - 1, 0)], 0.0)
    kk = b[:, None] * BL + xp.arange(BL)[None, :]
    kc = xp.minimum(kk, F - 1)
    hv = h[ar[:, None], kc]
    wv = (hv.real.astype(xp.float64) ** 2 + hv.imag.astype(xp.float64) ** 2) * ipsd[kc]
    tot = base + xp.where(kk < kend[:, None], wv, 0.0).sum(1)

    j = np.arange(B)[None, :]
    target = xp.asarray(j / nbu[:, None]) * tot[:, None]                   # edge = first k with cumsum[k] > target
    blk = xp.stack([(C <= target[:, c:c + 1]).sum(1) for c in range(B)], 1)
    blk = xp.minimum(blk, nblk - 1)
    before = xp.where(blk > 0, xp.take_along_axis(C, xp.maximum(blk - 1, 0), 1), 0.0)
    kk = blk[..., None] * BL + xp.arange(BL)[None, None, :]
    kc = xp.minimum(kk, F - 1)
    hv = h[ar[:, None, None], kc]
    wv = xp.where(kk < F, (hv.real.astype(xp.float64) ** 2 + hv.imag.astype(xp.float64) ** 2) * ipsd[kc], 0.0)
    loc = xp.cumsum(wv, axis=2) + before[..., None]
    e = blk * BL + (loc <= target[..., None]).sum(2)
    e = e.get() if hasattr(e, 'get') else np.asarray(e)
    e = np.clip(e, kmin, k1u[:, None])
    e = np.where(j < nbu[:, None], e, k1u[:, None])
    return np.concatenate([e, k1u[:, None]], axis=1)[inv]


_fsrc = r'''
#define NT 256
extern "C" __global__ void edges_fused(const float2* __restrict__ h, const long long ldh, const int* __restrict__ rows,
        const float* __restrict__ ips, const int* __restrict__ k1s, const int* __restrict__ nbs, const int nedge,
        int* __restrict__ out)
{
    // one block per row: float64 sum of |h|^2/psd over contiguous chunks, block scan, then each thread finds the
    // edges (first k with cumsum[k] > j/nb * total) whose targets fall inside its own chunk
    const int g = blockIdx.x, t = threadIdx.x;
    const float2* hr = h + (long long)rows[g] * ldh;
    const int k1 = k1s[g], nb = nbs[g];
    const int chunk = (k1 + NT - 1) / NT, a = min(t * chunk, k1), b = min(a + chunk, k1);
    double s = 0.0;
    for (int k = a; k < b; k++) { const float2 v = hr[k]; s += ((double)v.x * v.x + (double)v.y * v.y) * (double)ips[k]; }
    __shared__ double pre[NT + 1];
    pre[t + 1] = s;
    if (t == 0) pre[0] = 0.0;
    __syncthreads();
    if (t == 0) for (int i = 1; i <= NT; i++) pre[i] += pre[i - 1];
    __syncthreads();
    const double tot = pre[NT], lo = pre[t], hi = pre[t + 1];
    int* e = out + (long long)g * nedge;
    for (int i = t; i < nedge; i += NT) e[i] = k1;
    __syncthreads();
    if (b <= a) return;
    int j = 0;
    while (j < nb && ((double)j / (double)nb) * tot < lo) j++;
    if (j >= nb || ((double)j / (double)nb) * tot >= hi) return;
    double cs = lo;
    for (int k = a; k < b && j < nb; k++) {
        const float2 v = hr[k]; cs += ((double)v.x * v.x + (double)v.y * v.y) * (double)ips[k];
        while (j < nb && cs > ((double)j / (double)nb) * tot) { e[j] = k; j++; }
    }
    while (j < nb && ((double)j / (double)nb) * tot < hi) { e[j] = b; j++; }   // rounding between scan and walk
}
'''
_ffn = None


def edges_fused(xp, rows, nb_, kmin, k1, Hd, ips):
    """edges_blocked for one peak per row, in one GPU kernel without gathering rows: rows index Hd directly. Same
    (npk, B + 1) output; edges equal the float64 cumulative-sum ones up to summation order."""
    global _ffn
    if _ffn is None:
        _ffn = xp.RawKernel(_fsrc, 'edges_fused')
    B = int(nb_.max()); n = len(rows)
    out = xp.empty((n, B + 1), xp.int32)
    assert Hd.flags.c_contiguous
    _ffn((n,), (256,), (Hd, np.int64(Hd.shape[1]), xp.asarray(rows.astype(np.int32)), xp.ascontiguousarray(ips.astype(xp.float32)),
                        xp.asarray(k1.astype(np.int32)), xp.asarray(nb_.astype(np.int32)), np.int32(B + 1), out))
    e = out.get().astype(np.int64)
    j = np.arange(B + 1)[None, :]
    e = np.clip(e, kmin, k1[:, None])
    return np.where(j < nb_[:, None], e, k1[:, None])
