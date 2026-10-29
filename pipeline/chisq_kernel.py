"""GPU power chi-squared at peaks: one CUDA block per (template row, up to P peaks).

For each bin j of the template's equal-power bins, the block sums
q[row, k] * exp(+2 pi i k t / N) over k in the bin for each of its peaks, reduces
across threads, and accumulates |Z_j|^2. q is read once per group of P peaks, and the
phase index k*t mod N uses 32-bit wraparound (N is a power of two), so no int64 mod.
"""
import numpy as np
import cupy as cp

P = 8          # peaks per block (register accumulators)
THREADS = 256

_src = r'''
#define P %(P)d
#define NT %(NT)d
extern "C" __global__ void KNAME(const float2* __restrict__ q, const unsigned int ldq, const float2* __restrict__ dv,
        const unsigned int nmask, const float inv_half_n,
        const int* __restrict__ grow, const int* __restrict__ gstart, const int* __restrict__ gcount,
        const unsigned int* __restrict__ ptime, const int* __restrict__ edges, const int nedge,
        float* __restrict__ out, float2* __restrict__ zout)
{
    const int g = blockIdx.x;
    const int row = grow[g], p0 = gstart[g], np_ = gcount[g];
    const float2* qr = q + (size_t)row * ldq;
    const int* e = edges + (size_t)g * nedge;
    unsigned int t[P];
    for (int p = 0; p < P; p++) t[p] = (p < np_) ? ptime[p0 + p] : 0u;
    __shared__ float2 red[NT / 32][P];
    float tot[P]; float2 zt[P];
    for (int p = 0; p < P; p++) { tot[p] = 0.f; zt[p] = make_float2(0.f, 0.f); }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int j = 0; j < nedge - 1; j++) {
        const int k0 = e[j], k1 = e[j + 1];
        if (k0 >= k1) continue;                       // uniform across the block
        float2 acc[P];
        for (int p = 0; p < P; p++) acc[p] = make_float2(0.f, 0.f);
        for (int k = k0 + threadIdx.x; k < k1; k += NT) {
#ifdef FUSE
            const float2 hk = qr[k], dk = dv[k];                      // q = conj(h) d, never written to memory
            const float2 v = make_float2(hk.x * dk.x + hk.y * dk.y, hk.x * dk.y - hk.y * dk.x);
#else
            const float2 v = qr[k];
#endif
            #pragma unroll
            for (int p = 0; p < P; p++) {
                if (p < np_) {
                    const unsigned int m = ((unsigned int)k * t[p]) & nmask;   // k t mod N, 32-bit wrap is exact
                    float s, c;
                    sincospif((float)m * inv_half_n, &s, &c);                // angle 2 pi m / N
                    acc[p].x += v.x * c - v.y * s;
                    acc[p].y += v.x * s + v.y * c;
                }
            }
        }
        #pragma unroll
        for (int p = 0; p < P; p++) {
            float ax = acc[p].x, ay = acc[p].y;
            for (int o = 16; o > 0; o >>= 1) {
                ax += __shfl_down_sync(0xffffffff, ax, o);
                ay += __shfl_down_sync(0xffffffff, ay, o);
            }
            if (lane == 0) red[warp][p] = make_float2(ax, ay);
        }
        __syncthreads();
        if (threadIdx.x < P) {
            const int p = threadIdx.x;
            float ax = 0.f, ay = 0.f;
            for (int w = 0; w < NT / 32; w++) { ax += red[w][p].x; ay += red[w][p].y; }
            tot[p] += ax * ax + ay * ay;
            zt[p].x += ax; zt[p].y += ay;
        }
        __syncthreads();
    }
    if (threadIdx.x < P && threadIdx.x < np_) {
        out[p0 + threadIdx.x] = tot[threadIdx.x];
        if (zout) zout[p0 + threadIdx.x] = zt[threadIdx.x];
    }
}
''' % dict(P=P, NT=THREADS)
_fn = cp.RawKernel(_src.replace('KNAME', 'chisq_pts'), 'chisq_pts')
_fn_fused = cp.RawKernel('#define FUSE\n' + _src.replace('KNAME', 'chisq_pts_fused'), 'chisq_pts_fused')


def bin_power_sums(q, rows, times, edges_per_peak, want_z=False, d=None, n_time=None):
    """sum_j |Z_j(t)|^2 for each peak; q is (nrows, N) complex64 on the GPU, N a power of two.
    rows, times: int arrays per peak; edges_per_peak: (npeaks, nedge) int bin edges into q's row.
    want_z: also return z(t) = sum_j Z_j (the complex SNR numerator at the peak, exact when the
    bins cover all of q's nonzero band), as a second complex64 array.
    d, n_time: fused mode. q is then the template rows H (nrows, flen) and d the data (flen,); the kernel forms
    conj(H) d itself, and n_time (a power of two) is the time length N of the phase factor."""
    N = q.shape[1] if d is None else n_time
    assert N & (N - 1) == 0, 'N must be a power of two'
    assert q.dtype == cp.complex64 and q.flags.c_contiguous
    npk = len(rows)
    o = np.lexsort((times, rows))
    rows, times, edges_per_peak = rows[o], times[o], edges_per_peak[o]
    # groups of up to P peaks of the same row
    brk = np.r_[0, np.nonzero(np.diff(rows))[0] + 1, npk]
    starts, ends = [], []
    for s0, s1 in zip(brk[:-1], brk[1:]):
        for s in range(s0, s1, P):
            starts.append(s); ends.append(min(s + P, s1))
    starts = np.array(starts, np.int32); ends = np.array(ends, np.int32)
    gcount = (ends - starts).astype(np.int32)
    grow = rows[starts].astype(np.int32)
    gedges = np.ascontiguousarray(edges_per_peak[starts].astype(np.int32))   # edges depend on the row only
    out = cp.zeros(npk, cp.float32)
    zo = cp.zeros(npk, cp.complex64) if want_z else None
    fn = _fn if d is None else _fn_fused
    fn((len(starts),), (THREADS,),
        (q, np.uint32(q.shape[1]), d if d is not None else np.uint64(0), np.uint32(N - 1), np.float32(2.0 / N),
         cp.asarray(grow), cp.asarray(starts), cp.asarray(gcount),
         cp.asarray(times.astype(np.uint32)), cp.asarray(gedges), np.int32(gedges.shape[1]), out, zo if want_z else np.uint64(0)))
    res = np.empty(npk, np.float32)
    res[o] = out.get()
    if not want_z:
        return res
    zr = np.empty(npk, np.complex64)
    zr[o] = zo.get()
    return res, zr
