"""Sine-Gaussian chi-squared (pycbc.vetoes.sgchisq.SingleDetSGChisq.values) for many peaks of one segment in one GPU
launch: one CUDA block per (peak, tile). Same index conventions as fp_filter.sg_chisq_one (each tile's lower edge is
bounded by the previous tile's start; bins below round(flo / df) get weight 0), so the host computes every tile's
bin range in float64 exactly as the reference does, and the kernel only sums.

Per (peak, tile) the block sums g_k D_k exp(2 pi i k t / N) and g_k^2 IP_k over k in [kk, k1) in float32. The phasor
exp(2 pi i k t / N) is advanced by a recurrence and reseeded exactly every R bins from the integer phase k t mod N
(32-bit wraparound; N is a power of two). Relative error against the float64 reference is about 1e-6 (measured in
test_sg_kernel.py), far below anything the sg veto (thresholds at sg_chisq ~ 4) can see.
"""
import numpy as np

R = 32         # bins per thread between exact phasor reseeds
NT = 256       # threads per block

_SRC = r'''
#define R %(R)d
#define NT %(NT)d
extern "C" __global__ void sg_tiles(const float2* __restrict__ D, const float* __restrict__ IP,
        const unsigned int nmask, const float two_pi_over_n, const double df,
        const long long* __restrict__ kk, const long long* __restrict__ k1, const long long* __restrict__ kg,
        const double* __restrict__ fcen, const double* __restrict__ tau, const double* __restrict__ q2,
        const unsigned int* __restrict__ t, const int ntile, float2* __restrict__ zout, float* __restrict__ gsq)
{
    const int b = blockIdx.x, p = b / ntile;
    const long long a = kk[b], e = k1[b], g0 = kg[b];
    const float fc = (float)fcen[b], ta = (float)tau[b], qq = (float)(q2[b] / fcen[b]);
    const float amp = 0.886226925452758f * ta;                 // sqrt(pi) / 2 tau
    const float pit = 3.14159265358979f * ta;
    const unsigned int tp = t[p];
    float sre = 0.f, sim = 0.f, s2 = 0.f;
    float cs, sn; sincospif(2.f * (float)(tp & nmask) / (float)(nmask + 1u), &sn, &cs);
    for (long long c = a + (long long)threadIdx.x * R; c < e; c += (long long)NT * R) {
        unsigned int ph = ((unsigned int)c * tp) & nmask;
        float pr, pi_; sincospif(2.f * (float)ph / (float)(nmask + 1u), &pi_, &pr);
        const long long ce = min(c + R, e);
        for (long long k = c; k < ce; k++) {
            if (k >= g0) {
                const float f = (float)((double)k * df);
                const float x = pit * (f - fc);
                const float g = amp * __expf(-x * x) * (1.f + __expf(-qq * f));
                const float2 d = D[k];
                const float wr = d.x * pr - d.y * pi_, wi = d.x * pi_ + d.y * pr;   // D_k e^{i phase}
                sre += g * wr; sim += g * wi; s2 += g * g * IP[k];
            }
            const float nr = pr * cs - pi_ * sn; pi_ = pr * sn + pi_ * cs; pr = nr;
        }
    }
    __shared__ float r0[NT], r1[NT], r2[NT];
    r0[threadIdx.x] = sre; r1[threadIdx.x] = sim; r2[threadIdx.x] = s2;
    __syncthreads();
    for (int s = NT / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { r0[threadIdx.x] += r0[threadIdx.x + s]; r1[threadIdx.x] += r1[threadIdx.x + s]; r2[threadIdx.x] += r2[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { zout[b] = make_float2(r0[0], r1[0]); gsq[b] = r2[0]; }
}
''' % dict(R=R, NT=NT)

_kern = None


def tile_ranges(fpeak, vals, k_start, df):
    """Per peak and tile, the reference's bin range and constants, float64: kk, k1, kg (int64), fcen, tau, q^2,
    each (npk, ntile). Same arithmetic as fp_filter.sg_chisq_one."""
    npk, nt = len(fpeak), len(vals)
    kk_o = np.empty((npk, nt), np.int64); k1_o = np.empty_like(kk_o); kg_o = np.empty_like(kk_o)
    fc_o = np.empty((npk, nt)); tau_o = np.empty_like(fc_o); q2_o = np.empty_like(fc_o)
    kk = np.asarray(k_start, np.int64).copy()
    for j, (q, off) in enumerate(vals):
        fcen = fpeak + off
        flo = np.maximum(kk * df, fcen - 50); fhi = fcen + 50
        kk = (flo / df).astype(np.int64); k1 = (fhi / df).astype(np.int64)
        kg = np.round(flo / df).astype(np.int64)
        kk_o[:, j], k1_o[:, j], kg_o[:, j] = kk, k1, kg
        fc_o[:, j], tau_o[:, j], q2_o[:, j] = fcen, q / 2 / np.pi / fcen, q * q
    return kk_o, k1_o, kg_o, fc_o, tau_o, q2_o


def sg_chisq_points(D, IP, t, fpeak, vals, k_start, df, tlen):
    """sg_chisq for peaks at samples t (int array) sharing the tile list vals; fpeak, k_start per peak. D, IP: one
    segment's over-whitened data and 1/psd on the GPU. Returns (values float64, computed mask); like sg_chisq_batch,
    a peak whose tiles reach above 0.9 x Nyquist is not computed (value 1)."""
    import cupy as cp
    global _kern
    if _kern is None:
        _kern = cp.RawKernel(_SRC, 'sg_tiles')
    flen = tlen // 2 + 1
    fpeak = np.asarray(fpeak, np.float64); t = np.asarray(t, np.int64)
    ok = np.all([fpeak + off + 50 <= flen * df * 0.9 for _, off in vals], axis=0)
    out = np.ones(len(t), np.float64)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return out, ok
    kk, k1, kg, fc, tau, q2 = tile_ranges(fpeak[idx], vals, np.asarray(k_start)[idx], df)
    assert k1.max() <= flen
    nt = len(vals); nb = len(idx) * nt
    z = cp.empty(nb, cp.complex64); gs = cp.empty(nb, cp.float32)
    a_ = lambda x: cp.asarray(np.ascontiguousarray(x).ravel())
    _kern((nb,), (NT,), (D, IP, np.uint32(tlen - 1), np.float32(2 * np.pi / tlen), np.float64(df),
                         a_(kk), a_(k1), a_(kg), a_(fc), a_(tau), a_(q2), cp.asarray(t[idx].astype(np.uint32)),
                         np.int32(nt), z, gs))
    zz = (z.get().astype(np.complex128) * (4.0 * df)).reshape(-1, nt)
    gsig = np.sqrt(4 * df * gs.get().astype(np.float64)).reshape(-1, nt)
    out[idx] = (np.abs(zz / gsig) ** 2).sum(1) / (2 * nt)
    return out, ok
