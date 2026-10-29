"""Fused |SNR| peak finding: for each (template row, cluster window) one block reads the complex
SNR series once and returns the max |z|^2 and its sample index. Replaces abs, scale, reshape,
argmax and take_along_axis, which each made a full pass over the (rows, N) array."""
import numpy as np
import cupy as cp

THREADS = 256
_src = r'''
#define NT %(NT)d
extern "C" __global__ void win_peak(const float2* __restrict__ z, const unsigned int ldz,
        const int a0, const int cw, const int nwin, float* __restrict__ pmax, int* __restrict__ pidx)
{
    const int w = blockIdx.x, row = blockIdx.y;
    const float2* zr = z + (size_t)row * ldz + a0 + (size_t)w * cw;
    float best = -1.f; int bi = 0;
    for (int i = threadIdx.x; i < cw; i += NT) {
        const float2 v = zr[i];
        const float p = v.x * v.x + v.y * v.y;
        if (p > best) { best = p; bi = i; }
    }
    // first index wins on ties, matching numpy argmax
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_down_sync(0xffffffff, best, o);
        const int oi = __shfl_down_sync(0xffffffff, bi, o);
        if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
    }
    __shared__ float sb[NT / 32]; __shared__ int si[NT / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sb[warp] = best; si[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < NT / 32; k++)
            if (sb[k] > best || (sb[k] == best && si[k] < bi)) { best = sb[k]; bi = si[k]; }
        pmax[(size_t)row * nwin + w] = best; pidx[(size_t)row * nwin + w] = bi;
    }
}
''' % dict(NT=THREADS)
_fn = cp.RawKernel(_src, 'win_peak')


def window_peaks(z, a0, cw, nwin):
    """z: (rows, N) complex64. Returns (max |z|^2, argmax within window), each (rows, nwin)."""
    assert z.dtype == cp.complex64 and z.flags.c_contiguous
    rows = z.shape[0]
    pmax = cp.empty((rows, nwin), cp.float32); pidx = cp.empty((rows, nwin), cp.int32)
    _fn((nwin, rows), (THREADS,), (z, np.uint32(z.shape[1]), np.int32(a0), np.int32(cw), np.int32(nwin), pmax, pidx))
    return pmax, pidx
