"""Fused tensor-core GEMM + |z|^2 + max per BN output samples (Triton), for fp_filter.py.

z_re = Are @ Z, z_im = Aim @ Z, with Z (K x Tn) the stacked real/imag basis series. Each program keeps only the max
|z|^2 and its index per BN samples, so the M x Tn SNR array never reaches GPU memory. Lives in its own module because
Triton resolves a kernel's names from the module globals (a kernel defined inside a function cannot see tl)."""
import time
import torch
import triton
import triton.language as tl

BN = 128


@triton.jit
def gemm_absmax(Are, Aim, Z, V, I, M, Tn, K, s_a, s_z, s_v, BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc_r = tl.zeros((BM, BN), dtype=tl.float32)
    acc_i = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ar = tl.load(Are + rm[:, None] * s_a + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.)
        ai = tl.load(Aim + rm[:, None] * s_a + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.)
        z = tl.load(Z + (k0 + rk).to(tl.int64)[:, None] * s_z + rn[None, :], mask=rn[None, :] < Tn, other=0.)
        acc_r += tl.dot(ar, z)
        acc_i += tl.dot(ai, z)
    pw = acc_r * acc_r + acc_i * acc_i
    pw = tl.where(rn[None, :] < Tn, pw, -1.0)
    v = tl.max(pw, axis=1)
    i = tl.argmax(pw, axis=1)
    tl.store(V + rm * s_v + pid_n, v, mask=rm < M)
    tl.store(I + rm * s_v + pid_n, pid_n * BN + i, mask=rm < M)


def fused_peaks(Are, Aim, Z, M, Tn, cfg):
    """Are, Aim (M x K) and Z (K x Tn) fp16, K a multiple of cfg's BK. Returns max |z|^2 and its sample, (M, ceil(Tn/BN))."""
    bm, bk, w, st = cfg
    nblk = triton.cdiv(Tn, BN)
    V = torch.empty((M, nblk), device=Z.device, dtype=torch.float32)
    I = torch.empty((M, nblk), device=Z.device, dtype=torch.int32)
    gemm_absmax[(triton.cdiv(M, bm), nblk)](Are, Aim, Z, V, I, M, Tn, Are.shape[1], Are.stride(0), Z.stride(0),
                                            V.stride(0), BM=bm, BK=bk, BN=BN, num_warps=w, num_stages=st)
    return V, I


def pick_cfg(dev='cuda', M=1024, K=128, Tn=1 << 20):
    """Compile every candidate once and keep the fastest on a representative shape. Raises if none compiles."""
    Ar = torch.randn((M, K), device=dev).half(); Ai = torch.randn((M, K), device=dev).half()
    Zs = torch.randn((K, Tn), device=dev).half()
    best, res = None, {}
    for c in [(bm, bk, w, st) for bm in (64, 128) for bk in (32, 64) for w in (4, 8) for st in (2, 3)]:
        try:
            fused_peaks(Ar, Ai, Zs, M, Tn, c); torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(3):
                fused_peaks(Ar, Ai, Zs, M, Tn, c)
            torch.cuda.synchronize(); res[str(c)] = (time.perf_counter() - t0) / 3
        except Exception as ex:
            res[str(c)] = repr(ex)[:300]; continue
        if best is None or res[str(c)] < res[str(best)]:
            best = c
    if best is None:
        raise RuntimeError('no GEMM config compiled: %r' % res)
    return best, res
