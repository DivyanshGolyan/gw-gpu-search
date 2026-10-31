"""GPU check for fp_filter v9's basis_series: multiplying by the float32 reciprocal of a 0-d device scale gives the same
bits as v8's division by the Python float scale.item() (CUDA torch turns a CPU-scalar divisor into a reciprocal
multiply). Also checks stacked_A's product. Run on the GPU: python test_scale_gpu.py"""
import torch
torch.manual_seed(0)
bad = 0
for trial in range(50):
    n, T = int(torch.randint(1, 64, ())), int(torch.randint(1000, 300000, ()))
    zb = (torch.randn(n, T, device='cuda', dtype=torch.complex64) * float(10 ** torch.empty(()).uniform_(-30, 30)))[:, 7:]
    s_t = torch.sqrt((zb.real ** 2 + zb.imag ** 2).mean() / 2)
    s_f = float(s_t.item())
    inv = torch.reciprocal(s_t)
    for part in (zb.real, zb.imag):
        old = (part / s_f).half(); new = (part * inv).half()
        old32 = part / s_f; new32 = part * inv
        bad += int(not torch.equal(old, new)) + int(not torch.equal(old32, new32))
    nrm = torch.rand(512, device='cuda') * 1e3
    bad += int(not torch.equal(nrm * s_f, nrm * s_t))
print('scale check: 50 trials,', 'IDENTICAL' if bad == 0 else f'{bad} MISMATCHES')
raise SystemExit(1 if bad else 0)
