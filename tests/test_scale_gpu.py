"""GPU check for fp_filter v9's basis_series: multiplying by the float64-formed reciprocal of the 0-d device scale gives the same bits as v8's division by
the Python float scale.item(). Also reports two other forms (true division by the tensor, torch.reciprocal), and checks stacked_A's product.
Run on the GPU: python test_scale_gpu.py (exit 1 if the form v9 uses differs)."""
import torch
torch.manual_seed(0)
bad = dict(v9_mul_inv64=0, div_tensor=0, mul_reciprocal=0, stacked_A=0)
for trial in range(50):
    n, T = int(torch.randint(1, 64, ())), int(torch.randint(1000, 300000, ()))
    zb = (torch.randn(n, T, device='cuda', dtype=torch.complex64) * float(10 ** torch.empty(()).uniform_(-30, 30)))[:, 7:]
    s_t = torch.sqrt((zb.real ** 2 + zb.imag ** 2).mean() / 2)
    s_f = float(s_t.item())
    for part in (zb.real, zb.imag):
        old = part / s_f
        inv = (1.0 / s_t.double()).float()
        bad['v9_mul_inv64'] += int(not torch.equal(old, part * inv)) + int(not torch.equal(old.half(), (part * inv).half()))
        bad['div_tensor'] += int(not torch.equal(old, part / s_t)) + int(not torch.equal(old.half(), (part / s_t).half()))
        bad['mul_reciprocal'] += int(not torch.equal(old, part * torch.reciprocal(s_t)))
    nrm = torch.rand(512, device='cuda') * 1e3
    bad['stacked_A'] += int(not torch.equal(nrm * s_f, nrm * s_t))
print('scale check, 50 trials, mismatches:', bad, 'v9 form', 'IDENTICAL' if bad['v9_mul_inv64'] + bad['stacked_A'] == 0 else 'DIFFERS')
raise SystemExit(1 if bad['v9_mul_inv64'] + bad['stacked_A'] else 0)
