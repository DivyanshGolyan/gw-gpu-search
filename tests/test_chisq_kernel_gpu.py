"""GPU unit test: chisq_kernel.bin_power_sums vs a float64 direct sum, random data, N = 2^21."""
import numpy as np, cupy as cp, time, sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pipeline'))
from chisq_kernel import bin_power_sums
rng = np.random.default_rng(1)
N = 2 ** 21; F = N // 2 + 1; nrow = 4
q = cp.zeros((nrow, N), cp.complex64)
q[:, :F] = cp.asarray((rng.standard_normal((nrow, F)) + 1j * rng.standard_normal((nrow, F))).astype(np.complex64))
rows = np.array([0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]); npk = len(rows)
times = rng.integers(0, N, npk)
nb = 40
edges_row = np.sort(rng.choice(np.arange(30 * 1024, F), size=(nrow, nb + 1), replace=False), axis=1)
E = edges_row[rows]
got = bin_power_sums(q, rows, times, E)
qq = q.astype(cp.complex128)
ref = np.zeros(npk)
for i in range(npk):
    k = cp.arange(E[i, 0], E[i, -1], dtype=cp.int64)
    ph = cp.exp(2j * np.pi * ((k * int(times[i])) % N) / N)
    c = cp.concatenate([cp.zeros(1, cp.complex128), cp.cumsum(qq[rows[i], E[i, 0]:E[i, -1]] * ph)])
    Z = c[E[i, 1:] - E[i, 0]] - c[E[i, :-1] - E[i, 0]]
    ref[i] = float((cp.abs(Z) ** 2).sum())
rel = np.abs(got - ref) / ref
print('rel err max', rel.max(), 'median', np.median(rel))
# speed: 128 rows x 8 peaks, 200 bins over ~1M k
q2 = cp.zeros((128, N), cp.complex64); q2[:, :F] = 1
rows2 = np.repeat(np.arange(128), 8); t2 = rng.integers(0, N, len(rows2))
E2 = np.tile(np.linspace(30 * 1024, F - 1, 201).astype(np.int64), (len(rows2), 1))
bin_power_sums(q2, rows2, t2, E2); cp.cuda.Device().synchronize()
t0 = time.perf_counter(); bin_power_sums(q2, rows2, t2, E2); cp.cuda.Device().synchronize()
print('1024 peaks x 1M k:', time.perf_counter() - t0, 's')
assert rel.max() < 1e-3, 'kernel mismatch'
