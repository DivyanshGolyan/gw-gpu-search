"""Cheap frequency-domain templates: evaluate the waveform on a coarse grid on the CPU, fit a cubic spline
to the unwrapped phase there, then evaluate amplitude (linear) and phase (the spline) on the fine filter
grid on the GPU in one elementwise kernel.

The coarse grid is the union of points uniform in f^(-5/3), which makes the inspiral phase step
roughly constant from point to point, and a log grid that resolves merger and ringdown.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def coarse_grid(flow, fhigh, n_chirp=6000, n_log=1500):
    u = np.linspace(flow ** (-5 / 3), fhigh ** (-5 / 3), n_chirp) ** (-3 / 5)
    g = np.geomspace(flow, fhigh, n_log)
    x = np.unique(np.r_[u, g, flow, fhigh])
    return x[(x >= flow) & (x <= fhigh)]


def coarse_generate(rows, x, approximant, flow, threads=8, distance=None):
    """Amplitude and phase-spline coefficients of h+ at the coarse points x for each bank row.
    Returns amp (n, len(x)), coef (n, 4, len(x) - 1) with scipy's highest-order-first layout,
    and the max phase step between neighbouring points."""
    from pycbc.waveform import get_fd_waveform_sequence
    from pycbc.types import Array
    from pycbc import DYN_RANGE_FAC
    from scipy.interpolate import CubicSpline
    dist = 1.0 / DYN_RANGE_FAC if distance is None else distance
    xs = Array(x.astype(np.float64))

    def one(r):
        hp, _ = get_fd_waveform_sequence(approximant=approximant, mass1=float(r.mass1), mass2=float(r.mass2),
                                         spin1z=float(r.spin1z), spin2z=float(r.spin2z), sample_points=xs,
                                         distance=dist, f_ref=flow)
        h = np.asarray(hp.numpy(), np.complex128)
        ph = np.unwrap(np.angle(h))
        return np.abs(h), CubicSpline(x, ph).c, float(np.abs(np.diff(ph)).max())
    with ThreadPoolExecutor(threads) as pool:
        out = list(pool.map(one, rows))
    amp = np.stack([o[0] for o in out]); coef = np.stack([o[1] for o in out])
    return amp, coef, max(o[2] for o in out)


_EVAL = r'''
    int b = i / K, k = i % K, j = I[k];
    long long nx = NX;
    double d = DX[k], t = T[k];
    long long c = (long long)b * 4 * (nx - 1) + j;
    double p = ((C[c] * d + C[c + (nx - 1)]) * d + C[c + 2 * (nx - 1)]) * d + C[c + 3 * (nx - 1)];
    double a = A[b * nx + j] * (1.0 - t) + A[b * nx + j + 1] * t;
    double s, co;
    sincos(p, &s, &co);
    out = complex<float>((float)(a * co), (float)(a * s));
'''


class FineInterpolator:
    """Maps coarse (amp, coef) batches to complex64 templates on k * df, k in [kmin, kend)."""

    def __init__(self, x, df, kmin, kend, flen, xp):
        self.xp, self.flen, self.kmin, self.kend = xp, flen, kmin, kend
        f = np.clip(np.arange(kmin, kend) * df, x[0], x[-1])
        i = np.clip(np.searchsorted(x, f, side='right') - 1, 0, len(x) - 2)
        self.nx = len(x)
        self.i = xp.asarray(i.astype(np.int32))
        self.dx = xp.asarray(f - x[i])
        self.t = xp.asarray((f - x[i]) / (x[i + 1] - x[i]))
        if xp is not np:
            self.kern = xp.ElementwiseKernel(
                'raw float64 C, raw float64 A, raw int32 I, raw float64 DX, raw float64 T, int32 K, int32 NX',
                'complex64 out', _EVAL, 'interp_eval')

    def __call__(self, amp, coef):
        xp = self.xp
        out = xp.zeros((amp.shape[0], self.flen), xp.complex64)
        if xp is np:
            i, d, t = self.i, self.dx, self.t
            c = coef[:, :, i]
            p = ((c[:, 0] * d + c[:, 1]) * d + c[:, 2]) * d + c[:, 3]
            a = amp[:, i] * (1 - t) + amp[:, i + 1] * t
            out[:, self.kmin:self.kend] = (a * np.exp(1j * p)).astype(np.complex64)
            return out
        A = xp.asarray(amp, xp.float64); C = xp.asarray(coef, xp.float64)
        self.kern(C, A, self.i, self.dx, self.t, np.int32(self.kend - self.kmin), np.int32(self.nx),
                  out[:, self.kmin:self.kend])
        return out
