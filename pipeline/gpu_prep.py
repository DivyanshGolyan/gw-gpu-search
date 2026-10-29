#!/usr/bin/env python
"""PyCBC data conditioning (fp_filter.prep_one) on the GPU, step for step.

pycbc.strain.from_cli + StrainSegments + associate_psds_to_segments, for the options our argv uses:
  read hdf store (CPU) -> lal IIR highpass, float64 (CPU) -> 'ldas' FIR resample (GPU) -> injections (CPU, added on GPU)
  -> x DYN_RANGE_FAC, float32 -> autogating: whiten with a 4 s median-Welch PSD, |x| > threshold, findchirp cluster,
  Tukey gates (GPU; cluster and gate times on the CPU with LIGOTimeGPS, as PyCBC) -> lal IIR highpass, float32 (CPU)
  -> remove padding -> segment FFTs, overlapping median-Welch PSDs, interpolation, inverse spectrum truncation (GPU)
  -> D = stilde / psd and IP = 1/psd, zero below flow.
Every FFT keeps PyCBC's precision and scale factors (all exact powers of two here), so outputs agree with PyCBC to
float32 rounding. The two IIR highpasses stay in lal on the CPU (sequential recursions; ~2 s per 8.6 ks piece).
Imports nothing from pycbc unless --injection-file is given, so a piece costs no module-import time.

xp is cupy (production) or numpy (a CPU check of the same code path against PyCBC).

Use: prep(argv, xp) -> dict(D, IP (xp arrays), gates, flen, tlen, df, sr, gps_start, analyze, seg_start, timing)
"""
import argparse, time
import numpy as np

DYN_RANGE_FAC = 5.9029581035870565e+20      # pycbc.DYN_RANGE_FAC (checked in test_gpu_prep.py)


def parser():
    p = argparse.ArgumentParser(add_help=False)
    a = p.add_argument
    a('--hdf-store'); a('--channel-name'); a('--gps-start-time', type=int); a('--gps-end-time', type=int)
    a('--pad-data', type=int, default=0); a('--strain-high-pass', type=float); a('--sample-rate', type=float)
    a('--low-frequency-cutoff', type=float); a('--segment-length', type=int)
    a('--segment-start-pad', type=int); a('--segment-end-pad', type=int)
    a('--trig-start-time', type=int, default=0); a('--trig-end-time', type=int, default=0)
    a('--psd-estimation'); a('--psd-segment-length', type=float); a('--psd-segment-stride', type=float)
    a('--psd-num-segments', type=int); a('--psd-inverse-length', type=float); a('--invpsd-trunc-method')
    a('--autogating-threshold', type=float); a('--autogating-cluster', type=float, default=5.0)
    a('--autogating-width', type=float, default=0.25); a('--autogating-taper', type=float, default=0.25)
    a('--autogating-pad', type=float, default=16); a('--autogating-max-iterations', type=int, default=1)
    a('--gating-method', default='taper')
    a('--injection-file'); a('--injection-scale-factor', type=float, default=1); a('--injection-sample-rate', type=float)
    # options this port does not implement: refuse them rather than silently differ from PyCBC
    for k in ('--gating-file', '--zpk-z', '--normalize-strain', '--strain-low-pass', '--taper-data', '--witness-frame-type',
              '--sgburst-injection-file', '--psd-model', '--psd-file', '--asd-file', '--fake-strain', '--filter-inj-only',
              '--allow-zero-padding', '--frame-files', '--frame-cache', '--frame-type'):
        a(k, dest='unsupported_' + k[2:].replace('-', '_'))
    return p


def median_bias(n):
    if n >= 1000:
        return np.log(2)
    ans = 1
    for i in range(1, (n - 1) // 2 + 1):
        ans += 1.0 / (2 * i + 1) - 1.0 / (2 * i)
    return ans


def highpass_cpu(x, sr, epoch, freq, filter_order=8, attenuation=0.1):
    """pycbc.filter.highpass: lal Butterworth IIR, in place on a host array (float32 or float64)."""
    import lal
    f = {np.dtype('float64'): (lal.CreateREAL8TimeSeries, lal.HighPassREAL8TimeSeries),
         np.dtype('float32'): (lal.CreateREAL4TimeSeries, lal.HighPassREAL4TimeSeries)}[x.dtype]
    ts = f[0]('', lal.LIGOTimeGPS(epoch), 0, 1.0 / sr, lal.StrainUnit, len(x))
    ts.data.data[:] = x
    f[1](ts, freq, 1 - attenuation, filter_order)
    return ts.data.data


def resample_ldas(xp, x, factor):
    """pycbc resample_to_delta_t(method='ldas'): causal FIR y[n] = sum_k c[k] x[n-k] (PyCBC does it by FFT in chunks;
    identical up to float64 rounding), first 2*(numtaps//2) outputs zeroed, rolled by -numtaps//2, decimated."""
    import scipy.signal
    ntaps = factor * 20 + 1
    c = scipy.signal.firwin(ntaps, 1.0 / factor, window=('kaiser', 5))
    L = len(x)
    shift = -(-ntaps // 2)                     # python's -ntaps//2 is -(ntaps//2 + 1) for odd ntaps: roll left by that
    m = xp.arange(0, L, factor)                # output sample i = y[(i + shift) mod L]
    n = m + shift
    out = xp.zeros(len(m), dtype=x.dtype)
    ok = (n >= 2 * (ntaps // 2)) & (n < L)     # y[0:2*(ntaps//2)] are zeroed; wrapped indices land there too
    nv = n[ok]
    acc = xp.zeros(len(nv), dtype=x.dtype)
    for k in range(ntaps):
        acc += c[k] * x[nv - k]
    out[ok] = acc
    return out


def welch(xp, x, seg_len, seg_stride, dt, num_segments=None, avg='median'):
    """pycbc.psd.welch (hann window, require_exact_data_fit=False), for one series or a batch (rows) of equal length.
    Returns float32/64 PSD(s) with PyCBC's normalisation."""
    single = x.ndim == 1
    X = x[None] if single else x
    n = X.shape[1]
    if num_segments is None:
        num_segments = int(n // seg_stride)
        if (num_segments - 1) * seg_stride + seg_len > n:
            num_segments -= 1
    data_len = (num_segments - 1) * seg_stride + seg_len
    if data_len < n:
        diff = n - data_len
        start = diff // 2 + (diff % 2)
        X = X[:, start:n - diff // 2]
    assert X.shape[1] == data_len, 'Incorrect choice of segmentation parameters'
    w = np.hanning(seg_len).astype(X.dtype)
    idx = (xp.arange(num_segments) * seg_stride)[:, None] + xp.arange(seg_len)[None, :]
    wd = xp.asarray(w)
    psds = []
    for row in X:                               # one PSD per row; segments of a row are one batched FFT
        segs = row[idx] * wd
        st = xp.fft.rfft(segs, axis=1)
        st = (st * X.dtype.type(dt)).astype(np.complex64 if X.dtype == np.float32 else np.complex128)
        p = (st.real * st.real + st.imag * st.imag).astype(X.dtype)
        p[:, 0] /= 2
        p[:, -1] /= 2
        if avg == 'median':
            b = median_bias(num_segments)
            # PyCBC: median / bias is float64 when bias is np.log(2) (n >= 1000, a strong numpy scalar), else float32
            q = xp.median(p, axis=0)
            q = q.astype(np.float64) / float(b) if isinstance(b, np.floating) else q / b
        else:
            q = xp.mean(p, axis=0)
        psds.append(q)
    delta_f = 1. / dt / seg_len
    fac = 2 * delta_f * seg_len / (w * w).sum()        # numpy float32 scalar, exactly as PyCBC computes it
    out = (xp.stack(psds) * (float(fac) if psds[0].dtype == np.float64 else fac)).astype(X.dtype)
    return (out[0] if single else out), delta_f


def interpolate(xp, psd, df_old, df_new, length=None):
    """pycbc.psd.interpolate (np.interp in float64, cast back)."""
    new_n = (psd.shape[-1] - 1) * df_old / df_new + 1 if length is None else length
    samples = xp.asarray(np.arange(0, np.rint(new_n)) * df_new)
    fo = xp.asarray(np.arange(psd.shape[-1]) * df_old)
    P = psd[None] if psd.ndim == 1 else psd
    out = xp.stack([xp.interp(samples, fo, r.astype(np.float64)) for r in P]).astype(psd.dtype)
    return out[0] if psd.ndim == 1 else out


def inverse_spectrum_truncation(xp, psd, df, max_filter_len, low_frequency_cutoff, trunc_method=None):
    """pycbc.psd.inverse_spectrum_truncation for one PSD or a batch (rows). ifft then fft scales cancel exactly."""
    P = psd[None] if psd.ndim == 1 else psd
    flen = P.shape[1]
    N = (flen - 1) * 2
    ctype = np.complex64 if P.dtype == np.float32 else np.complex128
    kmin = int(low_frequency_cutoff / df) if low_frequency_cutoff else 1
    inv_asd = xp.zeros(P.shape, dtype=ctype)
    inv_asd[:, kmin:N // 2] = (1.0 / P[:, kmin:N // 2]) ** 0.5
    # PyCBC: q = ifft(inv_asd) * df, psd_trunc = fft(q) / (df N); unnormalised transforms here, scales are 2^k
    q = xp.fft.irfft(inv_asd, n=N, axis=1, norm='forward').astype(P.dtype) * P.dtype.type(df)
    ts, te = max_filter_len // 2, N - max_filter_len // 2
    if trunc_method == 'hann':
        tw = np.hanning(max_filter_len).astype(P.dtype)
        q[:, 0:ts] *= xp.asarray(tw[-ts:])
        q[:, te:N] *= xp.asarray(tw[0:max_filter_len // 2])
    if ts < te:
        q[:, ts:te] = 0
    pt = xp.fft.rfft(q, axis=1).astype(ctype) * P.dtype.type(1.0 / (df * N))
    out = (1. / xp.abs(pt * pt.conj())).astype(P.dtype)
    return out[0] if psd.ndim == 1 else out


def cluster_over_window(idx, vals, window):
    """pycbc.events.findchirp_cluster_over_window (the cython loop)."""
    if len(idx) == 0:
        return np.zeros(0, dtype=int)
    keep = [0]
    for i in range(1, len(idx)):
        if idx[i] < idx[keep[-1]] + window:
            if vals[i] > vals[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    return np.array(keep)


def detect_loud_glitches(xp, strain, sr, start_gps, threshold, cluster_window, low_freq_cutoff, corrupt_time,
                         psd_duration=4., psd_stride=2.):
    """pycbc.strain.detect_loud_glitches (median PSD, no high_freq_cutoff). Returns LIGOTimeGPS times."""
    import lal
    dt = 1.0 / sr
    s = strain.copy()
    n = len(s)
    cl = int(corrupt_time * sr)
    w = (np.arange(cl) / float(cl)).astype(s.dtype)
    s[0:cl] *= xp.asarray(w)
    s[n - cl:] *= xp.asarray(w[::-1])
    L = 1 << n.bit_length()
    pad_start = int(L / 2 - n / 2)
    sp = xp.zeros(L, dtype=s.dtype)
    sp[pad_start:pad_start + n] = s
    psd, pdf = welch(xp, s[cl:n - cl], int(psd_duration * sr), int(psd_stride * sr), dt)
    df = 1. / (L * dt)
    psd = interpolate(xp, psd, pdf, df)
    psd = inverse_spectrum_truncation(xp, psd, df, int(psd_duration * sr), low_freq_cutoff, 'hann')
    psd[0:int(low_freq_cutoff / df)] = np.inf
    norm = sr / 2. - low_freq_cutoff
    ct = np.complex64 if s.dtype == np.float32 else np.complex128
    st = xp.fft.rfft(sp).astype(ct) * s.dtype.type(dt)
    st *= (psd * norm) ** (-0.5)
    wt = xp.fft.irfft(st, n=L, norm='forward').astype(s.dtype) * s.dtype.type(df)
    mag = xp.abs(wt[pad_start:pad_start + n])
    mag[0:cl] = 0
    mag[n - cl:] = 0
    ind = xp.nonzero(mag > threshold)[0]
    ind_h = np.asarray(ind.get() if hasattr(ind, 'get') else ind)
    v = mag[ind]
    v_h = np.asarray(v.get() if hasattr(v, 'get') else v)
    keep = cluster_over_window(ind_h.astype(np.int32), v_h, int(cluster_window * sr))
    t0 = lal.LIGOTimeGPS(start_gps)
    return [int(i) * dt + t0 for i in ind_h[keep]]


def gate_taper(xp, x, sr, start_gps, gates):
    """pycbc.strain.gate_data (inverted Tukey), in place. Offsets use LIGOTimeGPS arithmetic like PyCBC."""
    import lal
    t0 = lal.LIGOTimeGPS(start_gps)
    duration = len(x) / sr
    for gt, gw, pw in gates:
        t_start = gt - gw - pw - t0
        t_end = gt + gw + pw - t0
        if t_start > duration or t_end < 0.:
            continue
        win_samples = int(2 * sr * (gw + pw))
        pad_samples = int(sr * pw)
        pad = 0.5 * (1. + np.cos(np.pi * np.arange(pad_samples) / pad_samples))
        window = np.concatenate((pad, np.zeros(win_samples - 2 * pad_samples), pad[::-1]))
        offset = int(t_start * sr)
        i1, i2 = max(0, -offset), min(len(window), len(x) - offset)
        # PyCBC multiplies the float32 data in place by the float64 window: product in float64, then rounded
        sl = slice(i1 + offset, i2 + offset)
        x[sl] = (x[sl].astype(np.float64) * xp.asarray(window[i1:i2])).astype(x.dtype)
    return x


def segment_slices(n, sr, start_gps, seg_len, start_pad, end_pad, trig_start, trig_end):
    """pycbc.strain.StrainSegments index arithmetic (no filter_inj_only / zero padding)."""
    end_gps = start_gps + n / sr
    if not trig_start:
        trig_start = int(start_gps) + start_pad
    if not trig_end:
        trig_end = int(end_gps) - end_pad
    assert trig_start >= int(start_gps) + start_pad and trig_end <= int(end_gps) - end_pad
    seg_width = seg_len - start_pad - end_pad
    analyzable = trig_end - trig_start
    data_start = (trig_start - start_pad) - int(start_gps)
    data_end = trig_end + end_pad - int(start_gps)
    data_dur = data_end - data_start
    data_start, data_end = data_start * sr, data_end * sr
    num_segs = int(np.ceil(float(analyzable) / float(seg_width)))
    seg_offset = int(np.ceil(analyzable / float(num_segs)))
    segs, anas = [], []
    for k in range(num_segs - 1):
        s0 = int(data_start + (k * seg_offset) * sr)
        segs.append((s0, int(s0 + seg_len * sr)))
        a0 = int(start_pad * sr)
        anas.append((a0, int(a0 + seg_offset * sr)))
    s1 = int(data_end)
    segs.append((int(s1 - seg_len * sr), s1))
    remaining = data_dur - ((num_segs - 1) * seg_offset + start_pad)
    anas.append((int((seg_len - remaining) * sr), int((seg_len - end_pad) * sr)))
    ts_i = (trig_start - int(start_gps)) * sr
    te_i = (trig_end - int(start_gps)) * sr
    out = []
    for (s0, s1), (a0, a1) in zip(segs, anas):
        if ts_i > a0 + s0:
            a0 += ts_i - (a0 + s0)
        if te_i < a1 + s0:
            a1 -= (a1 + s0) - te_i
        if a0 < a1:
            assert s0 >= 0 and s1 <= n, 'segment outside the data needs zero padding (not supported)'
            out.append(((s0, s1), (a0, a1)))
    return out


def overlapping_psd_spans(n, seg_len, seg_stride, num_segments):
    """pycbc.psd.generate_overlapping_psds: (start, end) sample spans of the PSD measurements."""
    psd_len = (num_segments - 1) * seg_stride + seg_len
    assert n >= psd_len
    if n == psd_len:
        return [(0, n)]
    m = int(2 * (n - 1) / psd_len)
    stride = int((n - psd_len) / m)
    return [(n - psd_len, n) if i == m - 1 else (stride * i, psd_len + stride * i) for i in range(m)]


def injection_signal(opt, epoch, n, sr):
    """Injections as PyCBC adds them (InjectionSet.apply onto float64 zeros at the resampled rate); CPU."""
    from pycbc.inject import InjectionSet
    from pycbc.types import TimeSeries
    import lal
    ts = TimeSeries(np.zeros(n), delta_t=1.0 / sr, epoch=lal.LIGOTimeGPS(epoch))
    InjectionSet(opt.injection_file).apply(ts, opt.channel_name.split(':')[0], distance_scale=opt.injection_scale_factor,
                                           injection_sample_rate=opt.injection_sample_rate)
    return ts.numpy()


class _Laps:
    def __init__(self, xp):
        self.xp, self.T, self.t = xp, {}, time.perf_counter()

    def __call__(self, k):
        if self.xp is not np:
            self.xp.cuda.get_current_stream().synchronize()
        self.T[k] = round(time.perf_counter() - self.t, 4); self.t = time.perf_counter()


def options(argv):
    opt, _ = parser().parse_known_args(argv)
    bad = [k for k, v in vars(opt).items() if k.startswith('unsupported_') and v]
    assert not bad, f'gpu_prep does not implement {bad}'
    assert opt.psd_estimation in ('median', 'mean') and opt.gating_method == 'taper' and opt.hdf_store
    return opt


def cpu_stage1(argv):
    """CPU: read the store and apply the first (float64) highpass. Returns (x float64 at the store rate, timing)."""
    import h5py
    L = _Laps(np)
    opt = options(argv)
    pad = opt.pad_data
    t0 = opt.gps_start_time - pad
    with h5py.File(opt.hdf_store, 'r') as f:
        g = f[opt.channel_name]
        starts, ends = g['segments']['start'][:], g['segments']['end'][:]
        diff = t0 - starts
        loc = np.where(diff >= 0)[0]
        sidx = loc[diff[loc].argmin()]
        assert ends[sidx] >= opt.gps_end_time + pad
        d = g[str(sidx)]
        sr0 = len(d) / (ends[sidx] - starts[sidx])
        x = d[int((t0 - starts[sidx]) * sr0):int((opt.gps_end_time + pad - starts[sidx]) * sr0)]
    L('read')
    x = highpass_cpu(x, sr0, t0, opt.strain_high_pass)
    L('highpass1_cpu')
    return x, L.T


def n_resampled(argv):
    o = options(argv)
    return int(round((o.gps_end_time - o.gps_start_time + 2 * o.pad_data) * o.sample_rate))


def injection_runs(argv):
    """CPU: the injection signal for one dataset as (start index, values) runs of nonzero samples (small to pickle)."""
    t = time.perf_counter()
    o = options(argv)
    x = injection_signal(o, o.gps_start_time - o.pad_data, n_resampled(argv), o.sample_rate)
    nz = np.flatnonzero(x)
    if len(nz) == 0:
        return [], time.perf_counter() - t
    cut = np.flatnonzero(np.diff(nz) > 1)
    lo, hi = np.r_[nz[0], nz[cut + 1]], np.r_[nz[cut], nz[-1]] + 1
    return [(int(a), x[a:b].copy()) for a, b in zip(lo, hi)], time.perf_counter() - t


def injection_dense(runs, n):
    x = np.zeros(n)
    for a, v in runs:
        x[a:a + len(v)] = v
    return x


def gpu_stage1(xp, x, argv, inj=None):
    """GPU: resample, add injections (dense float64 host array), x DYN_RANGE_FAC to float32, autogating.
    Returns (gated float32 strain on the device, gates, timing)."""
    L = _Laps(xp)
    opt = options(argv)
    t0 = opt.gps_start_time - opt.pad_data
    sr = opt.sample_rate
    sr0 = len(x) / (opt.gps_end_time - opt.gps_start_time + 2 * opt.pad_data)
    factor = int(round(sr0 / sr))
    xg = xp.asarray(x)
    if factor != 1:
        xg = resample_ldas(xp, xg, factor)
    L('resample')
    if opt.injection_file:
        if inj is None:
            inj = injection_signal(opt, t0, len(xg), sr)
        xg = xg + xp.asarray(inj)
        L('injections')
    s = (xg * DYN_RANGE_FAC).astype(np.float32)
    del xg
    gates = []
    if opt.autogating_threshold is not None:
        for _ in range(opt.autogating_max_iterations):
            gt = detect_loud_glitches(xp, s, sr, t0, opt.autogating_threshold, opt.autogating_cluster,
                                      opt.strain_high_pass, opt.pad_data + opt.autogating_pad)
            gp = [(g_, opt.autogating_width, opt.autogating_taper) for g_ in gt]
            gates += gp
            gate_taper(xp, s, sr, t0, gp)
            if not gt:
                break
    L('autogating')
    return s, [(float(g), w, tp) for g, w, tp in gates], L.T


def cpu_stage2(h, argv):
    """CPU: the second (float32) highpass, after gating. h: host float32 array. Returns (h, timing)."""
    L = _Laps(np)
    opt = options(argv)
    h = highpass_cpu(h, opt.sample_rate, opt.gps_start_time - opt.pad_data, opt.strain_high_pass)
    L('highpass2_cpu')
    return h, L.T


def gpu_stage2(xp, h, argv):
    """GPU: remove padding, segment FFTs, overlapping PSDs, interpolation, truncation, D and IP."""
    L = _Laps(xp)
    opt = options(argv)
    sr, pad = opt.sample_rate, opt.pad_data
    s = xp.asarray(h)
    if pad:
        s = s[int(pad * sr):int(len(s) - sr * pad)]
    n = len(s)
    gps_start = opt.gps_start_time
    seg = segment_slices(n, sr, gps_start, opt.segment_length, opt.segment_start_pad, opt.segment_end_pad,
                         opt.trig_start_time, opt.trig_end_time)
    tlen = int(opt.segment_length * sr)
    flen = tlen // 2 + 1
    df = 1.0 / opt.segment_length
    idx = xp.asarray(np.array([a for (a, _), _ in seg]))[:, None] + xp.arange(tlen)[None, :]
    stilde = (xp.fft.rfft(s[idx], axis=1).astype(np.complex64) * np.float32(1.0 / sr))
    del idx
    L('segment_fft')
    pl, ps = int(opt.psd_segment_length * sr), int(opt.psd_segment_stride * sr)
    spans = overlapping_psd_spans(n, pl, ps, opt.psd_num_segments)
    parts = xp.stack([s[a:b] for a, b in spans])
    psd, pdf = welch(xp, parts, int(opt.psd_segment_length * sr + 0.5), int(opt.psd_segment_stride * sr + 0.5),
                     1.0 / sr, num_segments=opt.psd_num_segments, avg=opt.psd_estimation)
    del parts
    if pdf != df:
        psd = interpolate(xp, psd, pdf, df, flen)
    L('psd_welch')
    if opt.psd_inverse_length:
        psd = inverse_spectrum_truncation(xp, psd, df, int(opt.psd_inverse_length * sr), opt.low_frequency_cutoff,
                                          opt.invpsd_trunc_method)
    psd = psd.astype(np.float32)
    L('psd_truncation')
    # associate_psds_to_segments: largest overlap, first wins ties
    pick = []
    for (a, b), _ in seg:
        best, ov = None, 0
        for j, (c, e) in enumerate(spans):
            o = min(b, e) - max(a, c)
            if o > 0 and o > ov:
                best, ov = j, o
        pick.append(best)
    P = psd[xp.asarray(np.array(pick))].astype(np.float64)
    kmin = int(opt.low_frequency_cutoff / df)
    ip = xp.where(P > 0, 1.0 / xp.where(P > 0, P, 1.0), 0.0)
    ip[:, :kmin] = 0
    D = (stilde * ip).astype(np.complex64)
    L('output')
    return dict(D=D, IP=ip.astype(np.float32), flen=flen, tlen=tlen, df=df, sr=float(sr), gps_start=float(gps_start),
                analyze=[a for _, a in seg], seg_start=[s0 for (s0, _), _ in seg], timing=L.T)


def prep(argv, xp, inj=None):
    """Condition one dataset, all stages in this process. argv: pycbc_inspiral options. inj: optional dense
    injection signal (injection_dense(injection_runs(argv), n)); made here when --injection-file is given and inj is None."""
    x, T = cpu_stage1(argv)
    s, gates, T1 = gpu_stage1(xp, x, argv, inj)
    t = time.perf_counter()
    h = s.get() if xp is not np else s
    del s
    h, T2 = cpu_stage2(h, argv)
    T2['highpass2_cpu'] = round(time.perf_counter() - t, 4)       # includes the device-to-host copy
    out = gpu_stage2(xp, h, argv)
    out['gates'] = gates
    out['timing'] = {**T, **T1, **T2, **out['timing']}
    return out

def cache_pycbc_fftw_plans():
    """For the CPU (PyCBC) path: PyCBC's FFTW backend makes and destroys a plan on every fft call (fftw.py fft/ifft),
    7 s of an 8.6 ks piece in resample_to_delta_t's chunked FFTs alone. Keep each plan, keyed by everything the plan
    depends on (size, dtypes, direction, measure level, alignment, threads, in-place); FFTW's new-array execute
    reuses it on other arrays of the same alignment class, with bit-identical results."""
    import functools
    from pycbc.fft import fftw
    if getattr(fftw.plan, 'cached', False):
        return
    make = fftw.plan
    keep = lambda p: None

    @functools.lru_cache(maxsize=None)
    def plan(size, idtype, odtype, direction, mlvl, aligned, nthreads, inplace):
        return make(size, idtype, odtype, direction, mlvl, aligned, nthreads, inplace)[0], keep
    plan.cached = True
    fftw.plan = plan
