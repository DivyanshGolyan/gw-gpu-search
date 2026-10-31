"""Opt-in GPU timeline profile of one tile of fp_filter (torch.profiler, CPU + CUDA activity).

Set GW_PROF=GROUP:BANK:TILE[,GROUP:BANK:TILE...] (0-based; GROUP counts the groups one --queue process has
filtered) to profile those tiles' segment loops; GW_PROF_OUT=DIR also keeps the chrome trace. finish() returns a summary for timing.json:
how much of the window the GPU was busy, GPU time by kernel family and by kernel, and the idle gaps sorted by
the host operation that was running when the GPU went idle. Off (returns None) without GW_PROF."""
import json, os, time
from collections import defaultdict


def gpu_profile(group, bank, tile):
    want = os.environ.get('GW_PROF')
    if not want or (group, bank, tile) not in {tuple(int(x) for x in w.split(':')) for w in want.split(',')}:
        return None
    return _Prof(f'trace_{group}_{bank}_{tile}.json')


class _Prof:
    def __init__(self, fname):
        self.fname = fname
        import torch
        from torch.profiler import profile, ProfilerActivity
        torch.cuda.synchronize()
        self.p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
        import threading
        self.tid = threading.get_native_id()
        self.p.start(); self.t0 = time.perf_counter()

    def finish(self, sync):
        sync(); wall = time.perf_counter() - self.t0
        self.p.stop()
        d = os.environ.get('GW_PROF_OUT') or '.'
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, self.fname)
        self.p.export_chrome_trace(path)
        s = summarize(json.load(open(path)), self.tid)
        s['wall_s'] = wall
        if not os.environ.get('GW_PROF_OUT'):
            os.remove(path)
        print(f'PROFILE {self.fname} ' + json.dumps(s), flush=True)
        return s


GPU_CATS = ('kernel', 'gpu_memcpy', 'gpu_memset')
HOST_CATS = ('cpu_op', 'cuda_runtime', 'user_annotation', 'python_function')


def family(name):
    n = name.lower()
    for key, fam in (('fft', 'cuFFT'), ('gemm_absmax', 'GEMM+peak (Triton)'), ('chisq', 'exact stage (cupy)'),
                     ('regen', 'exact stage (cupy)'), ('slot', 'exact stage (cupy)'), ('sg_tiles', 'sg_chisq (cupy)'),
                     ('blocksum', 'chisq edges (cupy)'), ('edges', 'chisq edges (cupy)'), ('memcpy', 'memcpy'),
                     ('memset', 'memset'), ('reduce', 'torch reduce'), ('nonzero', 'torch select/index'),
                     ('index', 'torch select/index'), ('gather', 'torch select/index'), ('scatter', 'torch select/index'),
                     ('sort', 'torch sort/scan'), ('scan', 'torch sort/scan'), ('elementwise', 'torch elementwise'),
                     ('fill', 'torch elementwise'), ('copy', 'torch elementwise')):
        if key in n:
            return fam
    return 'other'


def summarize(trace, tid=None):
    """Chrome-trace dict -> summary (times in seconds). tid: the filtering thread; gaps are attributed to its host ops."""
    ev = [e for e in trace.get('traceEvents', []) if e.get('ph') == 'X' and 'dur' in e]
    gpu = sorted((float(e['ts']), float(e['ts']) + float(e['dur']), e['name']) for e in ev if e.get('cat') in GPU_CATS)
    host = [e for e in ev if e.get('cat') in HOST_CATS]
    if tid is not None and any(e.get('tid') == tid for e in host):
        host = [e for e in host if e.get('tid') == tid]
    host = sorted((float(e['ts']), float(e['ts']) + float(e['dur']), e['name']) for e in host)
    if not gpu:
        return dict(n_gpu_events=0)
    t0 = min(gpu[0][0], host[0][0] if host else gpu[0][0]); t1 = max(max(g[1] for g in gpu), max((h[1] for h in host), default=0))
    busy, cur_s, cur_e, gaps = 0.0, gpu[0][0], gpu[0][1], []
    for s, e, _ in gpu[1:]:
        if s > cur_e:
            busy += cur_e - cur_s; gaps.append((cur_e, s)); cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    by_fam, by_name, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    for s, e, n in gpu:
        by_fam[family(n)] += e - s; by_name[n[:90]] += e - s; cnt[n[:90]] += 1
    # attribute each idle gap to the innermost host op running at its start (latest start among ops covering it)
    import bisect
    starts = [h[0] for h in host]
    gap_by_op, hist = defaultdict(float), defaultdict(float)
    for gs, ge in gaps:
        L = ge - gs
        hist['<0.1ms' if L < 100 else '0.1-1ms' if L < 1000 else '1-10ms' if L < 10000 else '>10ms'] += L
        i = bisect.bisect_right(starts, gs) - 1; best = None
        while i >= 0 and i >= bisect.bisect_right(starts, gs) - 400:
            if host[i][1] >= gs:
                best = host[i][2]; break
            i -= 1
        gap_by_op[best or '(no host op)'] += L
    us = 1e-6
    top = lambda d, k: [(n, round(v * us, 4)) for n, v in sorted(d.items(), key=lambda x: -x[1])[:k]]
    return dict(window_s=round((t1 - t0) * us, 4), gpu_busy_s=round(busy * us, 4), gpu_busy_frac=round(busy / (t1 - t0), 4),
                n_gpu_events=len(gpu), n_gaps=len(gaps), gap_hist_s={k: round(v * us, 4) for k, v in hist.items()},
                by_family_s=top(by_fam, 20), top_kernels_s=[(n, v, cnt[n]) for n, v in top(by_name, 15)],
                idle_by_host_op_s=top(gap_by_op, 12))
