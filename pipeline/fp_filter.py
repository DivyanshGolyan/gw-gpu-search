#!/usr/bin/env python
"""Matched filter for one template bank against many (detector, GPS chunk) datasets, by template compression.

Drop-in for batched_filter.py / multi_filter.py: same PyCBC data conditioning (multi_filter's prep_one), same
peaks (max |SNR| per cluster window above threshold), same output fields, so coinc_rank.py runs unchanged.

What is different is how the SNR time series are made. Templates are sorted by chirp time and cut into tiles of
--tile. Per tile, once per run:
  - the whitened, unit-normalised templates are compressed with an SVD to r basis filters (r ~ 0.03-0.06 x tile
    on the slice-1 bank) such that every template keeps >= 1 - eps of its power;
Per tile and data segment:
  - only the r basis filters go through an inverse FFT (basis series, written as fp16);
  - one tensor-core GEMM (Triton kernel, fused |z|^2 and max per 128 samples) rebuilds all templates' SNR, so the
    tile x time SNR array never reaches GPU memory;
  - windows whose rebuilt SNR is above threshold x (1 - --margin) are candidates. For each candidate the exact
    complex SNR at the chosen sample and the power chi-squared come from one pass of the chi-squared kernel over
    the exact template (chisq_kernel.bin_power_sums with want_z); the exact SNR decides the final threshold cut.
So reported snr, coa_phase and chisq are exact; the compressed filter only chooses which windows to look at and
the sample within the window.

Templates: --templates interp (default) evaluates IMR waveforms on a coarse grid on the CPU and interpolates on
the GPU (imr_interp.py, as multi_filter.py); --templates pycbc uses PyCBC's FilterBank (CPU), as batched_filter.py.

Usage:
  fp_filter.py --datasets datasets.txt --bank BANK.hdf --approximant IMRPhenomD --argv inspiral_argv.txt \\
      --out OUTDIR [--tile 1024] [--eps 1e-3] [--margin 0.04] -- [pycbc options]
datasets.txt: one "IFO GPS_START GPS_END STORE.hdf" per line. Writes OUTDIR/peaks/IFO-START-END.hdf per dataset
(batched_filter.py's fields) and OUTDIR/timing.json. Run with OMP_NUM_THREADS=1.
"""
import argparse, json, os, shlex, sys, time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from types import SimpleNamespace
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multi_filter import pycbc_parser, setopt, dataset_argv, edges_per_template
from chisq_edges import edges_blocked, edges_fused

CH = 1 << 16       # frequency chunk for Gram and basis products
RB = 64            # basis rows per inverse-FFT chunk
QB = 64            # template rows per chi-squared q buffer


def _put(f, name, x):
    """Peak-file dataset, gzip + shuffle (lossless, about 3.6x smaller; h5py readers need no change)."""
    x = np.asarray(x)
    f.create_dataset(name, data=x, **(dict(compression='gzip', compression_opts=4, shuffle=True) if x.size > 1000 else {}))


def prep_one(job):
    """Worker process: condition one dataset with PyCBC (as multi_filter.prep_one, but --injection-file is honoured),
    save the over-whitened segments and 1/psd to npz. With a cache dir, a finished result is reused: the key is the
    hash of the dataset's full argv, and files are written under a temporary name and renamed into place."""
    tag, argv, outdir, cache, wait_s = job
    import hashlib
    key = hashlib.sha1(' '.join(argv).encode()).hexdigest()[:16]
    if cache:
        cj = os.path.join(cache, f'{tag}-{key}.json')
        t_wait = time.time() + wait_s
        while wait_s and not os.path.exists(cj) and time.time() < t_wait:
            time.sleep(5)                      # another process (--prep-only) is conditioning this dataset
        if os.path.exists(cj):
            meta = json.load(open(cj)); meta['prep_s'] = 0.0; meta['cached'] = True
            return meta
    t0 = time.perf_counter()
    import pycbc, pycbc.strain, pycbc.psd, pycbc.fft
    from pycbc import DYN_RANGE_FAC
    opt, _ = pycbc_parser().parse_known_args(argv)
    if not getattr(opt, 'injection_file', None):
        opt.injection_file = None
    pycbc.fft.from_cli(opt)
    from gpu_prep import cache_pycbc_fftw_plans
    cache_pycbc_fftw_plans()                   # PyCBC re-plans every FFT; keep the plans (output within PyCBC's own spread)
    gw = pycbc.strain.from_cli(opt, dyn_range_fac=DYN_RANGE_FAC)
    ss = pycbc.strain.StrainSegments.from_cli(opt, gw)
    segs = ss.fourier_segments()
    flen, tlen, df = ss.freq_len, ss.time_len, ss.delta_f
    flow = opt.low_frequency_cutoff
    pycbc.psd.associate_psds_to_segments(opt, segs, gw, flen, df, flow, dyn_range_factor=DYN_RANGE_FAC, precision='single')
    kmin = int(flow / df)
    ipsd = np.stack([seg.psd.numpy() for seg in segs]).astype(np.float64)
    with np.errstate(divide='ignore'):
        ipsd = np.where(ipsd > 0, 1.0 / ipsd, 0.0)
    ipsd[:, :kmin] = 0
    D = (np.stack([seg.numpy() for seg in segs]) * ipsd).astype(np.complex64)
    fn = os.path.join(cache or outdir, f'{tag}-{key}.npz')
    np.savez(fn + '.part.npz', D=D, IP=ipsd.astype(np.float32)); os.replace(fn + '.part.npz', fn)
    gates = [tuple(map(float, g)) for kind in (gw.gating_info or {}).values() for g in kind]
    meta = dict(tag=tag, file=fn, flen=flen, tlen=tlen, df=df, sr=float(gw.sample_rate), gps_start=float(opt.gps_start_time),
                analyze=[(seg.analyze.start, seg.analyze.stop) for seg in segs], seg_start=[seg.seg_slice.start for seg in segs],
                gates=gates, prep_s=time.perf_counter() - t0, injection_file=opt.injection_file, cached=False)
    if cache:
        json.dump(meta, open(cj + '.part', 'w')); os.replace(cj + '.part', cj)
    return meta


def main():
    split = sys.argv.index('--') if '--' in sys.argv else len(sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets'); ap.add_argument('--bank', required=True)
    ap.add_argument('--approximant', required=True); ap.add_argument('--argv', required=True)
    ap.add_argument('--out')
    ap.add_argument('--queue', help='DIR: one process for many groups: filter DIR/*.job files ("DATASETS OUTDIR") in name order '
                    'until DIR/END exists; bank tables and coarse templates are made once, and the next job\'s CPU prep '
                    'runs while the GPU filters this one (needs --prep gpu)')
    ap.add_argument('--pre-consolidate', action='store_true',
                    help='with --consolidate: drop, on the GPU and per tile and segment, the rebuilt peaks that consolidate '
                         'would drop (same cut, gate veto and keep-loudest ranking), before they reach the host')
    ap.add_argument('--tile', type=int, default=1024); ap.add_argument('--eps', type=float, default=1e-3)
    ap.add_argument('--margin', type=float, default=0.04, help='candidate cut is snr-threshold x (1 - margin) on the rebuilt SNR')
    ap.add_argument('--templates', choices=['interp', 'pycbc'], default='interp')
    ap.add_argument('--prep-procs', type=int, default=8); ap.add_argument('--gen-threads', type=int, default=8)
    ap.add_argument('--n-chirp', type=int, default=6000); ap.add_argument('--n-log', type=int, default=1500)
    ap.add_argument('--max-templates', type=int, help='first N templates in bank order (smoke tests)')
    ap.add_argument('--prep-cache', help='directory for conditioned data, reused across runs (keyed by the full argv)')
    ap.add_argument('--wait-prep', type=float, default=0, help='seconds to wait for a dataset to appear in --prep-cache before conditioning it here')
    ap.add_argument('--gemm-cfg', help='BM,BK,warps,stages for the GEMM kernel; skips the tuning sweep')
    ap.add_argument('--prep-only', action='store_true', help='condition the datasets into --prep-cache and stop (CPU only)')
    ap.add_argument('--prep', choices=['cpu', 'gpu'], default='cpu',
                    help='gpu: condition with gpu_prep.py (PyCBC step for step on the GPU; store read, the two IIR highpasses '
                         'and injections run on --prep-procs CPU workers, overlapped with the GPU stages; no --prep-cache)')
    ap.add_argument('--edges', choices=['blocked', 'full', 'fused'], default='blocked', help='chi-squared bin edge algorithm')
    ap.add_argument('--exact-snr', type=float, default=0,
                    help='exact SNR, chi-squared and sg_chisq only where the rebuilt SNR >= this x (1 - margin); other peaks keep the '
                         'rebuilt SNR and phase with chisq = dof = 0 (0 = exact everywhere)')
    ap.add_argument('--consolidate', action='store_true',
                    help="pycbc_inspiral's consolidate_events per piece, with the whole bank as one job: drop peaks with newsnr "
                         "below --newsnr-threshold, then keep the --keep-loudest-num loudest by newsnr in each "
                         "--keep-loudest-interval (applied after every tile, so memory stays bounded)")
    ap.add_argument('--consolidate-exempt-snr', type=float, default=None,
                    help='with --consolidate: peaks with snr >= this are kept unconditionally and do not count toward the cap')
    ap.add_argument('--exact-stage', choices=['dense', 'regen'], default='dense',
                    help='regen: exact z and chi-squared sums from the coarse interp nodes (exact_stage.py, no reads of H; '
                         'needs --templates interp), checked against the dense path on the first segment of every tile')
    ap.add_argument('--fuse-q', action='store_true', help='the chi-squared kernel forms conj(H) D itself (no q buffer)')
    ap.add_argument('--sg-kernel', action='store_true', help='with --sg-batch: sg_chisq in one float32 CUDA launch per tile list (sg_kernel.py)')
    ap.add_argument('--sg-batch', action='store_true', help='sg_chisq for all qualifying peaks of a segment in one batched pass')
    ap.add_argument('--legacy-qzero', action='store_true', help='re-zero the unused half of the q buffer per group (old behaviour, for A/B)')
    ap.add_argument('--chunk', help='LEN,HIST,FUT seconds: filter overlap-save chunks of every segment instead of whole '
                    'segments (window_filter.py); templates must be shorter than HIST')
    ap.add_argument('--chunk2', help='LEN,HIST,FUT[,FMAX] as --chunk, for --bank2 only (after --bank ran on whole segments); '
                    'FMAX (Hz, sample rate / 4 or / 8 ...) band-limits the chunks and samples SNR at 2 x FMAX')
    ap.add_argument('--gate', help='hdf (ifo, time, mchirp): with --chunk, run a tile on a chunk only if a trigger of that '
                    'detector lies within --gate-window of it with mchirp inside the tile range widened by --gate-mc-ratio, '
                    'and keep only peaks within --gate-window of such a trigger')
    ap.add_argument('--gate-mc-ratio', type=float, default=1.25); ap.add_argument('--gate-window', type=float, default=1.0)
    ap.add_argument('--bank2', help='second bank filtered on the same conditioned data after --bank; peaks in OUT/peaks_b2')
    ap.add_argument('--approximant2', default='IMRPhenomD', help="approximant of --bank2; 'SEOBNRv5EHM_ecc' uses ecc/ecc_templates.py")
    ap.add_argument('--eps2', type=float, help='SVD worst-lost-power tolerance for --bank2 (default --eps)')
    ap.add_argument('--history', help='append this run to an analysis-history log (JSON lines, provenance.py)')
    a = ap.parse_args(sys.argv[1:split])
    assert not a.gate or a.chunk, '--gate needs --chunk'
    base = shlex.split(open(a.argv).read()) + sys.argv[split + 1:]
    base = setopt(setopt(base, '--bank-file', a.bank), '--approximant', a.approximant)
    assert bool(a.queue) != bool(a.datasets and a.out), 'give --datasets and --out, or --queue'
    assert not a.queue or a.prep == 'gpu', '--queue needs --prep gpu'
    assert not a.pre_consolidate or (a.consolidate and not a.gate), '--pre-consolidate needs --consolidate and no --gate'
    if a.prep_cache:
        os.makedirs(a.prep_cache, exist_ok=True)
    def ds_argv(i, s, e, st, *trig_start):
        # optional 5th column: --trig-start-time as GPS seconds after the piece start (keeps overlapping pieces seamless
        # when --segment-start-pad is shorter than the piece overlap)
        av = dataset_argv(base, i, int(s), int(e), st)
        return setopt(av, '--trig-start-time', str(int(s) + int(trig_start[0]))) if trig_start else av
    def read_jobs(ds_file, out):
        ds = [l.split() for l in open(ds_file) if l.strip() and not l.startswith('#')]
        return [(f'{d[0]}-{d[1]}-{d[2]}', ds_argv(*d), os.path.join(out, 'prep'), a.prep_cache, a.wait_prep) for d in ds]
    pools = []
    def submit_prep(ds_file):
        """--prep gpu: start the CPU stages of one group's datasets on the worker pool (forked once, before CUDA)."""
        import gpu_prep
        jobs = read_jobs(ds_file, '')
        if not pools:
            pools.append(Pool(a.prep_procs if a.queue else min(a.prep_procs, len(jobs))))
        metas, cpu1, injr = [], [], []
        for tag, av, *_ in jobs:
            metas.append(dict(tag=tag, argv=av, cached=False))
            cpu1.append(pools[0].apply_async(gpu_prep.cpu_stage1, (av,)))
            injr.append(pools[0].apply_async(gpu_prep.injection_runs, (av,)) if gpu_prep.options(av).injection_file else None)
        return metas, cpu1, injr
    prep_thread, write_pool = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    memo = {}       # --queue: what does not depend on the data, kept across groups (bank tables, bins, coarse templates)
    def cached(key, fn):
        if key not in memo:
            memo[key] = fn()
        return memo[key]
    seen = set()
    def next_job(wait):
        """--queue: the next DIR/*.job ('DATASETS OUTDIR') in name order; None once DIR/END exists and every job is taken."""
        while True:
            end = os.path.exists(os.path.join(a.queue, 'END'))
            js = sorted(f for f in os.listdir(a.queue) if f.endswith('.job') and f not in seen)
            if js:
                seen.add(js[0])
                return tuple(open(os.path.join(a.queue, js[0])).read().split()[:2])
            if end or not wait:
                return None
            time.sleep(2)

    def run_group(pend):
        """Filter a.datasets into a.out. pend: that group's CPU prep stages, already submitted (--prep gpu). With --queue,
        the next job's CPU stages are submitted once this group's data is on the GPU; returns (job, its stages) or None."""
        nxt_job = None
        wfuts = []
        T = dict(prep_s=0.0, load_s=0.0, templates_s=0.0, norm_s=0.0, svd_s=0.0, basis_fft_s=0.0, gemm_s=0.0,
                 refine_chisq_s=0.0, timing='v9: basis_fft_s, gemm_s, rebuilt_z_s are GPU event times; refine_chisq_s host wall of the back', n_candidates=0, n_chisq_points=0, rf_q_s=0.0, rf_edges_s=0.0, rf_kernel_s=0.0, rebuilt_z_s=0.0, n_exact_candidates=0, n_rebuilt_peaks=0)
        t_start = time.perf_counter()
        for d in ('peaks', 'prep'):
            os.makedirs(os.path.join(a.out, d), exist_ok=True)

        # ---- condition all datasets in parallel on the CPU (before CUDA, so the pool can fork)
        jobs = read_jobs(a.datasets, a.out)
        t0 = time.perf_counter()
        if a.prep == 'gpu':
            # CPU stages run on workers forked before CUDA (submitted by the caller, for --queue during the previous
            # group); the GPU stages run after the CUDA imports below
            assert not a.prep_only, '--prep-only is for the CPU path'
            import gpu_prep
            t_prep0 = t0
            pend = pend or submit_prep(a.datasets)
            metas, cpu1, injr = pend[:3]
            prep_pool = pools[0]
        else:
            with Pool(min(a.prep_procs, len(jobs))) as pool:
                metas = pool.map(prep_one, jobs, chunksize=1)
            T['prep_s'] = time.perf_counter() - t0
            T['prep_cpu_s_each'] = [m['prep_s'] for m in metas]
            T['prep_cached'] = sum(bool(m.get('cached')) for m in metas)
        if a.prep_only:
            assert a.prep_cache, '--prep-only needs --prep-cache'
            json.dump(T, open(os.path.join(a.out, 'prep_timing.json'), 'w'), indent=1)
            print(json.dumps(T)); return

        import cupy as cp
        import torch
        if a.queue and not memo.get('torch_alloc'):
            # one caching allocator for both libraries: with --queue the next group's prep (cupy) runs beside the
            # filter (torch), and two pools each keeping freed blocks ran a 24 GB card out of memory
            def torch_alloc(size):
                buf = torch.empty(max(size, 1), dtype=torch.uint8, device='cuda')
                return cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(buf.data_ptr(), size, buf), 0)
            cp.cuda.set_allocator(torch_alloc); memo['torch_alloc'] = True
        from fp_gemm import fused_peaks, pick_cfg, BN
        import pycbc, pycbc.waveform, pycbc.conversions as cv
        from pycbc.types import complex64
        from pycbc.vetoes.chisq import SingleDetPowerChisq
        from pycbc.events.ranking import newsnr
        from chisq_kernel import bin_power_sums
        from sg_kernel import sg_chisq_points
        from exact_stage import ExactStage, TemplateTile, Coarse, Grid, SegmentData, PeakRequest
        assert a.exact_stage == 'dense' or a.templates == 'interp', '--exact-stage regen needs --templates interp'
        from imr_interp import coarse_grid, coarse_generate, FineInterpolator
        dev = 'cuda'
        sync = torch.cuda.synchronize
        from seg_pipe import BackStream, EventClock, SegmentPipe
        bstream = cached(('bstream',), lambda: BackStream(torch, cp))   # one back stream per process (seg_pipe.py)
        bsync = bstream.synchronize
        opt, _ = pycbc_parser().parse_known_args(base)
        chisq_bins = opt.chisq_bins.strip('"\'')
        do_chisq = chisq_bins not in ('', '0')
        assert a.exact_stage == 'dense' or do_chisq, '--exact-stage regen needs chi-squared bins'
        thr = opt.snr_threshold
        sgp = argparse.ArgumentParser()
        sgp.add_argument('--sgchisq-snr-threshold', type=float); sgp.add_argument('--sgchisq-locations', nargs='+')
        sgo, _ = sgp.parse_known_args(base)
        klp = argparse.ArgumentParser()
        klp.add_argument('--newsnr-threshold', type=float); klp.add_argument('--keep-loudest-interval', type=float)
        klp.add_argument('--keep-loudest-num', type=int); klp.add_argument('--keep-loudest-stat')
        klo, _ = klp.parse_known_args(base)
        do_sg = do_chisq and sgo.sgchisq_snr_threshold is not None and bool(sgo.sgchisq_locations)

        # ---- data on the GPU: over-whitened segments D = stilde / psd and 1/psd, every dataset
        t0 = time.perf_counter()
        if a.prep == 'gpu':
            # stage 1 (resample, injections, gating) per dataset as its CPU stage arrives; second highpasses run on the
            # workers meanwhile; then stage 2 (segment FFTs, PSDs, D and IP) on the device, handed to torch without a copy
            def gpu_stages(metas, cpu1, injr):
                hp2 = []
                for i, m in enumerate(metas):
                    x, tc = cpu1[i].get()
                    inj = None
                    if injr[i] is not None:
                        runs, m['inj_cpu_s'] = injr[i].get()
                        inj = gpu_prep.injection_dense(runs, gpu_prep.n_resampled(m['argv']))
                    s_, m['gates'], tg = gpu_prep.gpu_stage1(cp, x, m['argv'], inj)
                    hp2.append(prep_pool.apply_async(gpu_prep.cpu_stage2, (s_.get(), m['argv'])))
                    m['prep_timing'] = {**tc, **tg}
                    del x, s_, inj
                for i, m in enumerate(metas):
                    h, tc = hp2[i].get()
                    g = gpu_prep.gpu_stage2(cp, h, m['argv'])
                    m['prep_timing'].update(tc, **g.pop('timing'))
                    m['D'] = torch.from_dlpack(g.pop('D')); m['IP'] = torch.from_dlpack(g.pop('IP'))
                    m.update(g)
                    m['prep_s'] = sum(m['prep_timing'].values())
            if len(pend) > 3:
                pend[3].result()        # --queue: done on a thread while the previous group filtered
            else:
                gpu_stages(metas, cpu1, injr)
            def prefetch():
                """--queue: the next job, if there is one yet: its CPU stages on the pool and its GPU stages on a thread."""
                nj = next_job(False)
                if nj:
                    np_ = submit_prep(nj[0])
                    return nj, np_ + (prep_thread.submit(gpu_stages, *np_),)
            if a.queue:
                nxt_job = prefetch()
            else:
                prep_pool.close(); prep_pool.join()
            cp.get_default_memory_pool().free_all_blocks(); cp.fft.config.get_plan_cache().clear()
            T['prep_s'] = time.perf_counter() - t_prep0
            T['prep_gpu_wait_s'] = time.perf_counter() - t0
            T['prep_cpu_s_each'] = [m['prep_s'] for m in metas]
            T['prep_timing'] = [m['prep_timing'] for m in metas]
            T['prep_cached'] = 0
            t0 = time.perf_counter()
        for m in metas:
            if a.prep == 'cpu':
                z = np.load(m['file'])
                m['D'] = torch.from_numpy(z['D']).to(dev); m['IP'] = torch.from_numpy(z['IP']).to(dev)
                if not a.prep_cache:
                    os.remove(m['file'])
            m['out'] = dict(tid=[], t=[], snr=[], chi=[], dof=[], ph=[], sg=[], sgc=[], sgd=[], ex=[])
        def to_chunks(spec):
            """Every segment becomes overlap-save chunks of the same over-whitened series (window_filter.py); from then on a
            chunk is a segment. spec LEN,HIST,FUT[,FMAX]: with FMAX the chunks keep bins below FMAX only, i.e. are sampled
            at 2 FMAX (same SNR for templates without power above FMAX)."""
            from window_filter import chunk_starts, valid, chunk_segment
            v = [float(x) for x in spec.split(',')]
            c_len, c_hist, c_fut = v[:3]
            for m in metas:
                sr_ = m['sr']; Lc, hist, fut = int(c_len * sr_), int(c_hist * sr_), int(c_fut * sr_)
                dec = int(round(sr_ / (2 * v[3]))) if len(v) > 3 else 1
                assert dec >= 1 and Lc % (2 * dec) == 0 and hist % dec == 0 and fut % dec == 0
                Ds, IPs, an, ss, first = [], [], [], [], []
                for s, (a0, a1) in enumerate(m['analyze']):
                    st = chunk_starts(m['tlen'], Lc, hist, fut, a0, a1); va, vb = valid(st, Lc, hist, fut, a1)
                    assert np.all(st % dec == 0) and np.all(va % dec == 0) and np.all(vb % dec == 0)
                    Dc_, IPc_ = chunk_segment(cp.from_dlpack(m['D'][s]), cp.from_dlpack(m['IP'][s]), m['tlen'], 1 / sr_, st, Lc, xp=cp)
                    nb = Lc // (2 * dec) + 1
                    first.append(len(an))
                    Ds.append(torch.from_dlpack(cp.ascontiguousarray(Dc_[:, :nb])))
                    IPs.append(torch.from_dlpack(cp.ascontiguousarray(IPc_[:, :nb])))
                    an += [(int(x - y) // dec, int(z - y) // dec) for x, z, y in zip(va, vb, st)]
                    ss += [int(m['seg_start'][s] + y) // dec for y in st]
                    assert m['seg_start'][s] % dec == 0
                m['D'] = torch.cat(Ds); m['IP'] = torch.cat(IPs); m['analyze'] = an; m['seg_start'] = ss; m['first'] = first
                m['tlen'], m['flen'], m['df'], m['sr'] = Lc // dec, Lc // (2 * dec) + 1, sr_ / Lc, sr_ / dec
                del Ds, IPs
            T.setdefault('chunk', []).append(dict(spec=spec, len_s=c_len, hist_s=c_hist, fut_s=c_fut, n_chunks=sum(len(m['analyze']) for m in metas)))
            torch.cuda.empty_cache()
            return c_hist

        def derive():
            flen, tlen, df, sr = metas[0]['flen'], metas[0]['tlen'], metas[0]['df'], metas[0]['sr']
            assert all(m['flen'] == flen and m['tlen'] == tlen and m['sr'] == sr for m in metas)
            kmin = int(flow / df)
            cw = int(opt.cluster_window * sr)
            assert cw % BN == 0, 'cluster window must be a multiple of %d samples (GEMM output block)' % BN
            IPall = torch.cat([m['IP'] for m in metas], 0)                       # (all segments, flen)
            # SVD metric: mean inverse PSD over every segment of every dataset. conj(h sw) (D / sw) = conj(h) D exactly, so the
            # metric only sets how well the basis compresses, never the SNR definition.
            sw = torch.sqrt(IPall.double().mean(0)).float()
            isw = torch.where(sw > 0, 1.0 / sw.clamp(min=1e-30), torch.zeros_like(sw))
            off = 0
            for m in metas:
                m['seg0'] = off; off += len(m['analyze'])
                m['NW'] = [(a1 - a0) // cw for a0, a1 in m['analyze']]
                m.setdefault('first', list(range(len(m['analyze']))))
            return flen, tlen, df, sr, kmin, cw, IPall, sw, isw, off

        flow = opt.low_frequency_cutoff
        c_hist = to_chunks(a.chunk) if a.chunk else None
        flen, tlen, df, sr, kmin, cw, IPall, sw, isw, nseg_all = derive()
        T['load_s'] = time.perf_counter() - t0
        T['gpu_mem_data_gib'] = sum(m['D'].numel() * 8 + m['IP'].numel() * 4 for m in metas) / 2 ** 30

        # ---- each bank in turn over the same conditioned data (a follow-up bank costs no second download or prep)
        ECC = 'SEOBNRv5EHM_ecc'
        if ECC in (a.approximant, a.approximant2):
            try:
                from ecc_templates import coarse_generate_ecc
            except ImportError:
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ecc'))
                from ecc_templates import coarse_generate_ecc
        from pycbc import DYN_RANGE_FAC
        from provenance import record, log as log_history
        opt_val = lambda k: base[base.index(k) + 1] if k in base else None
        prov = record(dict(bank=a.bank, bank2=a.bank2, datasets=a.datasets, argv_file=a.argv, gate=a.gate,
                           injection_file=opt_val('--injection-file')), argv=sys.argv + ['[resolved]'] + base,
                      dq_files=[a.datasets], dq_flags=[x for i, x in enumerate(base) if x.startswith('--autogating')
                                                       or (i and base[i - 1].startswith('--autogating'))])
        T['provenance'] = prov; prov_json = json.dumps(prov, sort_keys=True)
        log_history(a.history, 'fp_filter', 'start', a.out, prov)
        banks = [(a.bank, a.approximant, 'peaks', a.eps)] + ([(a.bank2, a.approximant2, 'peaks_b2', a.eps2 or a.eps)] if a.bank2 else [])
        T['tiles'] = []; T['n_peaks'] = {}
        for bank_i, (bank_path, appx, pdir, eps_b) in enumerate(banks):
          sfx = '' if bank_i == 0 else f'_b{bank_i + 1}'
          if bank_i == 1 and a.chunk2:
              assert not a.chunk, '--chunk2 needs bank 1 on whole segments'
              del IPall
              c_hist = to_chunks(a.chunk2)
              flen, tlen, df, sr, kmin, cw, IPall, sw, isw, nseg_all = derive()
          print(f'bank {bank_i + 1} start {time.perf_counter() - t_start:.1f}s', flush=True)
          os.makedirs(os.path.join(a.out, pdir), exist_ok=True)
          for m in metas:
              m['out'] = dict(tid=[], t=[], snr=[], chi=[], dof=[], ph=[], sg=[], sgc=[], sgd=[], ex=[]); m.pop('sig', None); m.pop('floor', None)
          # ---- bank and template generation
          bank = cached(('bank', bank_path, appx, flen, df, flow), lambda: pycbc.waveform.FilterBank(
              bank_path, flen, df, complex64, low_frequency_cutoff=flow, phase_order=opt.order, approximant=appx))
          if appx == ECC:                         # FilterBank drops the base-shape columns the generator needs
              def ecc_read():
                  with h5py.File(bank_path, 'r') as f_:
                      return {k: f_[k][:] for k in ('mass1', 'mass2', 'spin1z', 'ecc_b', 'anom_b', 'mtot_b', 'fstart_b')}
              ecc_cols = cached(('ecc_cols', bank_path), ecc_read)
          ntmp = len(bank) if a.max_templates is None else min(a.max_templates, len(bank))
          hashes = np.asarray(bank.table.template_hash)[:ntmp]
          m1, m2 = np.asarray(bank.table.mass1)[:ntmp], np.asarray(bank.table.mass2)[:ntmp]
          tau0_all = cv.tau0_from_mass1_mass2(m1, m2, flow)
          order = np.argsort(tau0_all, kind='stable')
          if c_hist is not None:
              T['chunk'][-1]['max_tau0_s'] = float(tau0_all.max())
              assert tau0_all.max() < c_hist, f'templates up to {tau0_all.max():.1f} s from {flow} Hz do not fit {c_hist} s of chunk history'
          gate_on = bool(a.gate)
          if gate_on:
              mc_all = cv.mchirp_from_mass1_mass2(m1, m2)
              with h5py.File(a.gate, 'r') as f:
                  g_ifo = f['ifo'][:].astype(str); g_t = f['time'][:]; g_mc = f['mchirp'][:]
              gw_s = int(a.gate_window * sr)
              for m in metas:
                  sel = g_ifo == m['tag'].split('-')[0]
                  o_ = np.argsort(g_t[sel]); gt_, gm_ = g_t[sel][o_], g_mc[sel][o_]
                  gs_ = np.round((gt_ - m['gps_start']) * sr).astype(np.int64)
                  m['gate'] = []
                  for s, (a0, a1) in enumerate(m['analyze']):
                      c = gs_ - m['seg_start'][s]
                      i0, i1 = np.searchsorted(c, a0 - gw_s), np.searchsorted(c, a1 + gw_s, side='right')
                      m['gate'].append((c[i0:i1], gm_[i0:i1]))
              T.update(gate_runs=0, gate_skips=0, gate_peaks_dropped=0)

          def near(samp, gs, w):
              j = np.searchsorted(gs, samp)
              return (np.abs(samp - gs[np.clip(j - 1, 0, len(gs) - 1)]) <= w) | (np.abs(samp - gs[np.clip(j, 0, len(gs) - 1)]) <= w)
          nbins_all = np.zeros(ntmp, np.int64); kmin_all = np.full(ntmp, kmin, np.int64); kmax_all = np.zeros(ntmp, np.int64)
          if do_chisq:
              nbins_all[:] = cached(('nbins', bank_path, chisq_bins, ntmp), lambda: [
                  int(SingleDetPowerChisq.parse_option(SimpleNamespace(params=bank.table[i]), chisq_bins)) for i in range(ntmp)])
          tiles = [order[b0:b0 + a.tile] for b0 in range(0, ntmp, a.tile)]
          def sg_parse():
              locs = [None] * ntmp
              for descr in sgo.sgchisq_locations:
                  region, values = descr.split(':')
                  mask = bank.table.parse_boolargs([(1, region), (0, 'else')])[0][:ntmp].astype(bool)
                  vals = [tuple(map(float, v.split('-'))) for v in values.split(',')]
                  for i in np.nonzero(mask)[0]:
                      locs[i] = vals
              return locs
          sg_locs = [None] * ntmp
          if do_sg:
              sg_locs = cached(('sg_locs', bank_path, tuple(sgo.sgchisq_locations), ntmp), sg_parse)
              T['n_sg_templates'] = int(sum(v is not None for v in sg_locs)); T['sg_chisq_s'] = 0.0; T['n_sg_chisq_points'] = 0

          def sg_chisq_one(Dc, IPc, t, fpeak, vals, k_start):
              """pycbc.vetoes.sgchisq.SingleDetSGChisq.values for one peak at sample t of a segment: reduced chi-squared of the
              over-whitened data against sine-Gaussian tiles at fpeak + offset (+-50 Hz). Same index conventions as pycbc,
              including the lower edge of each tile being bounded by the previous tile's start. None if a tile reaches above
              0.9 x Nyquist (pycbc then reports 1 for every peak of that template and segment)."""
              fstop = flen * df * 0.9
              if any(fpeak + off + 50 > fstop for _, off in vals):
                  return None
              kk = k_start; acc = 0.0
              for q, off in vals:
                  fcen = fpeak + off
                  flo = max(kk * df, fcen - 50); fhi = fcen + 50
                  kk = int(flo / df); k1 = int(fhi / df)
                  kg = int(round(flo / df))
                  k = cp.arange(kk, k1, dtype=cp.float64)
                  f = k * df
                  tau = q / 2 / np.pi / fcen
                  g = (np.pi ** 0.5 / 2 * tau) * cp.exp(-(np.pi * tau * (f - fcen)) ** 2) * (1 + cp.exp(-q ** 2 * f / fcen))
                  g = cp.where(k >= kg, g, 0.0)
                  gsig = cp.sqrt(4 * df * (g * g * IPc[kk:k1].astype(cp.float64)).sum())
                  ph = ((k.astype(cp.int64) * int(t)) % tlen).astype(cp.float64) * (2 * np.pi / tlen)
                  gs_ = (g * Dc[kk:k1].astype(cp.complex128) * cp.exp(1j * ph)).sum() * (4.0 * df) / gsig
                  acc += float(abs(gs_)) ** 2
              return acc / (2 * len(vals))
          def sg_chisq_batch(Dc, IPc, t, fpeak, vals, k_start):
              """sg_chisq_one for many peaks that share one tile list (vals): arrays t, fpeak, k_start. Returns values and a
              computed mask (False where a tile reaches above 0.9 x Nyquist)."""
              n = len(t); fstop = flen * df * 0.9
              ok = np.all([fpeak + off + 50 <= fstop for _, off in vals], axis=0)
              out = np.ones(n, np.float64)
              idx = np.nonzero(ok)[0]
              W = int(np.ceil(100 / df)) + 2
              for c0 in range(0, len(idx), 32):
                  ii = idx[c0:c0 + 32]; kk = k_start[ii].astype(np.int64); acc = cp.zeros(len(ii))
                  tt = cp.asarray(t[ii].astype(np.int64))[:, None]
                  for q, off in vals:
                      fcen = fpeak[ii] + off
                      flo = np.maximum(kk * df, fcen - 50); fhi = fcen + 50
                      kk = (flo / df).astype(np.int64); k1 = (fhi / df).astype(np.int64)
                      kg = np.round(flo / df).astype(np.int64)
                      k = cp.asarray(kk)[:, None] + cp.arange(W, dtype=cp.int64)[None, :]
                      inb = k < cp.asarray(k1)[:, None]
                      kc = cp.minimum(k, flen - 1)
                      f = k.astype(cp.float64) * df; fc = cp.asarray(fcen)[:, None]
                      tau = q / 2 / np.pi / fc
                      g = (np.pi ** 0.5 / 2 * tau) * cp.exp(-(np.pi * tau * (f - fc)) ** 2) * (1 + cp.exp(-q ** 2 * f / fc))
                      g = cp.where(inb & (k >= cp.asarray(kg)[:, None]), g, 0.0)
                      gsig = cp.sqrt(4 * df * (g * g * IPc[kc].astype(cp.float64)).sum(1))
                      ph = ((k * tt) % tlen).astype(cp.float64) * (2 * np.pi / tlen)
                      gs_ = (g * Dc[kc].astype(cp.complex128) * cp.exp(1j * ph)).sum(1) * (4.0 * df) / gsig
                      acc += cp.abs(gs_) ** 2
                  out[ii] = (acc / (2 * len(vals))).get()
              return out, ok

          gen_pool = ThreadPoolExecutor(1)
          if a.templates == 'interp':
              f_end = None if appx == ECC else bank.end_frequency(0)
              kend = int(f_end / df) if f_end and f_end < (flen - 1) * df else flen - 1
              ikey = (bank_path, appx, flow, df, kmin, kend, flen, a.n_chirp, a.n_log)
              x = cached(('x',) + ikey, lambda: coarse_grid(flow, kend * df, a.n_chirp, a.n_log))
              interp = cached(('interp',) + ikey, lambda: FineInterpolator(x, df, kmin, kend, flen, cp))
              grid = cached(('grid',) + ikey, lambda: Grid.from_interp(interp)) if a.exact_stage == 'regen' else None
              coarse_now = {}

              def cpu_part(ids):
                  if appx == ECC:
                      rows = [{k: v[int(i)] for k, v in ecc_cols.items()} for i in ids]
                      return coarse_generate_ecc(rows, x, flow, procs=a.gen_threads, distance=1.0 / DYN_RANGE_FAC)
                  return coarse_generate([bank.table[int(i)] for i in ids], x, appx, flow, threads=a.gen_threads)

              # a band-limited view (--chunk2 FMAX) ends every template with a cosine taper over the top 1/8 of the band, so
              # templates stay compact in time (a hard cut rings past the chunk history); on the coarse amplitude, so the
              # exact stage regenerates the same template
              f_top = (flen - 1) * df
              tap = np.clip(0.5 * (1 + np.cos(np.pi * (x - 0.875 * f_top) / (0.125 * f_top))), 0, 1) if bank_i == 1 and a.chunk2 and len(a.chunk2.split(',')) > 3 else None
              if tap is not None:
                  tap[x <= 0.875 * f_top] = 1.0

              def gpu_part(ids, res):
                  amp, coef, step = res
                  if tap is not None:
                      amp = amp * tap[None, :]
                  coarse_now.update(amp=amp, coef=coef)
                  T['max_phase_step_rad'] = max(T.get('max_phase_step_rad', 0.0), step)
                  H = torch.zeros((len(ids), flen), dtype=torch.complex64, device=dev)
                  for c0 in range(0, len(ids), 128):
                      H[c0:c0 + 128] = torch.from_dlpack(interp(amp[c0:c0 + 128], coef[c0:c0 + 128]))
                  kmax_all[ids] = kend
                  return H
          else:
              def cpu_part(ids):
                  with ThreadPoolExecutor(a.gen_threads) as pool:
                      hl = list(pool.map(lambda i: bank[int(i)], ids))
                  for i, h in zip(ids, hl):
                      kmin_all[i] = int(h.f_lower / df); kmax_all[i] = int(h.end_idx)
                  return np.stack([h.numpy() for h in hl])

              def gpu_part(ids, res):
                  return torch.from_numpy(res).to(dev)
          if a.templates == 'interp':     # coarse templates do not depend on the data: made once per process
              coarse_tile = lambda ti_: cached(('coarse', ikey, a.tile, ntmp, a.gen_threads, ti_), lambda: cpu_part(tiles[ti_]))
          else:                            # cpu_part also sets kmin_all and kmax_all
              coarse_tile = lambda ti_: cpu_part(tiles[ti_])
          nxt = gen_pool.submit(coarse_tile, 0)

          t0 = time.perf_counter()
          if a.gemm_cfg:
              KCFG, T['gemm_tune'] = tuple(int(x) for x in a.gemm_cfg.split(',')), {}
          else:
              KCFG, T['gemm_tune'] = pick_cfg()
          T['gemm_tune_s'] = time.perf_counter() - t0
          T['gemm_cfg'] = str(KCFG)

          def svd_tile(H):
              n = H.shape[0]
              G = torch.zeros((n, n), dtype=torch.complex64, device=dev)
              for c0 in range(kmin, flen, CH):
                  hc = H[:, c0:c0 + CH] * sw[c0:c0 + CH]
                  G += hc @ hc.conj().T
              nrm = torch.sqrt(G.diagonal().real.clamp(min=1e-30))
              lam, W = torch.linalg.eigh((G / (nrm[:, None] * nrm[None, :])).to(torch.complex128))
              lam, W = lam.flip(0).clamp(min=0), W.flip(1)
              worst = (1 - ((W.abs() ** 2) * lam[None, :]).cumsum(1)).max(0).values   # worst lost power after k+1 vectors
              r = min(int((worst > eps_b).sum().item()) + 1, n)
              Wr = W[:, :r].to(torch.complex64); s_r = torch.sqrt(lam[:r].clamp(min=1e-12 * float(lam[0]))).float()
              Bm = (Wr.conj().T / s_r[:, None]) / nrm[None, :]
              Vh = torch.zeros((r, flen), dtype=torch.complex64, device=dev)
              for c0 in range(kmin, flen, CH):
                  Vh[:, c0:c0 + CH] = Bm @ (H[:, c0:c0 + CH] * sw[c0:c0 + CH])
              A = (Wr * s_r[None, :]).conj() * nrm[:, None]                  # z = A @ IFFT(conj(Vh) D / sw)
              return r, A, Vh, float(worst[r - 1].item())

          qz = torch.zeros((RB, tlen), dtype=torch.complex64, device=dev)

          def basis_series(Vh, Dws, a0, Tn):
              """Z (Kp x Tn) fp16: rows 0..r-1 real and r..2r-1 imag parts of the r basis series over the analysed span,
              divided by their rms; returns Z and that scale as a 0-d float32 device tensor (no host sync). The division
              is a multiply by the float32 reciprocal: what CUDA torch does for a Python-float divisor (v8), bit for bit."""
              r = Vh.shape[0]
              Kp = ((2 * r + 63) // 64) * 64
              Z = torch.zeros((Kp, Tn), dtype=torch.float16, device=dev)
              scale = inv = None
              for c0 in range(0, r, RB):
                  nb = min(RB, r - c0)
                  qq = qz[:nb]; qq.zero_()
                  qq[:, :flen] = torch.conj(Vh[c0:c0 + nb]) * Dws
                  zb = torch.fft.ifft(qq, dim=1, norm='forward')[:, a0:a0 + Tn]
                  if scale is None:
                      scale = torch.sqrt((zb.real ** 2 + zb.imag ** 2).mean() / 2)
                      inv = torch.reciprocal(scale)
                  Z[c0:c0 + nb] = (zb.real * inv).half()
                  Z[r + c0:r + c0 + nb] = (zb.imag * inv).half()
              return Z, scale

          def stacked_A(A, nrm_s, scale, Kp):
              r = A.shape[1]
              As = A * (nrm_s * scale)[:, None]
              Are = torch.zeros((A.shape[0], Kp), dtype=torch.float32, device=dev); Aim = torch.zeros_like(Are)
              Are[:, :r] = As.real; Are[:, r:2 * r] = -As.imag
              Aim[:, :r] = As.imag; Aim[:, r:2 * r] = As.real
              return Are.half().contiguous(), Aim.half().contiguous()

          cons_pool = ThreadPoolExecutor(1)
          def cwait(m):
              f_ = m.pop('cfut', None)
              if f_ is not None:
                  tw = time.perf_counter(); f_.result(); T['consolidate_wait_s'] = T.get('consolidate_wait_s', 0.0) + time.perf_counter() - tw

          def consolidate(m):
              """pycbc EventManager.consolidate_events on one piece's peaks so far: gate veto (as at write), newsnr
              threshold, keep-loudest per interval. Applying it after each tile gives the same result as once at the end,
              since the loudest of a union are among the loudest of its parts."""
              tc_ = time.perf_counter()
              o = m['out']
              ks = [k for k, v in o.items() if v]
              if not ks:
                  return
              c = {k: np.concatenate(o[k]) for k in ks}
              ns = newsnr(c['snr'], c['chi'] / np.maximum(c['dof'], 1)) if do_chisq else c['snr']
              exempt = c['snr'] >= a.consolidate_exempt_snr if a.consolidate_exempt_snr is not None else np.zeros(len(ns), bool)
              keep = (ns >= (klo.newsnr_threshold or 0)) | exempt
              for g_t, g_w, g_taper in m['gates']:
                  keep &= np.abs(c['t'] - g_t) > g_w + g_taper + 0.5
              if klo.keep_loudest_interval:
                  assert klo.keep_loudest_stat in (None, 'newsnr'), klo.keep_loudest_stat
                  idx = np.nonzero(keep & ~exempt)[0]
                  ti = np.round((c['t'][idx] - m['gps_start']) * sr)
                  wt = (ti / (klo.keep_loudest_interval * sr)).astype(np.int64)
                  od = np.lexsort((-ns[idx], wt)); wt_s = wt[od]
                  first = np.r_[0, np.nonzero(np.diff(wt_s))[0] + 1]
                  rank = np.arange(len(od)) - np.repeat(first, np.diff(np.r_[first, len(od)]))
                  gated = keep.copy()
                  keep = np.zeros(len(keep), bool); keep[idx[od[rank < klo.keep_loudest_num]]] = True
                  keep |= gated & exempt
                  if a.pre_consolidate:   # K-th loudest newsnr of every full window: a later peak at or below it is never kept
                      full = np.diff(np.r_[first, len(od)]) >= klo.keep_loudest_num
                      fl = np.full(int(wt_s[-1]) + 1 if len(wt_s) else 1, -np.inf)
                      fl[wt_s[first[full]]] = ns[idx[od[first[full] + klo.keep_loudest_num - 1]]]
                      m['floor'] = torch.from_numpy(fl).to(dev)
              T['n_consolidated_out'] = T.get('n_consolidated_out', 0) + int((~keep).sum())
              T['consolidate_s'] = T.get('consolidate_s', 0.0) + time.perf_counter() - tc_
              for k in ks:
                  o[k] = [c[k][keep]]

          def pre_keep(m, s, samp_g, apx):
              """--pre-consolidate: mask over one tile and segment's candidates; False for rebuilt peaks that consolidate()
              drops anyway. Same newsnr cut (a rebuilt peak's newsnr is its SNR), gate veto and keep-loudest ranking with
              the same ties (stable, by position), on this part alone: a peak that is not among the loudest of its part
              is not among the loudest of the union. Times as at write, in float64. Exact candidates are all kept."""
              ex = apx >= a.exact_snr * (1 - a.margin)
              ok = ~ex & (apx > thr)
              T['n_prefilter_in'] = T.get('n_prefilter_in', 0) + int(ok.sum())
              t = m['gps_start'] + (int(m['seg_start'][s]) + samp_g).double() / sr
              exempt = ok & (apx >= a.consolidate_exempt_snr) if a.consolidate_exempt_snr is not None else torch.zeros_like(ok)
              ok &= (apx >= (klo.newsnr_threshold or 0)) | exempt
              for g_t, g_w, g_taper in m['gates']:
                  ok &= (t - g_t).abs() > g_w + g_taper + 0.5
              idx = torch.nonzero(ok & ~exempt)[:, 0]
              if klo.keep_loudest_interval and len(idx):
                  wt = (torch.round((t[idx] - m['gps_start']) * sr) / (klo.keep_loudest_interval * sr)).long()
                  if 'floor' in m:        # peaks at or below the K-th loudest already kept in their window (consolidate)
                      fl = m['floor']
                      bar = torch.where(wt < len(fl), fl[wt.clamp(max=len(fl) - 1)], torch.full_like(fl[:1], -np.inf))
                      above = apx[idx].double() > bar
                      ok[idx[~above]] = False; idx, wt = idx[above], wt[above]
                  o1 = torch.sort(-apx[idx], stable=True).indices
                  od = o1[torch.sort(wt[o1], stable=True).indices]
                  wt_s = wt[od]
                  ar = torch.arange(len(od), device=dev)
                  new = torch.ones(len(od), dtype=torch.bool, device=dev); new[1:] = wt_s[1:] != wt_s[:-1]
                  rank = ar - torch.cummax(torch.where(new, ar, torch.zeros_like(ar)), 0).values
                  top = torch.zeros_like(ok); top[idx[od[rank < klo.keep_loudest_num]]] = True
                  ok = top | (ok & exempt)
              T['n_prefilter_out'] = T.get('n_prefilter_out', 0) + int(ok.sum())
              return ex | ok

          qbuf = cp.zeros((QB, tlen), cp.complex64)
          rel = []                     # rebuilt / exact SNR of candidates, to check the margin
          dph = []                     # rebuilt - exact phase of candidates, rad
          for ti, ids in enumerate(tiles):
              if a.queue and nxt_job is None:
                  nxt_job = prefetch()     # the next job's data may have arrived since this group started
              n = len(ids)
              sync(); t0 = time.perf_counter()
              res = nxt.result()
              if ti + 1 < len(tiles):
                  nxt = gen_pool.submit(coarse_tile, ti + 1)
              H = gpu_part(ids, res); del res
              sync(); t1 = time.perf_counter(); T['templates_s'] += t1 - t0
              sig = torch.zeros((n, nseg_all), device=dev)
              for c0 in range(0, n, 128):
                  h = H[c0:c0 + 128]
                  sig[c0:c0 + 128] = (4.0 * df) * ((h.real ** 2 + h.imag ** 2) @ IPall.T)
              norm = (4.0 * df) / torch.sqrt(sig)
              sync(); t2 = time.perf_counter(); T['norm_s'] += t2 - t1
              r, A, Vh, worst = svd_tile(H)
              sync(); t3 = time.perf_counter(); T['svd_s'] += t3 - t2
              T['tiles'].append(dict(bank=bank_i, n=n, r=r, worst_lost_power=worst))
              # v9: depth-1 pipeline (seg_pipe.py): the front of segment s+1 (basis FFTs, GEMM + peaks) runs on the default
              # stream while the host does segment s's back (exact stage, sg_chisq, bookkeeping) on the back stream
              sig_h, norm_h = sig.cpu().numpy(), norm.cpu().numpy()     # one copy per tile, not per segment
              bstream.after_front()
              with bstream:
                  Hc = cp.from_dlpack(H)
                  stage = None
                  # regen's kernel needs chi-squared bins >= exact_stage.R fine bins; short --chunk2 chunks (df 1/8 Hz) give
                  # narrower low-frequency bins, so a chunked bank takes the dense exact path on its full templates H
                  if a.exact_stage == 'regen' and not (bank_i == 1 and a.chunk2):
                      stage = ExactStage(TemplateTile(Hc, nbins_all[ids], kmin, kmax_all[ids],
                                                      Coarse(grid, coarse_now['amp'], coarse_now['coef'])))
              if gate_on:
                  mc_lo, mc_hi = mc_all[ids].min() / a.gate_mc_ratio, mc_all[ids].max() * a.gate_mc_ratio

              def front_launch(m, s, a0, nw, Tn, gs):
                  """Enqueue segment s's GPU front; no host sync."""
                  with clock.span('basis_fft_s'):
                      Z, scale = basis_series(Vh, m['D'][s] * isw, a0, Tn)
                  with clock.span('gemm_s'):
                      Are, Aim = stacked_A(A, norm[:, gs], scale, Z.shape[0])
                      V, I = fused_peaks(Are, Aim, Z, n, Tn, KCFG)
                      vw, jj = V.view(n, nw, cw // BN).max(2)
                      iw = I.view(n, nw, cw // BN).gather(2, jj[..., None])[..., 0]
                  return dict(m=m, s=s, a0=a0, gs=gs, Z=Z, Are=Are, Aim=Aim, vw=vw, iw=iw, gsel=None)

              def front_collect(f):
                  """Candidate selection on the GPU (syncs the default stream for nonzero), then host copies only."""
                  ts_ = time.perf_counter()
                  m, s, a0, vw, iw, Z, Are, Aim = f['m'], f['s'], f['a0'], f['vw'], f['iw'], f['Z'], f['Are'], f['Aim']
                  tb, wb = torch.nonzero(vw > (thr * (1 - a.margin)) ** 2, as_tuple=True)
                  il = iw[tb, wb]
                  if a.pre_consolidate:
                      kp = pre_keep(m, s, a0 + il, torch.sqrt(vw[tb, wb]))
                      tb, wb, il = tb[kp], wb[kp], il[kp]
                  samp = (a0 + il).cpu().numpy().astype(np.int64)
                  approx = torch.sqrt(vw[tb, wb]).cpu().numpy()
                  if f['gsel'] is not None:
                      k_ = near(samp, f['gsel'], gw_s); kt = torch.from_numpy(k_).to(dev)
                      T['gate_peaks_dropped'] += int((~k_).sum())
                      tb, il, samp, approx = tb[kt], il[kt], samp[k_], approx[k_]
                  # rebuilt complex SNR at each candidate: the same A @ Z the GEMM did, for one column
                  with clock.span('rebuilt_z_s'):
                      zl = []
                      for c0 in range(0, len(samp), 1 << 18):
                          Zs = Z[:, il[c0:c0 + (1 << 18)]].T.float(); tt_ = tb[c0:c0 + (1 << 18)]
                          zr_ = (Are[tt_].float() * Zs).sum(1); zi_ = (Aim[tt_].float() * Zs).sum(1)
                          zl.append(torch.complex(zr_, zi_))
                  zreb = torch.cat(zl).cpu().numpy() if zl else np.zeros(0, np.complex64)
                  tb = tb.cpu().numpy()
                  T['select_s'] = T.get('select_s', 0.0) + time.perf_counter() - ts_
                  return dict(m=m, s=s, gs=f['gs'], tb=tb, samp=samp, approx=approx, zreb=zreb)

              def back(c):
                  """v8's per-segment refine, unchanged except: host copies of sig/norm, and the back stream's sync."""
                  t2 = time.perf_counter()
                  m, s, gs, tb, samp, approx, zreb = c['m'], c['s'], c['gs'], c['tb'], c['samp'], c['approx'], c['zreb']
                  # peaks below the exact cut keep their rebuilt values; the rest go to the exact stage
                  ex = approx >= a.exact_snr * (1 - a.margin)
                  rb = ~ex & (approx > thr)
                  if rb.any():
                      o = m['out']; tid_r = ids[tb[rb]]; nrb = int(rb.sum())
                      o['tid'].append(tid_r); o['t'].append(m['gps_start'] + (m['seg_start'][s] + samp[rb]) / sr)
                      o['snr'].append(approx[rb]); o['ph'].append(np.angle(zreb[rb]).astype(np.float32))
                      o['sg'].append(sig_h[:, gs][tb[rb]].astype(np.float64)); o['ex'].append(np.zeros(nrb, bool))
                      if do_chisq:
                          o['chi'].append(np.zeros(nrb, np.float32)); o['dof'].append(np.zeros(nrb, np.int32))
                          if do_sg:
                              o['sgc'].append(np.ones(nrb, np.float32)); o['sgd'].append(np.zeros(nrb, bool))
                      T['n_rebuilt_peaks'] += nrb
                  tb, samp, approx, zreb = tb[ex], samp[ex], approx[ex], zreb[ex]
                  T['n_exact_candidates'] += len(tb)
                  # exact complex SNR (and chi-squared) at each candidate, from the exact template
                  T['n_candidates'] += len(tb)
                  zx = np.zeros(len(tb), np.complex64); pw = np.zeros(len(tb), np.float32); fpk = np.zeros(len(tb))
                  ur = np.unique(tb)
                  ot = np.argsort(tb, kind='stable'); tbs = tb[ot]
                  Dc, IPc = cp.from_dlpack(m['D'][s]), cp.from_dlpack(m['IP'][s])
                  if stage is not None and len(tb):
                      tk = time.perf_counter()
                      pv = stage.evaluate(SegmentData(Dc, IPc, df, tlen), PeakRequest(tb, samp))
                      zx, pw, fpk = pv.z, pv.pw, pv.fpeak
                      bsync(); T['rf_kernel_s'] += time.perf_counter() - tk
                      ur = ur[:0]
                  for c0 in range(0, len(ur), QB):
                      rows = ur[c0:c0 + QB]
                      sel = ot[np.searchsorted(tbs, rows[0]):np.searchsorted(tbs, rows[-1], side='right')]
                      loc = np.searchsorted(rows, tb[sel])
                      tq = time.perf_counter()
                      need_hs = not a.fuse_q or (do_chisq and (a.edges != 'fused' or 'edges_check' not in T))
                      Hs = Hc[cp.asarray(rows)] if need_hs else None
                      if not a.fuse_q:
                          qv = qbuf[:len(rows)]
                          if a.legacy_qzero:
                              qv[:, flen:] = 0          # the kernel never reads above flen and qbuf starts zeroed
                          qv[:, :flen] = cp.conj(Hs) * Dc[None, :]
                      tids = ids[rows]
                      bsync(); te = time.perf_counter(); T['rf_q_s'] += te - tq
                      if do_chisq:
                          if a.edges == 'fused':
                              E = edges_fused(cp, rows, nbins_all[tids], kmin, kmax_all[tids], Hc, IPc)[loc]
                              if 'edges_check' not in T:    # once per run, on the GPU: fused edges against the full float64 ones
                                  E0 = edges_per_template(cp, loc, nbins_all[tids[loc]], kmin, kmax_all[tids[loc]], Hs, IPc)
                                  T['edges_check'] = dict(n=int(E.size), n_diff=int((E != E0).sum()),
                                                          max_abs_diff=int(np.abs(E - E0).max()))
                                  assert T['edges_check']['max_abs_diff'] <= 1, T['edges_check']
                          elif a.edges == 'full':
                              E = edges_per_template(cp, loc, nbins_all[tids[loc]], kmin, kmax_all[tids[loc]], Hs, IPc)
                          else:
                              E = edges_blocked(cp, loc, nbins_all[tids[loc]], kmin, kmax_all[tids[loc]], Hs, IPc)
                              if 'edges_check' not in T:    # once per run, on the GPU: blocked edges must equal the full ones
                                  E0 = edges_per_template(cp, loc, nbins_all[tids[loc]], kmin, kmax_all[tids[loc]], Hs, IPc)
                                  T['edges_check'] = dict(n=int(E.size), n_diff=int((E != E0).sum()))
                                  assert T['edges_check']['n_diff'] == 0, T['edges_check']
                          nbp = nbins_all[tids[loc]]; r_ = np.arange(len(sel))
                          # sine-Gaussian reference frequency, pycbc's (bins[-2] + (bins[-2] - bins[-3])) * df
                          fpk[sel] = (2 * E[r_, np.maximum(nbp - 1, 0)] - E[r_, np.maximum(nbp - 2, 0)]) * df
                      else:
                          E = np.stack([kmin_all[tids[loc]], kmax_all[tids[loc]]], 1)
                      bsync(); tk = time.perf_counter(); T['rf_edges_s'] += tk - te
                      if a.fuse_q:
                          pw[sel], zx[sel] = bin_power_sums(Hc, rows[loc], samp[sel], E, want_z=True, d=Dc, n_time=tlen)
                      else:
                          pw[sel], zx[sel] = bin_power_sums(qv, loc, samp[sel], E, want_z=True)
                      bsync(); T['rf_kernel_s'] += time.perf_counter() - tk
                      del Hs
                  nr = norm_h[:, gs][tb]
                  snr = np.abs(zx) * nr
                  if len(tb):
                      rel.append(approx / np.maximum(snr, 1e-30))
                      dph.append(np.angle(zreb * np.conj(zx)))
                  keep = snr > thr
                  tb, samp, snr, zx, pw, nr, fpk = tb[keep], samp[keep], snr[keep], zx[keep], pw[keep], nr[keep], fpk[keep]
                  tid = ids[tb]
                  o = m['out']
                  o['tid'].append(tid); o['t'].append(m['gps_start'] + (m['seg_start'][s] + samp) / sr)
                  o['snr'].append(snr); o['ph'].append(np.angle(zx).astype(np.float32)); o['ex'].append(np.ones(len(tb), bool))
                  o['sg'].append(sig_h[:, gs][tb].astype(np.float64))
                  if do_chisq:
                      chi_ = ((nbins_all[tid] * pw - np.abs(zx) ** 2) * nr ** 2).astype(np.float32)
                      dof_ = (2 * nbins_all[tid] - 2).astype(np.int32)
                      o['chi'].append(chi_); o['dof'].append(dof_)
                      T['n_chisq_points'] += int(len(tid))
                      if do_sg:
                          ts_ = time.perf_counter()
                          sgv = np.ones(len(tid), np.float32); sgd = np.zeros(len(tid), bool)
                          ns_ = newsnr(snr, chi_ / np.maximum(dof_, 1))
                          cand = [i for i in np.nonzero(ns_ >= sgo.sgchisq_snr_threshold)[0] if sg_locs[tid[i]] is not None]
                          if a.sg_batch and cand:
                              cand = np.array(cand)
                              groups = {}
                              for i in cand:
                                  groups.setdefault(tuple(sg_locs[tid[i]]), []).append(i)
                              for vals_, gi in groups.items():
                                  gi = np.array(gi)
                                  if a.sg_kernel:
                                      v_, ok_ = sg_chisq_points(Dc, IPc, samp[gi], fpk[gi], list(vals_), kmin_all[tid[gi]], df, tlen)
                                  else:
                                      v_, ok_ = sg_chisq_batch(Dc, IPc, samp[gi], fpk[gi], list(vals_), kmin_all[tid[gi]])
                                  sgv[gi[ok_]] = v_[ok_]; sgd[gi[ok_]] = True
                              if 'sg_batch_check' not in T:    # once per run: batched values equal the per-peak ones
                                  i0 = cand[:64 if a.sg_kernel else 4]
                                  ref = [sg_chisq_one(Dc, IPc, samp[i], fpk[i], sg_locs[tid[i]], int(kmin_all[tid[i]])) for i in i0]
                                  d_ = [abs(sgv[i] - r_) / max(abs(r_), 1e-30) for i, r_ in zip(i0, ref) if r_ is not None]
                                  T['sg_batch_check'] = dict(n=len(d_), max_rel=float(max(d_)) if d_ else 0.0)
                                  assert T['sg_batch_check']['max_rel'] < (1e-4 if a.sg_kernel else 1e-6), T['sg_batch_check']
                              cand = []
                          for i in cand:
                              v = sg_chisq_one(Dc, IPc, samp[i], fpk[i], sg_locs[tid[i]], int(kmin_all[tid[i]]))
                              if v is not None:
                                  sgv[i] = v; sgd[i] = True
                          o['sgc'].append(sgv); o['sgd'].append(sgd)
                          T['n_sg_chisq_points'] += int(sgd.sum()); T['sg_chisq_s'] += time.perf_counter() - ts_
                  bsync(); T['refine_chisq_s'] += time.perf_counter() - t2

              def finish_dataset(m):
                  m.setdefault('sig', np.zeros((ntmp, len(m['first']))))[ids] = sig_h[:, m['seg0'] + np.asarray(m['first'])]
                  if a.consolidate:
                      # --pre-consolidate: on a CPU thread while the GPU filters the other datasets (cwait before m is next used)
                      m['cfut'] = cons_pool.submit(consolidate, m) if a.pre_consolidate else consolidate(m)

              clock = EventClock(torch); pipe = SegmentPipe(back, bstream)
              for m in metas:
                  cwait(m)
                  for s, (a0, a1) in enumerate(m['analyze']):
                      gs = m['seg0'] + s
                      nw = m['NW'][s]; Tn = nw * cw
                      gsel = None
                      if gate_on:
                          gsamp, gmc = m['gate'][s]
                          gsel = gsamp[(gmc >= mc_lo) & (gmc <= mc_hi)]
                          if not len(gsel):
                              T['gate_skips'] += 1
                              continue
                          T['gate_runs'] += 1
                      def launch(m=m, s=s, a0=a0, nw=nw, Tn=Tn, gs=gs, gsel=gsel):
                          f = front_launch(m, s, a0, nw, Tn, gs); f['gsel'] = gsel
                          return f
                      pipe.step(launch, front_collect)
                  pipe.then(lambda m=m: finish_dataset(m))
              tw_ = time.perf_counter(); pipe.drain(); T['drain_s'] = T.get('drain_s', 0.0) + time.perf_counter() - tw_
              clock.flush(T)
              if stage is not None:
                  T.setdefault('regen_check', []).append(stage.check)
              del H, Hc, h, A, Vh, sig, norm, stage     # h (a view) kept the whole H alive into the next tile: 2 x 4 GiB at once
              torch.cuda.empty_cache()
              print(f'tile {ti + 1}/{len(tiles)} n={n} r={r} elapsed {time.perf_counter() - t_start:.0f}s gen {T["templates_s"]:.0f} '
                    f'svd {T["svd_s"]:.0f} basis {T["basis_fft_s"]:.0f} gemm {T["gemm_s"]:.0f} refine {T["refine_chisq_s"]:.0f}', flush=True)

          print(f'bank {bank_i + 1} tiles done {time.perf_counter() - t_start:.1f}s', flush=True)
          qz = qbuf = None             # else this bank's buffers stay alive while the next bank allocates its own
          for m in metas:
              cwait(m)
          print(f'bank {bank_i + 1} consolidated {time.perf_counter() - t_start:.1f}s', flush=True)
          rel = np.concatenate(rel) if rel else np.zeros(0)
          T['rebuilt_over_exact' + sfx] = dict(n=int(len(rel)), **({} if not len(rel) else dict(
              min=float(rel.min()), p01=float(np.percentile(rel, 0.1)), p1=float(np.percentile(rel, 1)), median=float(np.median(rel)),
              p99=float(np.percentile(rel, 99)), max=float(rel.max()))))

          dph = np.abs(np.concatenate(dph)) if dph else np.zeros(0)
          T['rebuilt_phase_err_rad' + sfx] = dict(n=int(len(dph)), **({} if not len(dph) else dict(
              median=float(np.median(dph)), p99=float(np.percentile(dph, 99)), p999=float(np.percentile(dph, 99.9)), max=float(dph.max()))))

          def write_one(tag, o, sig, gates, pdir_, hashes_, sfx_):
              tw_ = time.perf_counter()
              cat = lambda k, dt: np.concatenate(o[k]) if o[k] else np.array([], dt)
              tid, t, snr, ph, sg = cat('tid', int), cat('t', float), cat('snr', float), cat('ph', np.float32), cat('sg', float)
              keep = np.ones(len(snr), bool)
              for g_t, g_w, g_taper in gates:
                  keep &= np.abs(t - g_t) > g_w + g_taper + 0.5
              srt = np.lexsort((t[keep], tid[keep]))
              sel = np.nonzero(keep)[0][srt]
              fn = os.path.join(a.out, pdir_, tag + '.hdf')
              with h5py.File(fn + '.part', 'w') as f:
                  _put(f, 'template_id', tid[sel]); _put(f, 'template_hash', hashes_[tid[sel]]); _put(f, 'snr', snr[sel]); _put(f, 'end_time', t[sel])
                  _put(f, 'sigmasq', sig)
                  _put(f, 'coa_phase', ph[sel]); _put(f, 'peak_sigmasq', sg[sel])
                  _put(f, 'exact', cat('ex', bool)[sel])          # False: rebuilt SNR and phase, no chi-squared (--exact-snr)
                  if do_chisq:
                      chi, dof = cat('chi', np.float32)[sel], cat('dof', np.int32)[sel]
                      _put(f, 'chisq', chi); _put(f, 'chisq_dof', dof); _put(f, 'newsnr', newsnr(snr[sel], chi / np.maximum(dof, 1)))
                  if do_sg:
                      # pycbc convention: 1 (no penalty) where not computed; sg_chisq_computed says where it was
                      _put(f, 'sg_chisq', cat('sgc', np.float32)[sel]); _put(f, 'sg_chisq_computed', cat('sgd', bool)[sel])
                  f.attrs['n_gates'] = len(gates)
                  f.attrs['provenance'] = prov_json; f.attrs['config_md5'] = prov['config_md5']
                  gt = np.array(gates, np.float64).reshape(-1, 3)        # gated stretches, for vetoes and live time
                  f['gates/time'] = gt[:, 0]; f['gates/width'] = gt[:, 1]; f['gates/taper'] = gt[:, 2]
              os.replace(fn + '.part', fn)
              T['n_peaks'][tag + sfx_] = int(len(sel))
              T['write_s'] = T.get('write_s', 0.0) + time.perf_counter() - tw_
          for m in metas:       # --queue: h5py compresses without the GIL, so peak files are written while the GPU goes on
              job_ = (m['tag'], m['out'], m['sig'], m['gates'], pdir, hashes, sfx)
              wfuts.append(write_pool.submit(write_one, *job_)) if a.queue else write_one(*job_)
          print(f'bank {bank_i + 1} written {time.perf_counter() - t_start:.1f}s', flush=True)
        for f_ in wfuts:
            f_.result()
        T['total_s'] = time.perf_counter() - t_start
        T['gpu_peak_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30; torch.cuda.reset_peak_memory_stats()
        snap = os.environ.get('GW_MEMSNAP')        # DIR: allocation history of the first GW_MEMSNAP_N groups, for memory profiling
        if snap and memo.get('n_groups', 0) + 1 == int(os.environ.get('GW_MEMSNAP_N', '3')):
            torch.cuda.memory._dump_snapshot(os.path.join(snap, 'mem_snapshot.pickle')); torch.cuda.memory._record_memory_history(enabled=None)
        memo['n_groups'] = memo.get('n_groups', 0) + 1
        rs = np.array([t_['r'] for t_ in T['tiles']]); ns = np.array([t_['n'] for t_ in T['tiles']])
        T.update(n_templates=ntmp, n_datasets=len(metas), tile=a.tile, eps=a.eps, margin=a.margin, templates=a.templates,
                 approximant=a.approximant, r_over_n=float(rs.sum() / ns.sum()), bank2=a.bank2, gpu=torch.cuda.get_device_name(0),
                 data_ks=sum((a1 - a0) / m['sr'] for m in metas for a0, a1 in m['analyze']) / 1000)
        json.dump(T, open(os.path.join(a.out, 'timing.json'), 'w'), indent=1, default=float)
        log_history(a.history, 'fp_filter', 'done', a.out, prov, n_peaks=sum(T['n_peaks'].values()), total_s=T['total_s'])
        print(json.dumps({k: v for k, v in T.items() if k not in ('n_peaks', 'prep_cpu_s_each', 'prep_timing', 'tiles', 'gemm_tune', 'provenance')}, default=float), flush=True)
        if a.queue:
            del metas, IPall
            cp.get_default_memory_pool().free_all_blocks(); torch.cuda.empty_cache()
        return nxt_job

    if not a.queue:
        run_group(submit_prep(a.datasets) if a.prep == 'gpu' else None)
        return
    if os.environ.get('GW_MEMSNAP'):
        import torch
        torch.cuda.memory._record_memory_history(max_entries=200000)
    job, pend = next_job(True), None
    while job:
        a.datasets, a.out = job
        nxt = run_group(pend)
        job, pend = nxt if nxt else (next_job(True), None)
    pools[0].close(); pools[0].join()


if __name__ == '__main__':
    main()
