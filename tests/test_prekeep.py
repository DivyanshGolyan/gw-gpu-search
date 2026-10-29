"""pre_keep (GPU pre-consolidation) must leave consolidate()'s result unchanged: random parts, ties, gates, exempt peaks."""
import sys, textwrap, types, numpy as np, torch
from pycbc.events.ranking import newsnr
src = open(sys.argv[1]).read().split('\n')
def grab(name):
    i = next(k for k, l in enumerate(src) if l.strip().startswith(f'def {name}('))
    ind = len(src[i]) - len(src[i].lstrip()); out = [src[i]]
    for l in src[i + 1:]:
        if l.strip() and len(l) - len(l.lstrip()) <= ind: break
        out.append(l)
    return textwrap.dedent('\n'.join(out))
rng = np.random.default_rng(1)
fails = 0
for trial in range(300):
    a = types.SimpleNamespace(pre_consolidate=True, exact_snr=5.5, margin=0.04, consolidate_exempt_snr=[5.0, None][trial % 2])
    klo = types.SimpleNamespace(newsnr_threshold=4.0, keep_loudest_interval=[1.072, None][trial % 7 == 0], keep_loudest_num=int(rng.integers(1, 6)), keep_loudest_stat='newsnr')
    sr = 2048.0; thr = 4.0
    import time; g = dict(time=time, a=a, klo=klo, sr=sr, thr=thr, T={}, dev='cpu', torch=torch, np=np, newsnr=newsnr, do_chisq=True)
    exec(grab('consolidate'), g); exec(grab('pre_keep'), g)
    m = dict(gps_start=1380077576.0, seg_start=[0, 1636352], gates=[(1380077576.0 + rng.uniform(0, 1500), 0.25, 0.25) for _ in range(rng.integers(0, 3))])
    full = dict(m, out={k: [] for k in ('tid', 't', 'snr', 'chi', 'dof', 'ph', 'sg', 'sgc', 'sgd', 'ex')})
    pre = dict(m, out={k: [] for k in full['out']})
    for part in range(int(rng.integers(1, 6))):
        s = int(rng.integers(0, 2)); n = int(rng.integers(0, 3000))
        samp = torch.from_numpy(rng.integers(0, int(rng.choice([20, 800])) * 2048, n)).to(torch.int32)
        apx = torch.from_numpy(np.round(rng.uniform(3.84, 7, n), 2).astype(np.float32))    # rounded: many ties
        kp = g['pre_keep'](pre, s, samp, apx).numpy()
        sp, ap_ = samp.numpy().astype(np.int64), apx.numpy()
        ex = ap_ >= a.exact_snr * (1 - a.margin); rb = ~ex & (ap_ > thr)
        chi_ex = rng.uniform(0, 3, n).astype(np.float32) * 20
        for mm, keep in ((full, np.ones(n, bool)), (pre, kp)):
            o = mm['out']
            for sel, isex in ((rb & keep, False), (ex & keep, True)):
                o['tid'].append(np.nonzero(sel)[0] + 10000 * part); o['t'].append(mm['gps_start'] + (mm['seg_start'][s] + sp[sel]) / sr)
                o['snr'].append(ap_[sel]); o['ph'].append(np.zeros(sel.sum(), np.float32)); o['sg'].append(np.zeros(sel.sum()))
                o['ex'].append(np.full(sel.sum(), isex)); o['chi'].append(chi_ex[sel] if isex else np.zeros(sel.sum(), np.float32))
                o['dof'].append(np.full(sel.sum(), 10 if isex else 0, np.int32)); o['sgc'].append(np.ones(sel.sum(), np.float32)); o['sgd'].append(np.zeros(sel.sum(), bool))
        g['consolidate'](full); g['consolidate'](pre)
    key = lambda mm: sorted(zip(np.concatenate(mm['out']['tid']).tolist(), np.concatenate(mm['out']['t']).tolist())) if mm['out']['tid'] else []
    if key(full) != key(pre):
        fails += 1
tot = 0
print('trials 300, mismatches', fails, 'prefilter in/out last', g['T'].get('n_prefilter_in'), g['T'].get('n_prefilter_out'))
