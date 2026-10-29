#!/usr/bin/env python
"""Whole-run readout from the per-chunk rankings (results/o4a/CHUNK/rank_{aligned,ecc}/fg.hdf).

Per chunk: live time, background, FAR floor. Candidates: foreground of both layers merged within 1 s; reported FAR =
2 x min(layer FARs) (frozen union bound), clamped at the chunk floor. Cross-match within 1 s of any catalog row
(catalogs/o4_known.csv: GWTC-4.0/4.1/5.0 + candidate releases + GraceDB) -> KNOWN, else NEW.
Catalog recovery: GWTC confident events (GWTC-4.0/4.1, catalog FAR < 1/yr) inside coincident time.
Usage: o4a_results.py RESULTS_DIR KNOWN_CSV OUT_PREFIX [--skip wk01]
"""
import argparse, csv, glob, json, os
import numpy as np, h5py

ap = argparse.ArgumentParser()
ap.add_argument('root'); ap.add_argument('known'); ap.add_argument('out')
ap.add_argument('--skip', nargs='*', default=['wk01'], help='development chunks (FREEZE.md: never a discovery claim)')
a = ap.parse_args()
YEAR = 365.25 * 86400

chunks = sorted(d for d in os.listdir(a.root) if d.startswith('wk') and all(
    os.path.exists(f'{a.root}/{d}/rank_{l}/fg.hdf') for l in ('aligned', 'ecc')) and d not in a.skip)
cat = [r for r in csv.DictReader(open(a.known))]
ct = np.array([float(r['gps']) for r in cat])

def far(r):
    try: return float(r['far_per_yr'])
    except ValueError: return np.inf

rows, cand, cat_rec = [], [], []
for c in chunks:
    segs = np.loadtxt(f'{a.root}/{c}/rank_aligned/coinc_segments.txt', ndmin=2)
    L = {}
    for l in ('aligned', 'ecc'):
        with h5py.File(f'{a.root}/{c}/rank_{l}/fg.hdf') as f:
            g = f['foreground']; at = dict(f.attrs)
            L[l] = {k: g[k][:] for k in g}
    rows.append(dict(chunk=c, live_s=float(at['live_time_s']), bg_yr=float(at['background_time_yr']),
                     floor=float(at['far_floor_per_yr']), n_gates=int(at.get('n_gates', -1))))
    floor = rows[-1]['floor']
    ev = []
    for l in L:
        for i in range(len(L[l]['stat'])):
            ev.append(dict(layer=l, gps=float(L[l]['H1_time'][i]), far=max(float(L[l]['far_excl_per_yr'][i]), floor),
                           ul=bool(L[l]['far_is_upper_limit'][i]), stat=float(L[l]['stat'][i]),
                           H1_snr=float(L[l]['H1_snr'][i]), L1_snr=float(L[l]['L1_snr'][i]),
                           m1=float(L[l]['mass1'][i]), m2=float(L[l]['mass2'][i]), tid=int(L[l]['template_id'][i]),
                           gate_dt=float(L[l]['gate_dt'][i]), env_H1=int(L[l]['env_H1'][i]), env_L1=int(L[l]['env_L1'][i])))
    ev.sort(key=lambda e: e['gps'])
    groups = []
    for e in ev:
        if groups and e['gps'] - groups[-1][-1]['gps'] < 1: groups[-1].append(e)
        else: groups.append([e])
    for g in groups:
        best = min(g, key=lambda e: e['far'])
        fl = {e['layer']: e['far'] for e in sorted(g, key=lambda e: -e['far'])}
        j = np.nonzero(np.abs(ct - best['gps']) < 1)[0]
        names = sorted({cat[k]['name'] for k in j})
        cand.append(dict(chunk=c, **best, far_aligned=fl.get('aligned'), far_ecc=fl.get('ecc'),
                         far_union=max(2 * best['far'], floor), status='KNOWN' if len(j) else 'NEW',
                         known='; '.join(names), cat_far_min=min((far(cat[k]) for k in j), default=None)))
    live = lambda t: ((segs[:, 0] <= t) & (segs[:, 1] > t)).any()
    seen = set()
    for r in cat:
        if r['source'] in ('GWTC-4.0', 'GWTC-4.1', 'GWTC-5.0') and far(r) < 1 and r['name'] not in seen and live(float(r['gps'])):
            seen.add(r['name'])
            m = [x for x in cand if x['chunk'] == c and abs(x['gps'] - float(r['gps'])) < 1]
            cat_rec.append(dict(chunk=c, name=r['name'], cat_far=far(r), far_union=m[0]['far_union'] if m else None,
                                found_layers=[k for k in ('aligned', 'ecc') if m and m[0]['far_' + k] is not None]))

T = sum(r['live_s'] for r in rows) / YEAR
res = dict(chunks=rows, n_chunks=len(rows), live_days=T * 365.25, catalog=cat_rec,
           below_1=[x for x in cand if x['far_union'] < 1],
           top_uncatalogued=sorted([x for x in cand if x['status'] == 'NEW'], key=lambda x: x['far_union'])[:10])
for x in res['below_1'] + res['top_uncatalogued']:
    x['fap_whole_run'] = float(1 - np.exp(-x['far_union'] * T))
json.dump(res, open(a.out + '.json', 'w'), indent=1, default=float)
rec = sum(1 for r in cat_rec if r['far_union'] is not None and r['far_union'] < 1)
print(f"{len(rows)} chunks, {T * 365.25:.1f} d live; catalog events (GWTC FAR<1/yr) {len(cat_rec)}, recovered <1/yr {rec}")
print('below 1/yr:', len(res['below_1']), 'NEW:', sum(x['status'] == 'NEW' for x in res['below_1']))
for x in res['top_uncatalogued']:
    print(f"  {x['chunk']} {x['gps']:.3f} union {x['far_union']:.3g}/yr layer {x['layer']} H1 {x['H1_snr']:.1f} L1 {x['L1_snr']:.1f} "
          f"m {x['m1']:.1f}+{x['m2']:.1f} gate_dt {x['gate_dt']:.1f} env {x['env_H1']}/{x['env_L1']}")
