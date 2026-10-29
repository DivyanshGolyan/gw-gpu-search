"""Pieces for all of an observing run, cut once from one GWOSC timeline query, then split into weekly chunks.

Coincident H1 and L1 science time (IFO_DATA) over [--start, --end) is cut span by span into pieces of at most
--max-len s whose neighbours overlap by 176 s. Spans shorter than --min-len are dropped. A piece's trigger window is
[s + 56, e - 24] for the first piece of a span (--segment-start-pad 48) and [s + 152, e - 24] for the others (5th
datasets column 152), so windows abut inside a span. Each piece belongs to the week that holds its trigger start.
Week 1 is [1372349000, 1372955000) (the slice-1 week); later weeks are 604800 s long from 1372955000, earlier ones
604800 s long ending at 1372349000 and are named wkm01, wkm02, ... going back.
Writes OUT/wkNN/{datasets.txt, coinc_segments.txt, pieces.json} and OUT/weeks.json.
Usage: make_pieces_o4a.py OUT [--start 1368975618] [--end 1389456018] [--store-dir /work/stores]   (REQUESTS_CA_BUNDLE for gwosc.org)"""
import argparse, json, math, os
from gwosc.timeline import get_segments
OV, PAD, SP0, EP = 176, 8, 48, 16
W1S, W1E, WEEK = 1372349000, 1372955000, 604800
ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--start', type=int, default=1368975618); ap.add_argument('--end', type=int, default=1389456018)
ap.add_argument('--store-dir', default='/work/stores'); ap.add_argument('--max-len', type=int, default=11000)
ap.add_argument('--min-len', type=int, default=1100)
a = ap.parse_args()


def week_of(t):
    if W1S <= t < W1E: return 'wk01'
    if t >= W1E: return 'wk%02d' % (2 + (t - W1E) // WEEK)
    return 'wkm%02d' % (1 + (W1S - 1 - t) // WEEK)


H, L = (get_segments(f'{i}_DATA', a.start, a.end) for i in ('H1', 'L1'))
co, i, j = [], 0, 0
while i < len(H) and j < len(L):
    s, e = max(H[i][0], L[j][0]), min(H[i][1], L[j][1])
    if e - s >= a.min_len: co.append((s, e))
    if H[i][1] < L[j][1]: i += 1
    else: j += 1
weeks = {}
for s, e in co:
    n = max(1, math.ceil((e - s - OV) / (a.max_len - OV)))
    step = (e - s - OV) / n
    for k in range(n):
        ps = int(s + round(k * step)); pe = int(s + round((k + 1) * step) + OV) if k < n - 1 else e
        ts = ps + (PAD + 144 if k else PAD + SP0); te = pe - PAD - EP
        w = weeks.setdefault(week_of(ts), dict(ds=[], cs=[]))
        for ifo in ('H1', 'L1'):
            w['ds'].append(f'{ifo} {ps} {pe} {a.store_dir}/{ifo}-{ps}-{pe}.hdf' + (' 152' if k else ''))
        w['cs'].append((ts, te))
summary = {}
for name, w in sorted(weeks.items()):
    d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'datasets.txt'), 'w').write('\n'.join(w['ds']) + '\n')
    cs = sorted(w['cs'])
    assert all(x[1] <= y[0] for x, y in zip(cs, cs[1:])), name
    open(os.path.join(d, 'coinc_segments.txt'), 'w').write(''.join(f'{x} {y}\n' for x, y in cs))
    summary[name] = dict(n_pieces=len(w['ds']) // 2, analysed_s=sum(y - x for x, y in cs),
                         data_s=sum(int(l.split()[2]) - int(l.split()[1]) for l in w['ds'] if l.startswith('H1')))
    json.dump(summary[name], open(os.path.join(d, 'pieces.json'), 'w'), indent=1)
json.dump(dict(start=a.start, end=a.end, n_spans=len(co), coinc_s=sum(e - s for s, e in co), weeks=summary),
          open(os.path.join(a.out, 'weeks.json'), 'w'), indent=1)
print(len(summary), 'weeks,', sum(v['n_pieces'] for v in summary.values()), 'pieces,',
      '%.1f days analysed' % (sum(v['analysed_s'] for v in summary.values()) / 86400))
