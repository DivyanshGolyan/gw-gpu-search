#!/usr/bin/env python
"""Compare our recovery of catalogued mergers with each LIGO search pipeline's own false-alarm rate.

Fetches every catalogued event in results/o4{a,b}_results.json from the GWOSC event API, keeps the
per-pipeline search FARs (pycbc, gstlal, MBTA, cWB), and prints our recovery rate binned by the PyCBC FAR.
Usage: explain_misses.py [--cacert BUNDLE] [--cached catalog_with_pipelines.json]
gwosc.org needs the "emSign Root CA - G1" certificate. Pass a bundle that has it if TLS fails."""
import argparse, json, os, ssl, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--cacert')
ap.add_argument('--cached', help='skip the download and read this file')
a = ap.parse_args()


def fetch():
    ctx = ssl.create_default_context(cafile=a.cacert)
    get = lambda u: json.load(urllib.request.urlopen(u, context=ctx, timeout=60))
    latest = {}
    for e in get('https://gwosc.org/eventapi/json/GWTC/')['events'].values():
        n = e['commonName']
        if n not in latest or e['version'] > latest[n]['version']:
            latest[n] = e
    cat = [dict(run=r, **c) for r in ('o4a', 'o4b') for c in json.load(open(f'{RES}/{r}_results.json'))['catalog']]

    def one(c):
        ev = list(get(latest[c['name']]['jsonurl'].rstrip('/') + '/')['events'].values())[0]
        s = {}
        for k, v in ev.get('parameters', {}).items():
            if v.get('pipeline_type') == 'search':
                p = [x for x in k.split('_') if not x.startswith(('GW', 'GWTC'))]
                s[p[0] if p else k] = (v.get('far'), v.get('network_matched_filter_snr'))
        return dict(c, search=s, ifos=sorted({x.get('detector') for x in ev.get('strain', [])}),
                    net_snr=ev.get('network_matched_filter_snr'), catalog_name=ev.get('catalog.shortName'))
    with ThreadPoolExecutor(6) as ex:
        return list(ex.map(one, cat))


cat = json.load(open(a.cached)) if a.cached else fetch()
found = lambda o: o['far_union'] is not None and o['far_union'] < 1
pycbc = lambda o: (o['search'].get('pycbc') or (None,))[0]
n_conf = lambda o: sum(1 for f, _ in o['search'].values() if f is not None and f < 1)
print(f'{sum(map(found, cat))} of {len(cat)} catalogued events found below 1/yr')
for lo, hi, lab in [(0, 0.01, '< 0.01'), (0.01, 0.1, '0.01-0.1'), (0.1, 1, '0.1-1'), (1, 1e99, '>= 1')]:
    sel = [o for o in cat if pycbc(o) is not None and lo <= pycbc(o) < hi]
    print(f'PyCBC FAR {lab:9} /yr: {sum(map(found, sel)):3} of {len(sel):3}')
sel = [o for o in cat if pycbc(o) is None]
print(f'no PyCBC entry       : {sum(map(found, sel)):3} of {len(sel):3}')
for n in range(1, 5):
    sel = [o for o in cat if n_conf(o) == n]
    print(f'{n} pipeline(s) below 1/yr: {sum(map(found, sel)):3} of {len(sel):3}')
