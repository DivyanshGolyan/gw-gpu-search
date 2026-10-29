#!/usr/bin/env python
"""Known O4 events and alerts, for flagging candidates that are in no catalog.

Sources (all public):
  GWOSC event API: GWTC-4.0, GWTC-4.1, GWTC-5.0, O4_Discovery_Papers (includes marginal entries with far/p_astro)
  GWTC-4.0 and GWTC-5.0 candidate data releases (zenodo 16053640 -> 17602502, 20348004): the search summary
  tables, i.e. every candidate each pipeline (cWB, GstLAL, MBTA, PyCBC) reported down to its release
  threshold, subthreshold ones included; one row per pipeline candidate
  GraceDB public superevents for runid O4a, O4b, O4c (every public alert, low-significance ones included;
  retracted ones are kept and marked, since a retracted alert at our time is still worth knowing)
Writes a CSV: name, gps, source, far_per_yr, p_astro, note.
Usage: fetch_catalogs.py OUT.csv [--cacert BUNDLE]  (gwosc needs the emSign bundle in fresh containers)
"""
import argparse, csv, json, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--cacert', default=None)
ap.add_argument('--runs', default='O4a,O4b,O4c')
a = ap.parse_args()


def get(url, cacert=None):
    cmd = ['curl', '-sfL', '--retry', '4', url] + (['--cacert', cacert] if cacert else [])
    return json.loads(subprocess.run(cmd, check=True, capture_output=True).stdout)


rows = []
for cat in ('GWTC-4.0', 'GWTC-4.1', 'GWTC-5.0', 'O4_Discovery_Papers'):
    ev = get(f'https://gwosc.org/eventapi/json/{cat}/', a.cacert)['events']
    for k, e in ev.items():
        rows.append(dict(name=e['commonName'], gps=e['GPS'], source=cat, far_per_yr=e.get('far'),
                         p_astro=e.get('p_astro'), note=''))
    print(cat, len(ev), file=sys.stderr)

YEAR = 365.25 * 86400

# subthreshold candidate releases: zenodo record -> label
SST = {'16053640': 'GWTC-4.0-cand', '20348004': 'GWTC-5.0-cand'}
import os, tempfile
import numpy as np
import h5py
for rec, label in SST.items():
    files = get(f'https://zenodo.org/api/records/{rec}')['files']
    url = [f['links']['self'] for f in files if 'SearchSummaryTable' in f['key']][0]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'sst.hdf5')
        subprocess.run(['curl', '-sfL', '--retry', '4', '-o', p, url], check=True)
        with h5py.File(p) as f:
            n0 = len(rows)
            for pipe in f:
                d = f[pipe][:]
                names = d.dtype.names
                if 'end_time' in names:
                    t = d['end_time'] + d['end_time_ns'] * 1e-9
                elif 'gps_time' in names:
                    t = d['gps_time']
                else:
                    t = np.where(d['time_H1'] > 0, d['time_H1'], d['time_L1'])
                far = d['combined_far'] if 'combined_far' in names else d['far']
                pa = d['p_Astro'] if 'p_Astro' in names else 1 - d['p_Terrestrial']
                gw = [x.decode() if isinstance(x, bytes) else str(x) for x in d['gw_name']]
                for i in range(len(d)):
                    rows.append(dict(name=gw[i] or f'{label}-{pipe}-{i}', gps=float(t[i]), source=f'{label}:{pipe}',
                                     far_per_yr=float(far[i]) * YEAR, p_astro=float(pa[i]), note=''))
            print(label, len(rows) - n0, file=sys.stderr)
for run in a.runs.split(','):
    url = f'https://gracedb.ligo.org/api/superevents/?query=runid%3A+{run}&count=200&format=json'
    n = 0
    while url:
        d = get(url)
        for s in d['superevents']:
            if s.get('category') != 'Production':
                continue
            labels = s.get('labels') or []
            note = ';'.join(x for x in labels if x in ('RETRACTED', 'ADVNO', 'LOW_SIGNIF_LOCKED', 'SIGNIF_LOCKED'))
            rows.append(dict(name=s['superevent_id'], gps=s['t_0'], source=f'GraceDB-{run}',
                             far_per_yr=(s['far'] * YEAR if s.get('far') is not None else None), p_astro=None,
                             note=note + (f";gw_id={s['gw_id']}" if s.get('gw_id') else '')))
            n += 1
        url = d.get('links', {}).get('next')
        time.sleep(0.2)
    print(f'GraceDB {run}', n, file=sys.stderr)

rows.sort(key=lambda r: r['gps'])
with open(a.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['name', 'gps', 'source', 'far_per_yr', 'p_astro', 'note'])
    w.writeheader(); w.writerows(rows)
print(a.out, len(rows), file=sys.stderr)
