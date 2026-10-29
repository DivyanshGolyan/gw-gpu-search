#!/usr/bin/env python
"""Run provenance (review item R9): what code, configuration, bank and data-quality inputs produced a set of outputs,
and an append-only analysis-history log of every run.

record(...) returns a dict: md5 of every input file, the command line, md5 of each loaded module from this repo, and
one 'config_md5' over all of it, so two runs with equal config_md5 used identical inputs and code. fp_filter.py stores
it in timing.json and as the 'provenance' attribute of every peak file.

DQ in this pipeline is the GWOSC {IFO}_DATA science segments (make_pieces_o4a.py), fixed into each week's
datasets.txt and coinc_segments.txt, plus pycbc autogating with the argv's --autogating-* settings. There is no
CBC_CAT1/CAT2 veto file. So the DQ mask version is the md5 of those segment files plus the autogating flags.

CLI, for shell stages (rank_week.sh): provenance.py --log HISTORY.jsonl --stage NAME [--status S] [--out DIR] -- FILE...
"""
import argparse, datetime, hashlib, json, os, socket, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLACE_OPTS = {'--out', '--output', '--prep-cache', '--hdf-store', '--history'}


def md5(path, bs=1 << 22):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def code_md5s():
    """md5 of every loaded module whose file lies under this directory (fp_filter.py, kernels, ecc/...)."""
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if f and os.path.abspath(f).startswith(HERE + os.sep) and f.endswith('.py'):
            out[os.path.relpath(os.path.abspath(f), HERE)] = md5(f)
    main = os.path.abspath(sys.argv[0])
    if main.endswith('.py'):
        out[os.path.basename(main)] = md5(main)
    return dict(sorted(out.items()))


def record(inputs, argv=None, dq_files=(), dq_flags=()):
    """inputs: {role: path}; dq_files: segment files that define analysed time; dq_flags: gating/veto options."""
    p = dict(argv=list(sys.argv if argv is None else argv),
             inputs={k: dict(path=v, md5=md5(v)) for k, v in inputs.items() if v},
             code=code_md5s(),
             dq=dict(source='GWOSC IFO_DATA segments + pycbc autogating',
                     files={os.path.basename(f): md5(f) for f in dq_files if f}, flags=list(dq_flags)))
    p['dq']['version'] = hashlib.md5(json.dumps(p['dq'], sort_keys=True).encode()).hexdigest()[:12]
    # config_md5 ignores where things live (paths, output and cache dirs), only what they contain
    av = p['argv'][1:]
    av = [os.path.basename(x) if '/' in x else x for i, x in enumerate(av) if not (i and av[i - 1] in PLACE_OPTS)]
    key = dict(argv=av, inputs={k: v['md5'] for k, v in p['inputs'].items()}, code=p['code'], dq=p['dq']['version'])
    p['config_md5'] = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return p


def log(history, stage, status, out=None, prov=None, **extra):
    """Append one line to the analysis-history log (never rewritten)."""
    if not history:
        return
    e = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), host=socket.gethostname(),
             pod=os.environ.get('RUNPOD_POD_ID'), stage=stage, status=status, out=out,
             config_md5=(prov or {}).get('config_md5'), dq_version=((prov or {}).get('dq') or {}).get('version'), **extra)
    os.makedirs(os.path.dirname(os.path.abspath(history)), exist_ok=True)
    with open(history, 'a') as f:
        f.write(json.dumps(e, default=str) + '\n')


if __name__ == '__main__':
    split = sys.argv.index('--') if '--' in sys.argv else len(sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True); ap.add_argument('--stage', required=True)
    ap.add_argument('--status', default='ok'); ap.add_argument('--out')
    a = ap.parse_args(sys.argv[1:split])
    files = sys.argv[split + 1:]
    prov = record({os.path.basename(f): f for f in files}, argv=[a.stage] + files)
    log(a.log, a.stage, a.status, a.out, prov, inputs={k: v['md5'] for k, v in prov['inputs'].items()})
    print(prov['config_md5'])
