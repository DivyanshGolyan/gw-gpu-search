#!/usr/bin/env python3
"""Local preflight for pod_launch.sh: catch plumbing bugs on the CPU before a GPU pod is rented.

Checks, in order (any FAIL stops the launch):
  1. syntax   py_compile on every .py in --dir, bash -n on every .sh, on the command and on the boot script.
  2. argv     every `python[3] X.py ARGS` in the command (and in .sh files it runs, where the line has no
              unresolved $variables) is run locally up to its argparse call and stopped there. Modules that
              are not installed here (cupy, pycbc, torch, ...) are replaced by inert stubs, so the parse is
              reached without a GPU. A bad flag or a missing required argument fails.
  3. hook     if --dir has preflight.sh, it runs locally (GW_PREFLIGHT=1, GW_OUT=<temp dir>, cwd = a copy
              of --dir, 120 s limit). Put lane-specific checks there, above all: run your gate / report code
              on a tiny fake result file with the exact keys the real run writes.
  4. timers   the real boot script runs locally with stubbed runpodctl and nvidia-smi: a hung command must
              be killed by the --max-hours timer (scaled to 3 s), and a finished command must be followed by
              the linger timer (scaled to 1 s) and its output copied. This catches self-destruct bugs such
              as a bare `wait` that blocks on the timer itself.
Checks against the pod as recorded in pod_env.json (POD_ENV; every pod writes a fresh pod_env.json to its outputs,
merge it in with --refresh-env):
  5. command  every program in the command resolves (bash builtin, file in --dir, local or pod binary). A word that
              does not is exit 127 on the pod, e.g. a quoted argument split into two words.
  6. import   modules imported by --dir's .py files and by python snippets in its .sh files are in the pod venv,
              the stdlib, --dir, or a `pip install` in the job.
  7. volume   the network volume has >= MIN_VOLUME_FREE_GB free.
  8. glibc    shared objects in --dir, or in local files its scripts reference (wheels PUT to the pod), need no
              newer GLIBC / GLIBCXX than the pod image has.
  9. nyquist  pyseobnr (SEOBNRv5*) injections in referenced injection files: the (4,4) ringdown frequency must be
              below the Nyquist frequency of the job's --sample-rate, unless lmax_nyquist turns pyseobnr's check
              off (set in the file, or patched in by a .sh in --dir).
 10. float32  CHECK_TOL self-check tolerances on float32 quantities sit above float32 rounding noise.

Usage: pod_preflight.py --cmd CMD_SH --boot BOOT_SH [--dir DIR] [--no-volume]
       pod_preflight.py --refresh-env POD_ENV_JSON_OR_URL
"""
import argparse, ast, glob, json, os, re, shlex, shutil, subprocess, sys, tarfile, tempfile, time, zipfile

POD_ENV = os.environ.get('GW_POD_ENV') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pod_env.json')
MIN_VOLUME_FREE_GB = 2
GPU_MODULES = {'cupy', 'cupyx', 'torch', 'triton', 'pycuda', 'numba'}
POD_COMMANDS = {'nvidia-smi', 'runpodctl', 'python', 'python3', 'pip', 'pip3'}
# Summing 2^20 float32 terms carries ~sqrt(2^20) * 2^-24 = 6e-5 relative rounding error, so a tolerance on such a
# quantity below 1e-4 trips on noise (pilot 2026-10-28: 1.01e-5 and 1.14e-5 against 1e-5).
FLOAT32_TOL_FLOOR = 1e-4
FLOAT32_TOL_KEYS = ('snr_rel', 'phase_rad')
MTSUN_SI = 4.925490947641267e-06
# Re(M omega) of the (4,4,0) Kerr QNM (qnm package), prograde and retrograde, against remnant spin |a|
QNM44_A = (0, .1, .2, .3, .4, .5, .6, .7, .75, .8, .85, .9, .95, .99)
QNM44_PRO = (.8092, .8387, .8717, .9092, .9525, 1.0034, 1.065, 1.1427, 1.1906, 1.2475, 1.3179, 1.4104, 1.5486, 1.7731)
QNM44_RETRO = (.8092, .7826, .7585, .7365, .7162, .6974, .68, .6637, .656, .6485, .6413, .6343, .6275, .6221)

STUB_SITE = r'''
import sys, os, types, importlib.abc, importlib.machinery, argparse
class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Any()
    def __getattr__(self, k):
        if k.startswith('__') and k.endswith('__'): raise AttributeError(k)
        return _Any()
    def __iter__(self): return iter(())
    def __len__(self): return 0
    def __bool__(self): return False
    def __getitem__(self, k): return _Any()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __mro_entries__(self, bases): return (object,)
    def __index__(self): return 0
    def __int__(self): return 0
    def __float__(self): return 0.0
    def __str__(self): return 'stub'
for _op in ('add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow', 'and', 'or', 'lt', 'le', 'gt', 'ge', 'matmul'):
    setattr(_Any, f'__{_op}__', lambda self, *a: _Any()); setattr(_Any, f'__r{_op}__', lambda self, *a: _Any())
class _Mod(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith('__') and k.endswith('__'): raise AttributeError(k)
        return _Any()
class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        m = _Mod(spec.name); m.__path__ = []; m.__spec__ = spec; return m
    def exec_module(self, m): pass
class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        top = name.split('.')[0]
        if top.startswith('_') or top in sys.stdlib_module_names:
            return None                  # let optional stdlib probes (e.g. _winapi) fail as they should
        return importlib.machinery.ModuleSpec(name, _Loader(), is_package=True)
sys.meta_path.append(_Finder())          # consulted only when the real import fails
_inner = [False]
def _wrap(orig, outer):
    def f(self, *a, **k):
        if _inner[0]:
            return orig(self, *a, **k)   # parse_known_args called from inside parse_args
        _inner[0] = outer
        r = orig(self, *a, **k)          # exits 2 on a bad flag, as on the pod
        sys.stdout.flush(); print('PREFLIGHT_ARGV_OK', flush=True); os._exit(0)
    return f
argparse.ArgumentParser.parse_args = _wrap(argparse.ArgumentParser.parse_args, True)
argparse.ArgumentParser.parse_known_args = _wrap(argparse.ArgumentParser.parse_known_args, False)
'''

STUB_RUNPODCTL = '#!/bin/bash\necho "$(date +%s.%N) $*" >> "$GW_WORK/removed"\nkill -9 0\n'
STUB_NVSMI = '#!/bin/bash\nexit 0\n'

results = []


def report(kind, ok, msg):
    results.append((kind, ok, msg))
    print(f'  [{"ok" if ok is True else "WARN" if ok is None else "FAIL"}] {kind}: {msg}', flush=True)


def check_syntax(dirpath, cmd, boot):
    for f in [cmd, boot] + sorted(p for p in walk(dirpath) if p.endswith('.sh')):
        r = subprocess.run(['bash', '-n', f], capture_output=True, text=True)
        report('syntax', r.returncode == 0, f'bash -n {short(f, dirpath)}' + (f': {r.stderr.strip()}' if r.returncode else ''))
    for f in sorted(p for p in walk(dirpath) if p.endswith('.py')):
        r = subprocess.run([sys.executable, '-c', 'import sys; compile(open(sys.argv[1], "rb").read(), sys.argv[1], "exec")', f],
                           capture_output=True, text=True)   # in memory: writes no __pycache__ into --dir
        report('syntax', r.returncode == 0, f'py_compile {short(f, dirpath)}' + (f': {r.stderr.strip().splitlines()[-1]}' if r.returncode else ''))
        # undefined names (a NameError on the pod after the GPU is paid for, 2026-10-28 gmemprof); skipped if uvx is missing
        r = subprocess.run(['uvx', '-q', 'pyflakes', f], capture_output=True, text=True) if shutil.which('uvx') else None
        if r is not None and r.returncode in (0, 1):
            und = [l for l in r.stdout.splitlines() if 'undefined name' in l]
            report('syntax', not und, f'pyflakes {short(f, dirpath)}' + (f': {und[0].split(":", 1)[1].strip()}' if und else ''))


def walk(d):
    if not d:
        return []
    return [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs]


def short(f, d):
    return os.path.relpath(f, d) if d and f.startswith(d) else os.path.basename(f)


def simple_commands(text):
    """Token lists of each simple command in a shell script (split on ; && || | and newlines, quotes respected)."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            continue
        line = re.sub(r'(^|\s)\d?(>>|>|<|&>)(&\d+|\s*[^\s;&|]+)', ' ', line)   # drop redirections
        lx = shlex.shlex(line, posix=True, punctuation_chars=';&|')
        lx.whitespace_split = True; lx.commenters = '#'
        try:
            toks = list(lx)
        except ValueError:
            continue
        cur = []
        for t in toks + [';']:
            if t and set(t) <= set(';&|'):
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(t)
    return out


def python_calls(text):
    """(argv list, source text) for each `python[3] X.py ...` simple command in a shell script."""
    out = []
    for tok in simple_commands(text):
        src = ' '.join(tok)
        while tok and (re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', tok[0]) or tok[0] in ('timeout', 'exec', 'nohup', 'env')):
            tok = tok[2:] if tok[0] == 'timeout' else tok[1:]
        if len(tok) >= 2 and re.match(r'^(python3?|python3\.\d+)$', os.path.basename(tok[0])) and tok[1].endswith('.py'):
            keep = tok
            out.append((keep, src))
    return out


def shell_scripts_run(text):
    return [t[1] for t in simple_commands(text) if len(t) >= 2 and t[0] in ('bash', 'sh') and t[1].endswith('.sh')]


def check_argv(dirpath, cmd, work):
    stub = os.path.join(work, 'stubsite'); os.makedirs(stub)
    open(os.path.join(stub, 'sitecustomize.py'), 'w').write(STUB_SITE)
    run_dir = os.path.join(work, 'argv_job')
    shutil.copytree(dirpath, run_dir) if dirpath else os.makedirs(run_dir)
    text = open(cmd).read()
    calls = python_calls(text)
    for sh in shell_scripts_run(text):
        p = os.path.join(run_dir, sh)
        if os.path.exists(p):
            calls += [(t, s) for t, s in python_calls(open(p).read())]
    if not calls:
        report('argv', None, 'no python script calls found in the command; nothing to parse')
    for tok, src in calls:
        outd = os.path.join(work, 'argv_out'); os.makedirs(outd, exist_ok=True)
        tok = [t.replace('${GW_OUT}', outd).replace('$GW_OUT', outd) for t in tok]
        if any('$' in t for t in tok):
            report('argv', None, f'not checked (unresolved shell variable): {src[:100]}')
            continue
        script = os.path.join(run_dir, tok[1])
        if not os.path.exists(script):
            report('argv', False if not os.path.isabs(tok[1]) else None,
                   f'{tok[1]} not in --dir' + ('' if not os.path.isabs(tok[1]) else ' (absolute path; assumed to exist on the pod)'))
            continue
        env = dict(os.environ, PYTHONPATH=stub + os.pathsep + run_dir, GW_PREFLIGHT='1', GW_OUT=os.path.join(work, 'argv_out'))
        try:
            r = subprocess.run([sys.executable, script] + tok[2:], cwd=run_dir, env=env, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            report('argv', None, f'{tok[1]}: no argparse call within 30 s; not checked')
            continue
        tail = (r.stderr.strip().splitlines() or [''])[-1]
        if 'PREFLIGHT_ARGV_OK' in r.stdout:
            report('argv', True, f'{tok[1]} {" ".join(tok[2:])[:80]}')
        elif r.returncode == 2 and 'usage:' in r.stderr:
            report('argv', False, f'{tok[1]}: {tail}')
        elif re.search(r'\b(NameError|SyntaxError|UnboundLocalError|IndentationError)\b', r.stderr):
            report('argv', False, f'{tok[1]} crashed before parsing its arguments: {tail}')
        else:
            report('argv', None, f'{tok[1]} did not reach argparse locally (rc {r.returncode}: {tail[:120]}); not checked')


def check_hook(dirpath, work):
    hook = os.path.join(dirpath, 'preflight.sh') if dirpath else ''
    if not hook or not os.path.exists(hook):
        report('hook', None, 'no preflight.sh in --dir: gate/report code is untested. Add one that runs your gate on a fake result file.')
        return
    run_dir = os.path.join(work, 'hook_job'); shutil.copytree(dirpath, run_dir)
    out = os.path.join(work, 'hook_out'); os.makedirs(out)
    try:
        r = subprocess.run(['bash', 'preflight.sh'], cwd=run_dir, env=dict(os.environ, GW_PREFLIGHT='1', GW_OUT=out),
                           capture_output=True, text=True, timeout=120)
        msg = (r.stdout + r.stderr).strip().splitlines()
        report('hook', r.returncode == 0, f'preflight.sh exit {r.returncode}' + (': ' + ' | '.join(msg[-3:]) if msg else ''))
    except subprocess.TimeoutExpired:
        report('hook', False, 'preflight.sh took over 120 s')


def load_pod_env():
    if not os.path.exists(POD_ENV):
        report('env', None, f'no {POD_ENV}: import/volume/glibc checks skipped')
        return {}
    env = json.load(open(POD_ENV))
    age_h = (time.time() - time.mktime(time.strptime(env['as_of'], '%Y-%m-%dT%H:%MZ')) + time.timezone) / 3600
    if age_h > 24:
        report('env', None, f'{os.path.basename(POD_ENV)} is {age_h:.0f} h old; merge a recent pod\'s pod_env.json with --refresh-env')
    return env


def check_commands(dirpath, cmd):
    text = open(cmd).read()
    funcs = set(re.findall(r'^\s*([A-Za-z_][\w-]*)\s*\(\)', text, re.M))
    builtins = set(subprocess.run(['bash', '-c', 'compgen -b -k'], capture_output=True, text=True).stdout.split())
    bad = 0
    for tok in simple_commands(text):
        src = ' '.join(tok)
        while tok and (re.match(r'^[A-Za-z_]\w*=', tok[0]) or tok[0] in ('exec', 'nohup', 'time', 'env', 'timeout', 'nice')):
            tok = tok[2:] if tok[0] == 'timeout' else tok[1:]
            while tok and tok[0].startswith('-'):
                tok = tok[1:]
        if not tok or '$' in tok[0]:
            continue
        prog = tok[0]
        if '/' in prog:
            ok = os.path.isabs(prog) or os.path.exists(os.path.join(dirpath, prog))
        else:
            ok = prog in builtins or prog in funcs or prog in POD_COMMANDS or bool(shutil.which(prog))
        script = tok[1] if prog in ('bash', 'sh') and len(tok) > 1 and not tok[1].startswith('-') else ''
        if not ok:
            bad += 1
            report('command', False, f'`{prog}` is not a builtin, a file in --dir or a command on the pod, so the pod exits 127. '
                                     f'A quoted argument split into words looks like this: {src[:160]}')
        elif script and '$' not in script and not os.path.isabs(script) and not os.path.exists(os.path.join(dirpath, script)):
            bad += 1
            report('command', False, f'{script} is not in --dir: {src[:160]}')
    if not bad:
        report('command', True, 'every program in the command resolves')


def python_sources(dirpath):
    """(label, source) of each .py in --dir and each python snippet (-c string, heredoc) in its .sh files."""
    out = [(short(p, dirpath), open(p, errors='replace').read()) for p in sorted(walk(dirpath)) if p.endswith('.py')]
    for sh in sorted(p for p in walk(dirpath) if p.endswith('.sh')):
        text = open(sh, errors='replace').read()
        for m in re.finditer(r'python3?\b[^\n]*?<<-?\s*[\'"]?(\w+)[\'"]?[^\n]*\n(.*?)\n\1$', text, re.S | re.M):
            out.append((f'{short(sh, dirpath)} heredoc', m.group(2)))
        for tok in simple_commands(text):
            if tok and re.match(r'^python3?$', os.path.basename(tok[0])) and '-c' in tok[:-1]:
                out.append((f'{short(sh, dirpath)} -c', tok[tok.index('-c') + 1]))
    return out


def imports_of(source):
    """(module-level, in-function) top-level module names; imports inside try/except are optional and left out."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), set()
    need, lazy = set(), set()

    def visit(node, in_func):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.Try):
                for b in ch.handlers + ch.orelse + ch.finalbody:
                    visit(b, in_func)
                continue
            names = [a.name for a in ch.names] if isinstance(ch, ast.Import) else \
                [ch.module] if isinstance(ch, ast.ImportFrom) and ch.level == 0 and ch.module else []
            (lazy if in_func else need).update(n.split('.')[0] for n in names)
            visit(ch, in_func or isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))
    visit(tree, False)
    return need, lazy - need


def pip_installed(dirpath):
    """Module names that a .sh in --dir installs with pip (from wheel file names or requirement names)."""
    out = set()
    for sh in (p for p in walk(dirpath) if p.endswith('.sh')):
        for tok in simple_commands(open(sh, errors='replace').read()):
            if len(tok) > 2 and re.match(r'^pip3?$', os.path.basename(tok[0])) and tok[1] == 'install':
                skip = False
                for t in tok[2:]:
                    if skip or t.startswith('-'):
                        skip = t in ('-c', '-r', '-t', '--target', '--index-url', '-i', '--constraint', '--requirement')
                        continue
                    name = os.path.basename(t).split('-')[0] if t.endswith('.whl') else re.split(r'[<>=!~\[;]', t)[0]
                    out.add(name.lower().replace('-', '_'))
    return out


def check_imports(dirpath, env, use_volume):
    key = 'modules' if use_volume else 'image_modules'      # no volume, no venv: only the image's own python packages
    if not env.get(key):
        if env and not use_volume:
            report('import', None, 'no image_modules in pod_env.json (refresh from a --no-volume pod); imports not checked')
        return
    have = set(env[key]) | set(sys.stdlib_module_names) | {'__future__'} | pip_installed(dirpath)
    have |= {os.path.splitext(os.path.basename(p))[0] for p in walk(dirpath) if p.endswith('.py')}
    have |= {os.path.basename(os.path.dirname(p)) for p in walk(dirpath) if p.endswith('__init__.py')}
    missing = {}
    for label, src in python_sources(dirpath):
        need, lazy = imports_of(src)
        for n in need - have:
            missing.setdefault((n, True), []).append(label)
        for n in lazy - have:
            missing.setdefault((n, False), []).append(label)
    for (n, hard), where in sorted(missing.items()):
        report('import', False if hard else None, f'{n} is not in the pod {"venv" if use_volume else "image"} ({env["as_of"]}), imported by {", ".join(where[:3])}'
                                                  + ('' if hard else ' inside a function'))
    if not any(hard for _, hard in missing):
        report('import', True, 'every module-level import is in the pod venv, the stdlib, --dir or pip-installed by the job')


def check_volume(env, use_volume):
    v = env.get('volume')
    if not use_volume or not v:
        return
    ok = v['free_gb'] >= MIN_VOLUME_FREE_GB
    report('volume', ok, f'volume {v["id"]}: {v["free_gb"]:.1f} GB free of {v["size_gb"]} GB ({v["as_of"]})'
                         + ('' if ok else f'; under {MIN_VOLUME_FREE_GB} GB, so writes fail. Grow it first'))


def referenced_files(dirpath):
    """Files in --dir plus existing local files (/mnt, /home) named by absolute paths in its .sh files ($VAR=... expanded,
    globs): the inputs that put scripts PUT to the pod."""
    out = set(walk(dirpath))
    for sh in (p for p in walk(dirpath) if p.endswith('.sh')):
        text = open(sh, errors='replace').read()
        var = dict(re.findall(r'(?:^|[\s;])([A-Za-z_]\w*)=([^\s;\'"$`]+)', text))
        for raw in re.findall(r'[$\w{}./*-]*/[$\w{}./*-]+', text):
            p = re.sub(r'\$\{?(\w+)\}?', lambda m: var.get(m.group(1), m.group(0)), raw)
            if p.startswith(('/mnt/', '/home/')) and '$' not in p:
                out.update(f for f in glob.glob(p) if os.path.isfile(f) and os.path.getsize(f) < 200e6)
    return sorted(out)


def _elf_blobs(path):
    """(name, bytes) of each ELF file in path: the file itself, or the members of a wheel/zip/tar."""
    with open(path, 'rb') as f:
        head = f.read(262)
    if head[:4] == b'\x7fELF':
        yield os.path.basename(path), open(path, 'rb').read()
    elif head[:2] == b'PK':
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                b = z.read(n)
                if b[:4] == b'\x7fELF':
                    yield f'{os.path.basename(path)}:{n}', b
    elif head[257:262] == b'ustar':
        with tarfile.open(path) as t:
            for m in t:
                b = t.extractfile(m).read() if m.isfile() else b''
                if b[:4] == b'\x7fELF':
                    yield f'{os.path.basename(path)}:{m.name}', b


def check_glibc(refs, env):
    if not env.get('glibc'):
        return
    pod = {'GLIBC': env['glibc'], 'GLIBCXX': env.get('glibcxx', '0')}
    ver = lambda s: tuple(map(int, s.split('.')))
    n = bad = 0
    for p in refs:
        try:
            blobs = list(_elf_blobs(p))
        except (OSError, zipfile.BadZipFile, tarfile.TarError):
            continue
        for name, b in blobs:
            n += 1
            for lib, have in pod.items():
                vs = [v.decode() for v in re.findall(lib.encode() + rb'_(\d+(?:\.\d+)+)\x00', b)]
                top = max(vs, key=ver, default=None)
                if top and ver(top) > ver(have):
                    bad += 1
                    report('glibc', False, f'{name} ({p}) needs {lib}_{top}; the pod image has {have}, so it fails to load there')
    if n and not bad:
        report('glibc', True, f'{n} shared objects need no newer GLIBC/GLIBCXX than the pod\'s {pod["GLIBC"]}/{pod["GLIBCXX"]}')


def _remnant(m1, m2, s1, s2):
    """Final mass (fraction of m1+m2) and spin for aligned spins: Rezzolla et al. 2008 spin, nonspinning radiated energy."""
    import numpy as np
    q = np.minimum(m1, m2) / np.maximum(m1, m2); nu = m1 * m2 / (m1 + m2) ** 2
    sa = np.where(m1 >= m2, s1, s2); sb = np.where(m1 >= m2, s2, s1)
    at = (sa + sb * q ** 2) / (1 + q ** 2)
    af = at + at * nu * (-0.1229 * at + 0.4537 * nu - 2.8904) + nu * (2 * 3 ** 0.5 - 3.5171 * nu + 2.5763 * nu ** 2)
    return 1 - (1 - (8 / 9) ** 0.5) * nu - 0.498 * nu ** 2, af


def f44_hz(m1, m2, s1, s2):
    """(4,4,0) ringdown frequency in Hz for detector-frame masses (solar) and aligned spins."""
    import numpy as np
    mf, af = _remnant(m1, m2, s1, s2)
    w = np.where(af >= 0, np.interp(np.abs(af), QNM44_A, QNM44_PRO), np.interp(np.abs(af), QNM44_A, QNM44_RETRO))
    return w / (2 * np.pi * mf * (m1 + m2) * MTSUN_SI)


def check_nyquist(dirpath, refs):
    files = [p for p in refs if p.endswith(('.hdf', '.h5', '.hdf5'))]
    if not files:
        return
    try:
        import numpy as np, h5py
    except ImportError:
        report('nyquist', None, 'numpy/h5py missing locally; injection Nyquist check skipped')
        return
    rates = set()
    for p in walk(dirpath):
        if p.endswith(('.txt', '.sh')):
            rates.update(float(r) for r in re.findall(r'--sample-rate[ =]+(\d+(?:\.\d+)?)', open(p, errors='replace').read()))
    patched = [short(p, dirpath) for p in walk(dirpath) if p.endswith('.sh') and 'lmax_nyquist' in open(p, errors='replace').read()]
    for p in files:
        try:
            with h5py.File(p, 'r') as f:
                if 'static_args' not in f.attrs or 'mass1' not in f:
                    continue
                appr = f['approximant'][:] if 'approximant' in f else np.full(len(f['mass1']), f.attrs.get('approximant', ''))
                appr = np.array([a.decode() if isinstance(a, bytes) else str(a) for a in appr])
                sel = np.char.startswith(appr, 'SEOBNRv5')
                if not sel.any():
                    continue
                if 'lmax_nyquist' in f.attrs or 'lmax_nyquist' in f:
                    report('nyquist', True, f'{os.path.basename(p)}: lmax_nyquist set in the file')
                    continue
                m1, m2, s1, s2 = (f[k][:][sel] if k in f else np.zeros(sel.sum()) for k in ('mass1', 'mass2', 'spin1z', 'spin2z'))
        except OSError:
            continue
        if not rates:
            report('nyquist', None, f'{os.path.basename(p)}: no --sample-rate in --dir; not checked')
            continue
        fny = min(rates) / 2
        fr = f44_hz(m1, m2, s1, s2); nbad = int((fr > fny).sum())
        if not nbad:
            report('nyquist', True, f'{os.path.basename(p)}: all {sel.sum()} pyseobnr injections ring down below {fny:.0f} Hz')
        elif patched:
            report('nyquist', True, f'{os.path.basename(p)}: {nbad} of {sel.sum()} ring down above {fny:.0f} Hz; '
                                    f'lmax_nyquist patched in by {", ".join(patched)}')
        else:
            i = int(np.argmax(fr))
            report('nyquist', False, f'{os.path.basename(p)}: {nbad} of {sel.sum()} pyseobnr injections have a (4,4) ringdown above '
                                     f'Nyquist {fny:.0f} Hz (worst {fr[i]:.0f} Hz at m1+m2 {m1[i] + m2[i]:.1f}), and pyseobnr raises on the '
                                     f'first. Inject from a copy with the static arg lmax_nyquist=1')


def check_float32_tol(dirpath):
    for p in (p for p in walk(dirpath) if p.endswith('.py')):
        try:
            tree = ast.parse(open(p, errors='replace').read())
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CHECK_TOL' for t in node.targets):
                try:
                    tol = ast.literal_eval(node.value) if isinstance(node.value, ast.Dict) else \
                        {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
                except (ValueError, AttributeError):
                    continue
                low = {k: v for k, v in tol.items() if k in FLOAT32_TOL_KEYS and v < FLOAT32_TOL_FLOOR}
                shown = ', '.join(f'{k}={tol[k]:g}' for k in FLOAT32_TOL_KEYS if k in tol)
                report('float32', not low, f'{short(p, dirpath)} CHECK_TOL {shown}' + (
                    f': below the float32 rounding floor {FLOAT32_TOL_FLOOR:g}, so the self-check can abort the run on noise' if low else ''))


def check_gpu_used(dirpath):
    """A GPU pod whose job imports no GPU module bills an idle GPU for its whole run."""
    used = set()
    for _, src in python_sources(dirpath):
        need, lazy = imports_of(src)
        used |= (need | lazy) & GPU_MODULES
    if dirpath and not used:
        report('gpu', None, 'nothing in --dir imports ' + '/'.join(sorted(GPU_MODULES)) + ': the GPU would sit idle. Use --cpu')


def refresh_env(src):
    """Merge a pod's pod_env.json (path or URL) into POD_ENV."""
    if re.match(r'^https?://', src):
        import urllib.request
        snap = json.load(urllib.request.urlopen(src, timeout=30))
    else:
        snap = json.load(open(src))
    env = json.load(open(POD_ENV)) if os.path.exists(POD_ENV) else {}
    env.update({k: v for k, v in snap.items() if v})
    json.dump(env, open(POD_ENV, 'w'), indent=1)
    print(f'{POD_ENV}: as_of {env["as_of"]}, {len(env.get("modules", []))} modules, volume {env.get("volume")}')


def run_boot(boot, work, tag, command, max_s, linger_s):
    w = os.path.join(work, tag); os.makedirs(w)
    bin_ = os.path.join(w, 'bin'); os.makedirs(bin_)
    for name, body in (('runpodctl', STUB_RUNPODCTL), ('nvidia-smi', STUB_NVSMI)):
        p = os.path.join(bin_, name); open(p, 'w').write(body); os.chmod(p, 0o755)
    import base64
    env = dict(os.environ, PATH=bin_ + os.pathsep + os.environ['PATH'], GW_WORK=w, GW_LOCAL_TEST='1', GW_VOL=os.path.join(w, 'vol'),
               CMD_B64=base64.b64encode(command.encode()).decode(), JOB_NAME='preflight', RUNPOD_POD_ID='local',
               JOB_MAX_S=str(max_s), JOB_LINGER_S=str(linger_s))
    env = {k: v for k, v in env.items() if not re.match(r'^JOB_\d+$', k)}
    t0 = time.time()
    p = subprocess.Popen(['bash', boot], env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
    removed = os.path.join(w, 'removed')
    t_rm = float(open(removed).read().split()[0]) - t0 if os.path.exists(removed) else None
    status = open(os.path.join(w, 'out', 'status.txt')).read() if os.path.exists(os.path.join(w, 'out', 'status.txt')) else ''
    return w, t_rm, status


def check_timers(boot, work):
    w, t_rm, status = run_boot(boot, work, 'timer_hung', 'sleep 60\n', 3, 1)
    ok = t_rm is not None and 2.5 <= t_rm <= 6 and 'EXIT' not in status
    report('timers', ok, f'hung command, cap 3 s: pod removal called at {t_rm if t_rm is None else round(t_rm, 1)} s' + ('' if ok else f'; status: {status.strip()[-200:]}'))
    w, t_rm, status = run_boot(boot, work, 'timer_done', 'echo hi > "$GW_OUT/probe.txt"\n', 30, 1)
    copied = os.path.exists(os.path.join(w, 'vol', 'results', 'preflight-local', 'probe.txt'))
    ok = t_rm is not None and t_rm <= 8 and 'EXIT 0' in status and copied
    report('timers', ok, f'finished command, linger 1 s: EXIT 0 {"seen" if "EXIT 0" in status else "MISSING"}, outputs copied {copied}, '
                         f'removal at {t_rm if t_rm is None else round(t_rm, 1)} s' + ('' if ok else f'; status: {status.strip()[-200:]}'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cmd'); ap.add_argument('--boot'); ap.add_argument('--dir', default='')
    ap.add_argument('--no-volume', action='store_true'); ap.add_argument('--cpu-pod', action='store_true')
    ap.add_argument('--refresh-env', metavar='POD_ENV_JSON_OR_URL')
    a = ap.parse_args()
    if a.refresh_env:
        return refresh_env(a.refresh_env)
    if not (a.cmd and a.boot):
        ap.error('--cmd and --boot are required')
    d = os.path.abspath(a.dir) if a.dir else ''
    t0 = time.time()
    print('preflight (local, no pod yet):', flush=True)
    with tempfile.TemporaryDirectory() as work:
        check_syntax(d, a.cmd, a.boot)
        if not any(ok is False for _, ok, _ in results):
            env = load_pod_env()
            refs = referenced_files(d) if d else []
            check_commands(d, a.cmd)
            check_imports(d, env, not a.no_volume)
            check_volume(env, not a.no_volume)
            check_glibc(refs, env)
            check_nyquist(d, refs)
            check_float32_tol(d)
            if not a.cpu_pod:
                check_gpu_used(d)
            check_argv(d, a.cmd, work)
            check_hook(d, work)
            check_timers(a.boot, work)
    n_fail = sum(ok is False for _, ok, _ in results); n_warn = sum(ok is None for _, ok, _ in results)
    print(f'preflight: {n_fail} failed, {n_warn} warnings, {time.time() - t0:.1f} s', flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
