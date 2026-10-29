#!/usr/bin/env python3
"""Watch every running RunPod pod and flag waste without anyone asking.

Every --every seconds (default 120) it lists the account's pods (RunPod REST API, RUNPOD_KEY via curl) and, for
pods started by pod_launch.sh, reads their own logs on https://POD-8000.proxy.runpod.net/: nvsmi.csv (GPU use every
2 s), status.txt (EXIT, volume line), job_status.txt (job's own "volume N GB free" lines). It raises an ALERT when:
  gpu-low     GPU mean < --gpu-min % (default 70) over the last 10 min, while the job is filtering (job_status.txt has a
              "s filter" line and no "all weeks filtered" line); jobs without those markers fall back to pod age > 10 min
  gpu-idle    GPU at 0% for the whole last 5 min, same phase rule: nothing is using the GPU
  job-fail    a FAIL / Traceback / No space left line in job_status.txt
  work-fill   pod-local /work projected to drop under 2 GB free within 30 min (slope of mem.txt work_free_gb)
  ended       one line per pod after EXIT: GPU mean over its whole logged life (boot and waits included)
  lingering   the job printed EXIT and the pod is still billed more than linger + 5 min later
  work-low    pod-local /work free space under --work-min GB (default 8), from the job's mem.txt
  volume-low  gw-shared free space under --vol-min GB (default 5), from the newest pod log line
  no-logs     a running pod older than 10 min whose :8000 logs are unreachable (not started by pod_launch.sh?)
  over-cap    pod running longer than --max-hours (default 6)
Each alert carries the pod's last job_status line and repeats at most every 30 min per pod and kind; that de-duplication
is seeded from --log, so a restart does not repeat recent alerts. Alerts go to stdout and are appended to --log (default
budget/pod_alerts.log, shared by all lanes). Only one watcher runs (lock file next to the log): sessions tail the log
(`tail -n 0 -F budget/pod_alerts.log`) instead of starting their own. Started automatically by pod_launch.sh;
a session that launches pods must keep a Monitor on its output or on the log (see bench/pod_launch.sh header).
Usage: pod_watch.py [--every 120] [--once] [--log PATH]
"""
import argparse, json, os, re, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--every', type=int, default=120)
ap.add_argument('--once', action='store_true')
ap.add_argument('--gpu-min', type=float, default=70)
ap.add_argument('--vol-min', type=float, default=5)
ap.add_argument('--work-min', type=float, default=8)
ap.add_argument('--max-hours', type=float, default=6)
ap.add_argument('--log', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'budget', 'pod_alerts.log'))
ap.add_argument('--duration', type=float, default=0, help='exit after this many hours (0 = run until no pods for 30 min)')
a = ap.parse_args()
if not a.once:
    import fcntl
    lockf = open('/tmp/pod_watch.lock', 'a')    # local disk: flock on the project share (network mount) never blocks
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f'another pod_watch.py holds /tmp/pod_watch.lock: tail -n 0 -F {a.log} instead', file=sys.stderr)
        sys.exit(0)


def curl(url, auth=False, timeout=20):
    cmd = ['curl', '-sf', '--max-time', str(timeout), url]
    if auth:
        cmd[1:1] = ['-H', 'Authorization: Bearer ' + os.environ['RUNPOD_KEY']]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


last = {}
try:
    for line_ in open(a.log).read().splitlines()[-2000:]:
        m_ = re.match(r'(\S+)Z ALERT (\S+) (\S+) ', line_)
        if m_:
            last[(m_.group(3), m_.group(2))] = time.mktime(time.strptime(m_.group(1), '%Y-%m-%dT%H:%M:%S')) - time.timezone
except OSError:
    pass


def alert(pod, kind, msg):
    k = (pod.split()[0], kind)
    if time.time() - last.get(k, 0) < 1800:
        return
    last[k] = time.time()
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ALERT {kind} {pod}: {msg}" + (f' | last: {lastline.get(pod.split()[0], "")}' if lastline.get(pod.split()[0]) else '')
    print(line, flush=True)
    try:
        with open(a.log, 'a') as f:
            f.write(line + '\n')
    except OSError:
        pass


def check(p):
    pid, name = p['id'], p.get('name', '')
    tag = f'{pid} ({name}, ${p.get("costPerHr")}/h)'
    base = f'https://{pid}-8000.proxy.runpod.net'
    if name == 'gw-hub':    # the Vast relay (job_hub) serves its own tree, not /work/out: check it is up and has disk
        df = curl(base + '/f/hub_df.txt')
        if df is None:
            if time.time() - first_seen.setdefault(pid, time.time()) > 300:
                alert(tag, 'hub-down', 'relay unreachable for 5+ min: Vast pushes and lane pulls are failing')
            return
        first_seen.pop(pid, None); free = int((df.split() or ['99'])[0].rstrip('G') or 99)
        if free < 8:
            alert(tag, 'hub-disk', f'relay has {free} GB free (uploads refused under 4): the lane is not pulling fast enough')
        return
    st = curl(base + '/status.txt')
    if st is None:
        started = first_seen.setdefault(pid, time.time())
        if time.time() - started > 600:
            alert(tag, 'no-logs', 'logs on :8000 unreachable for 10+ min while the pod is RUNNING: its server died or it was not started by pod_launch.sh; check or stop it')
        return
    js = curl(base + '/job_status.txt') or ''
    jl = [l for l in js.splitlines() if l.strip()]
    if jl:
        lastline[pid] = jl[-1][:160]
    bad = [l for l in jl if re.search(r'FAIL|Traceback|No space left', l)]
    if bad:
        alert(tag, 'job-fail', bad[-1][:200])
    m = re.match(r'(\S+)Z start', st)
    t0 = time.mktime(time.strptime(m.group(1)[:19], '%Y-%m-%dT%H:%M:%S')) - time.timezone if m else time.time()
    age = time.time() - t0
    if age > a.max_hours * 3600:
        alert(tag, 'over-cap', f'running {age / 3600:.1f} h')
    ex = re.search(r'(\S+)Z EXIT (\d+)', st)
    nv = curl(base + '/nvsmi.csv', timeout=40)
    u_all = []
    for line in (nv or '').strip().splitlines():
        f = line.split(',')
        if len(f) >= 2 and f[1].strip().rstrip('%').strip().isdigit():
            u_all.append(int(f[1].strip().rstrip('%').strip()))
    if ex:
        if pid not in ended and u_all:
            ended.add(pid)
            alert(tag, 'ended', f'job exited {ex.group(2)}; GPU mean {sum(u_all) / len(u_all):.0f}% over its whole logged life ({len(u_all) * 2 / 60:.0f} min, boot and waits included)')
        te = time.mktime(time.strptime(ex.group(1)[:19], '%Y-%m-%dT%H:%M:%S')) - time.timezone
        ling = re.search(r'deleting in (\d+) s', st)
        lim = (int(ling.group(1)) if ling else 900) + 300
        if time.time() - te > lim:
            alert(tag, 'lingering', f'job exited {ex.group(2)} {int((time.time() - te) / 60)} min ago and the pod is still billed: delete it')
        return
    # judge GPU use only while the job filters: after its first "s filter" line, before "all weeks filtered";
    # jobs that print neither fall back to the pod-age rule
    marked = any(' s filter' in l or 'all weeks filtered' in l or 'waiting for inputs' in l for l in jl)
    fl = [l for l in jl if ' s filter' in l]
    t_fl = time.mktime(time.strptime(fl[0][:19], '%Y-%m-%dT%H:%M:%S')) - time.timezone if fl and re.match(r'\d{4}-\d\d-\d\dT', fl[0]) else None
    # the 10-min window must lie wholly after the first finished group, or it still holds the start-up download wait
    filtering = t_fl is not None and time.time() - t_fl >= 600 and not any('all weeks filtered' in l for l in jl)
    if (filtering if marked else age > 600) and len(u_all) >= 100:
        u = u_all[-300:]                     # 2 s samples: last 10 min
        mean = sum(u) / len(u)
        if all(x == 0 for x in u[-150:]):
            alert(tag, 'gpu-idle', 'GPU at 0% for the last 5 min while the job is filtering')
        elif mean < a.gpu_min:
            z = 100 * sum(x == 0 for x in u) / len(u)
            alert(tag, 'gpu-low', f'GPU mean {mean:.0f}% over 10 min ({z:.0f}% of samples at 0%), threshold {a.gpu_min:.0f}%')
    mt = curl(base + '/mem.txt') or ''
    wf = [(l.split()[0], int(x)) for l in mt.splitlines() for x in re.findall(r'work_free_gb (\d+)', l)]
    if wf and wf[-1][1] < a.work_min:
        alert(tag, 'work-low', f'pod-local /work {wf[-1][1]} GB free (< {a.work_min:.0f} GB)')
    if len(wf) >= 40:                         # 15 s samples: slope over the last 10 min
        drop = wf[-40][1] - wf[-1][1]
        if drop > 0 and (wf[-1][1] - 2) / (drop / 10) < 30:
            alert(tag, 'work-fill', f'/work fell {drop} GB in 10 min to {wf[-1][1]} GB: under 2 GB in ~{(wf[-1][1] - 2) / (drop / 10):.0f} min; look for a leak')
    v = re.findall(r'volume (\d+(?:\.\d+)?) GB free', js) or re.findall(r'(\d+)G used of (\d+)G', st)
    if v:
        x = v[-1]
        free = float(x) if isinstance(x, str) else float(x[1]) - float(x[0])
        if free < a.vol_min:
            alert(tag, 'volume-low', f'gw-shared {free:.0f} GB free (< {a.vol_min:.0f} GB)')


first_seen = {}; lastline = {}; ended = set()
t_start = time.time(); t_nopods = None
while True:
    r = curl('https://rest.runpod.io/v1/pods', auth=True)
    pods = [p for p in (json.loads(r) if r else []) if p.get('desiredStatus') == 'RUNNING']
    for p in pods:
        try:
            check(p)
        except Exception as e:
            print(f'watch error {p.get("id")}: {e}', file=sys.stderr)
    if a.once:
        break
    t_nopods = None if pods else (t_nopods or time.time())
    if t_nopods and time.time() - t_nopods > 1800 and not a.duration:
        break
    if a.duration and time.time() - t_start > a.duration * 3600:
        break
    time.sleep(a.every)
