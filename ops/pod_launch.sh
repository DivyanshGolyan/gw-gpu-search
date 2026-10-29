#!/bin/bash
# Start a RunPod GPU pod with the shared network volume at /workspace, run one command, log nvidia-smi
# every 2 s, serve /work/out on https://POD-8000.proxy.runpod.net/, copy outputs to the volume, self-delete.
#
# Usage: pod_launch.sh [options] -- COMMAND...
#   --name NAME         pod name (default gw-job)
#   --dir DIR           local directory shipped to the pod as /work/job (scripts, small inputs; <~60 KB xz)
#   --gpu TYPE          default "NVIDIA A40"
#   --cpu FLAVOR        CPU-only pod instead (cpu3c|cpu3g|cpu3m|cpu5c|cpu5g|cpu5m: RAM 2/4/8 GB per vCPU); --min-vcpu sets
#                       its vCPU count, --min-ram is ignored. For CPU-bound stages (ranking) so no GPU is billed idle.
#   --volume ID         network volume id (default $GW_VOLUME_ID)
#   --dc DC             datacenter of that volume (default $GW_VOLUME_DC, else CA-MTL-1)
#   --max-hours H       hard cap; the pod deletes itself after H hours whatever happens (default 3)
#   --linger MIN        keep serving /work/out for MIN minutes after the command exits (default 15)
#   --min-vcpu N        minimum vCPUs for the pod (default 8; without it RunPod can hand out 1 vCPU)
#   --min-ram GB        minimum RAM in GB (default 40)
#   --disk GB           container (pod-local /work) disk in GB (default 40)
#   --no-volume         run without the volume (no venv; for smoke tests)
#   --preflight-only    run the local preflight and stop (no pod, no RUNPOD_KEY needed)
#   --skip-preflight    create the pod without the preflight (emergencies only; say why in your ledger line)
# Before any pod is created, bench/pod_preflight.py runs locally (about 10 s): syntax of every .py/.sh,
# argv parse of every python call, DIR/preflight.sh if present (put your gate-on-fake-result test there),
# and the self-destruct timers of this boot script; then, against bench/pod_env.json (the pod image, venv and volume
# as last recorded by a pod): commands resolve, imports exist, volume free space, GLIBC of shipped/PUT binaries,
# pyseobnr injection Nyquist, float32 self-check tolerances. Any failure stops the launch.
# Watching: after creating a pod this starts bench/pod_watch.py on this machine (one per machine, by a lock in /tmp). It polls
# every running pod every 2 min and appends ALERT lines (GPU < 70% over 10 min while filtering, GPU idle, job FAIL lines,
# /work filling, pod lingering after EXIT, gw-shared < 5 GB, logs unreachable, over 6 h, whole-life GPU % when a job
# ends) to budget/pod_alerts.log. The session that launched the pod must
# keep a Monitor on it:  tail -n 0 -F budget/pod_alerts.log   and act on every alert (fix, or stop the pod).
# COMMAND: one word is run as a shell line; several words keep their quoting (-- env W="a b" bash job.sh works).
# Needs RUNPOD_KEY in the environment. Prints the pod id; poll status.txt for "EXIT <rc>".
#
# On the pod the command runs in /work/job under bash, with /workspace/venv activated,
# LD_LIBRARY_PATH including /workspace/sys/lib (fftw), OMP_NUM_THREADS=1 and GW_OUT=/work/out.
# Write results to /work/out; they are also copied to /workspace/results/NAME-POD_ID/ on exit.
# Data on the volume: see /workspace/README.md.
set -euo pipefail
NAME=gw-job; DIR=; GPU="NVIDIA A40"; VOL=${GW_VOLUME_ID:-}; DC=${GW_VOLUME_DC:-CA-MTL-1}; HOURS=3; LINGER=15; NOVOL=0; VCPU=8; RAM=40; DISK=40; SKIPPF=0; PFONLY=0; CPU=
while [ $# -gt 0 ]; do
  case $1 in
    --name) NAME=$2; shift 2;; --dir) DIR=$2; shift 2;; --gpu) GPU=$2; shift 2;; --cpu) CPU=$2; shift 2;;
    --volume) VOL=$2; shift 2;; --dc) DC=$2; shift 2;; --max-hours) HOURS=$2; shift 2;;
    --linger) LINGER=$2; shift 2;; --min-vcpu) VCPU=$2; shift 2;; --min-ram) RAM=$2; shift 2;; --disk) DISK=$2; shift 2;; --no-volume) NOVOL=1; shift;; --skip-preflight) SKIPPF=1; shift;; --preflight-only) PFONLY=1; shift;; --) shift; break;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ $# -gt 0 ] || { echo "need -- COMMAND" >&2; exit 2; }
[ $PFONLY = 1 ] || [ -n "${RUNPOD_KEY:-}" ] || { echo "RUNPOD_KEY not set" >&2; exit 2; }
[ $PFONLY = 1 ] || [ $NOVOL = 1 ] || [ -n "$VOL" ] || { echo "need --volume or GW_VOLUME_ID" >&2; exit 2; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# one word is a shell line run as is; several words keep their argv (quoted), so -- env W="a b" bash job.sh stays one W
if [ $# = 1 ]; then printf '%s\n' "$1"; else printf '%q ' "$@"; echo; fi > "$TMP/cmd.sh"
cat > "$TMP/boot.sh" <<'EOF'
# GW_WORK / GW_VOL / GW_LOCAL_TEST let pod_preflight.py run this exact script locally with stubs.
W=${GW_WORK:-/work}; VOLD=${GW_VOL:-/workspace}
mkdir -p $W/out $W/job && cd $W
st() { echo "$(date -u +%FT%T.%3NZ) $*" >> $W/out/status.txt; }
bye() { sleep "$1"; runpodctl remove pod "$RUNPOD_POD_ID"; }
( bye "$JOB_MAX_S" ) &
st start
if env | grep -q '^JOB_0='; then
  for v in $(env | grep -o '^JOB_[0-9]*' | sort -t_ -k2 -n); do printf %s "${!v}"; done | base64 -d | tar xJ -C $W/job
fi
echo "$CMD_B64" | base64 -d > $W/cmd.sh
[ -n "${GW_LOCAL_TEST:-}" ] || (cd $W/out && python3 -m http.server 8000 >/dev/null 2>&1 &)
nvidia-smi > $W/out/nvidia-smi.txt 2>&1
echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null) (quota/period = usable vCPUs; nproc shows 1 when OMP_NUM_THREADS=1)" > $W/out/cpus.txt
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -l 2 > $W/out/nvsmi.csv 2>/dev/null &
if [ -d $VOLD/venv ]; then source $VOLD/venv/bin/activate; export LD_LIBRARY_PATH=$VOLD/sys/lib:${LD_LIBRARY_PATH:-}; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GW_OUT=$W/out
[ -d $VOLD ] && st "volume: $(df -h $VOLD 2>/dev/null | awk 'NR==2{print $3" used of "$2", "$5}')"
st "run: $(cat $W/cmd.sh)"
cd $W/job
bash $W/cmd.sh > $W/out/cmd.log 2>&1; rc=$?
st "EXIT $rc"
[ -d $VOLD ] && st "volume: $(df -h $VOLD 2>/dev/null | awk 'NR==2{print $3" used of "$2", "$5}')"
# what the local preflight checks jobs against (merge with: pod_preflight.py --refresh-env URL/pod_env.json)
python3 - "$VOLD" > $W/out/pod_env.json 2>/dev/null <<'PY' || true
import json, os, platform, re, shutil, sys, time
vol = sys.argv[1]; mods = set()
for p in sys.path:
    if os.path.isdir(p):
        mods.update(n for n in (f.split('.')[0] for f in os.listdir(p)) if n.isidentifier())
lib = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
cxx = re.findall(rb'GLIBCXX_(\d+\.\d+\.\d+)\x00', open(lib, 'rb').read()) if os.path.exists(lib) else []
du = shutil.disk_usage(vol) if os.path.isdir(vol) else None
json.dump(dict(as_of=time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()), glibc=platform.libc_ver()[1],
               glibcxx=max((v.decode() for v in cxx), key=lambda s: tuple(map(int, s.split('.'))), default=''),
               python='%d.%d' % sys.version_info[:2], **{'modules' if os.environ.get('VIRTUAL_ENV') else 'image_modules': sorted(mods)},
               volume=du and dict(id=os.environ.get('VOLUME_ID', ''), size_gb=round(du.total / 2**30), free_gb=round(du.free / 2**30, 1),
                                  as_of=time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()))), sys.stdout, indent=1)
PY
mkdir -p $VOLD/results 2>/dev/null
if [ -w $VOLD/results ]; then d=$VOLD/results/$JOB_NAME-$RUNPOD_POD_ID; mkdir -p "$d" && cp -r $W/out/. "$d"/ && st "copied to $d"; fi
st "deleting in $JOB_LINGER_S s"
bye "$JOB_LINGER_S"
EOF

if [ $SKIPPF = 1 ]; then
  echo "preflight SKIPPED (--skip-preflight)" >&2
else
  python3 "$(dirname "$0")/pod_preflight.py" --cmd "$TMP/cmd.sh" --boot "$TMP/boot.sh" ${DIR:+--dir "$DIR"} $([ $NOVOL = 1 ] && echo --no-volume) $([ -n "$CPU" ] && echo --cpu-pod) >&2 \
    || { echo "preflight failed: no pod created" >&2; exit 3; }
fi
[ $PFONLY = 1 ] && { echo "preflight only: no pod created" >&2; exit 0; }

python3 - "$TMP" "$NAME" "$GPU" "$VOL" "$DC" "$NOVOL" "$DIR" "$HOURS" "$LINGER" "$VCPU" "$RAM" "$DISK" "$CPU" <<'PY'
import json, sys, base64, io, tarfile, os
tmp, name, gpu, vol, dc, novol, d, hours, linger, vcpu, ram, disk, cpu = sys.argv[1:]
b64 = lambda p: base64.b64encode(open(os.path.join(tmp, p), 'rb').read()).decode()
env = dict(BOOT_B64=b64('boot.sh'), CMD_B64=b64('cmd.sh'), JOB_NAME=name, VOLUME_ID='' if novol == '1' else vol,
           JOB_MAX_S=str(int(float(hours) * 3600)), JOB_LINGER_S=str(int(float(linger) * 60)))
if d:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:xz') as t:
        skip = lambda ti: None if '__pycache__' in ti.name.split('/') or ti.name.endswith('.pyc') else ti
        for f in sorted(x for x in os.listdir(d) if x != "__pycache__"):
            t.add(os.path.join(d, f), arcname=f, filter=skip)
    s = base64.b64encode(buf.getvalue()).decode()
    if len(s) > int(os.environ.get('GW_MAX_PAYLOAD', 85000)):
        sys.exit(f'--dir payload is {len(s)} base64 bytes; RunPod rejects bodies over ~100 KB. Put big inputs on the volume.')
    for i in range(0, len(s), 60000):
        env[f'JOB_{i // 60000}'] = s[i:i + 60000]
b = dict(name=name, cloudType='SECURE', gpuTypeIds=[gpu], gpuCount=1,
         imageName='runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04',
         containerDiskInGb=int(disk), ports=['8000/http'], env=env, minVCPUPerGPU=int(vcpu), minRAMPerGPU=int(ram),
         dockerEntrypoint=['bash', '-c', 'echo "$BOOT_B64" | base64 -d > /boot.sh && bash /boot.sh'])
if cpu:
    for k in ('gpuTypeIds', 'gpuCount', 'minVCPUPerGPU', 'minRAMPerGPU'):
        b.pop(k)
    b.update(computeType='CPU', cpuFlavorIds=[cpu], vcpuCount=int(vcpu),
             imageName='runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04')
if novol == '1':
    b['volumeInGb'] = 0
else:
    b.update(networkVolumeId=vol, dataCenterIds=[dc], volumeMountPath='/workspace')
json.dump(b, open(os.path.join(tmp, 'body.json'), 'w'))
PY
R=$(curl -s -X POST -H "Authorization: Bearer $RUNPOD_KEY" -H 'Content-Type: application/json' -d @"$TMP/body.json" https://rest.runpod.io/v1/pods)
ID=$(printf '%s' "$R" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id') or '')" 2>/dev/null || true)
[ -n "$ID" ] || { echo "create failed: $R" >&2; exit 1; }
echo "$ID"
W_=$(cd "$(dirname "$0")" && pwd)
# pod_watch.py takes a lock and exits at once if a watcher already runs (pgrep -f also matches the caller's own shell)
(cd "$W_/.." && setsid nohup python3 "$W_/pod_watch.py" >> "$W_/../budget/pod_alerts.out" 2>&1 < /dev/null &) > /dev/null 2>&1  # must not hold a caller's $(...) open
echo "watch: keep a Monitor on: tail -n 0 -F $W_/../budget/pod_alerts.log" >&2
echo "status: https://$ID-8000.proxy.runpod.net/status.txt" >&2
echo "delete: curl -X DELETE -H \"Authorization: Bearer \$RUNPOD_KEY\" https://rest.runpod.io/v1/pods/$ID" >&2
