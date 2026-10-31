#!/bin/bash
# One-shot Vast box: A/B of fp_filter v9 (seg_pipe.py overlap) against the frozen v8 on two O4a wk09 groups.
# Runs v8, v9, v8, v9 on the same downloaded data (first run also warms the kernel caches), then prints one RESULT
# block (bitwise peak comparison, counters, timings, GPU busy) to the container log and destroys the box.
# Env: BRANCH (git branch with v9), MAXMIN (kill timer, minutes; default 70).
set -uo pipefail
mkdir -p /work/out; touch /work/out/log; (tail -n +1 -F /work/out/log > /proc/1/fd/1 2>/dev/null &); exec >> /work/out/log 2>&1
say() { echo "AB $(date -u +%T) $*"; }
bye() { say "destroying box"; curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" -H "Authorization: Bearer $CONTAINER_API_KEY"; sleep 600; }
( sleep $(( ${MAXMIN:-70} * 60 )); say "KILL TIMER"; bye ) &
trap 'say "FAILED at line $LINENO"; bye' ERR
set -e
cd /work
apt-get update -qq > /dev/null && apt-get install -y -qq git curl unzip libgsl-dev swig > /dev/null
git clone -q --depth 5 -b "$BRANCH" https://github.com/divyanshgolyan/gw-gpu-search r
cd r; git fetch -q --depth 1 origin main; git show FETCH_HEAD:pipeline/fp_filter.py > pipeline/fp_filter_v8.py
[ "$(md5sum < pipeline/fp_filter_v8.py | cut -c1-32)" = 436c31dc879cc3a4d874dccf5153d700 ] || { say "v8 md5 wrong"; false; }
say "code $(git rev-parse --short HEAD); v9 $(md5sum < pipeline/fp_filter.py | cut -c1-8)"
# GWOSC's chain ends at emSign Root CA - G1, which certifi 2024.2.2 still carries
pip download -q --no-deps certifi==2024.2.2 -d /work/cert && (cd /work/cert && unzip -q -o certifi-*.whl)
export REQUESTS_CA_BUNDLE=/work/cert/certifi/cacert.pem SSL_CERT_FILE=/work/cert/certifi/cacert.pem
pip install -q gwosc==0.8.3 h5py==3.16.0 > /dev/null
# data first (26 lines of wk09 = 2 groups of 13), in the background while the environment builds
mkdir -p /work/stores /work/gwdata /work/grp; head -26 data/segments/o4a/wk09/datasets.txt > /work/grp/all.txt
split -l 13 -d /work/grp/all.txt /work/grp/g; T0=$SECONDS
( awk '{print $1, $2, $3, $4}' /work/grp/all.txt | xargs -P 10 -L 1 bash -c 'for k in 1 2 3; do python3 pipeline/chunk_prep.py $0 $1 $2 $3 --cache /work/gwdata > /dev/null && exit 0; sleep 10; done; echo "DL FAIL $0 $1"' > /work/out/dl.txt 2>&1; echo done >> /work/out/dl.txt ) &
grep -v -E '^(torch|torchvision|torchaudio|triton|nvidia-|filelock)' environment/pip_freeze_pod.txt > /work/cons.txt || true
grep -v -E '^(torch|triton|--extra-index-url|pyseobnr)' environment/requirements.txt > /work/req.txt
pip install -q -r /work/req.txt -c /work/cons.txt > /work/out/pip.txt 2>&1 || { tail -20 /work/out/pip.txt; false; }
pip install -q "setuptools<70" wheel numpy==2.2.6 > /dev/null; pip install -q pygsl-lite --no-build-isolation > /dev/null 2>&1 || true
pip install -q pyseobnr==0.3.7 "filelock>=3.16" > /work/out/pip2.txt 2>&1 || { tail -20 /work/out/pip2.txt; false; }
python3 -c "import pycbc, cupy, pyseobnr, torch, triton; print('env', torch.__version__, cupy.__version__, triton.__version__, torch.cuda.get_device_name(0))"
say "env ready after $((SECONDS-T0)) s"
python3 tests/test_scale_gpu.py > /work/out/scale.txt 2>&1 || true; cat /work/out/scale.txt
until grep -q '^done' /work/out/dl.txt; do sleep 5; done
say "data ready after $((SECONDS-T0)) s: $(du -sh /work/stores | cut -f1); $(grep -c 'DL FAIL' /work/out/dl.txt) failed"
grep -q 'DL FAIL' /work/out/dl.txt && { cat /work/out/dl.txt; false; }
rm -rf /work/gwdata
mkdir -p /work/job && cd /work/job && cp /work/r/data/inspiral_argv_H1_000.txt . && python3 /work/r/ops/setup_job.py /work/r/data/banks/bank_v2.hdf /work/stores > /dev/null
sed -i 's/--snr-threshold [0-9.]*/--snr-threshold 4/; s/--keep-loudest-num [0-9]*/--keep-loudest-num 30/; s/--segment-start-pad [0-9]*/--segment-start-pad 48/' argv_base.txt
grep -q -- '--low-frequency-cutoff 20 ' argv_base.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 CUPY_CACHE_DIR=/work/cupy
FLAGS=(--tile 512 --prep gpu --prep-procs 4 --gen-threads 6 --gemm-cfg 64,64,4,2 --exact-snr 5.5 --fuse-q --sg-batch --sg-kernel
       --exact-stage regen --consolidate --consolidate-exempt-snr 5 --pre-consolidate
       --bank2 /work/r/data/banks/vt_45_100_mm0.85.hdf --approximant2 SEOBNRv5EHM_ecc --eps2 0.02 --history /work/out/hist.jsonl)
SG=(--sgchisq-snr-threshold 6 --sgchisq-locations "mtotal>30:20-15,20-30,20-45,20-60,20-75,20-90,20-105,20-120")
for R in v8a v9a v8b v9b; do
  F=/work/r/pipeline/fp_filter.py; [ ${R:0:2} = v8 ] && F=/work/r/pipeline/fp_filter_v8.py
  QD=/work/q_$R; mkdir -p $QD; n=0
  for g in /work/grp/g??; do n=$((n+1)); echo "$g /work/res/$R/grp_0$((n-1))" > $QD/$(printf %04d $n).job; done; touch $QD/END
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 > /work/out/smi_$R.txt & SMI=$!
  t=$SECONDS; python3 $F --queue $QD --bank /work/r/data/banks/bank_v2.hdf --approximant IMRPhenomD --argv argv_base.txt "${FLAGS[@]}" -- "${SG[@]}" > /work/out/filter_$R.log 2>&1 \
    || { kill $SMI; tail -30 /work/out/filter_$R.log; false; }
  kill $SMI; say "$R done in $((SECONDS-t)) s"
done
cd /work/r
{
echo "=== RESULT BEGIN"
cat /work/out/scale.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for R in v8a v9a v8b v9b; do
  python3 - $R <<'PY'
import glob, json, sys
import numpy as np
R = sys.argv[1]
u = np.array([float(l.split(',')[0]) for l in open(f'/work/out/smi_{R}.txt') if l.strip() and l.split(',')[0].strip().replace('.', '').isdigit()])
ts = [json.load(open(p)) for p in sorted(glob.glob(f'/work/res/{R}/grp_*/timing.json'))]
keys = ('total_s', 'basis_fft_s', 'gemm_s', 'rebuilt_z_s', 'refine_chisq_s', 'rf_kernel_s', 'select_s', 'drain_s', 'templates_s', 'svd_s', 'gpu_peak_gib')
print(R, f'smi n={len(u)} mean {u.mean():.1f} median {np.median(u):.0f} zero {np.mean(u == 0):.3f}',
      ' | '.join(' '.join(f'{k}={t.get(k, 0):.1f}' for k in keys) for t in ts))
PY
done
for p in "v8a v8b" "v8a v9a" "v8b v9b" "v9a v9b"; do set -- $p; echo "--- $1 vs $2"; python3 tests/cmp_runs.py /work/res/$1 /work/res/$2; done
grep -h -E "^bank|^tile" /work/out/filter_v9b.log | tail -4
echo "=== RESULT END"
} 2>&1
trap - ERR
bye
