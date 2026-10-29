#!/bin/bash
# pod_launch.sh job: O4a production (cascade/O4_RUN_PLAN.md go 2026-10-28), search pass only, frozen config (FREEZE.md).
#   WEEKS=wk03,wk10,... Per week and group: the frozen fp_filter with the base bank and the eccentric --bank2 on whole
#   segments, logged to the analysis history; peaks go to gw-shared /workspace/results/o4a/wkNN/search/grp_NN.
#   A finished week is linked into /work/out/o4a/wkNN for the lane container to pull; a PUT of got_wkNN (its md5 list)
#   deletes that week's raw peaks from the volume (manifest kept), so the volume never fills.
# Inputs arrive by HTTP PUT (serve.py on :8000): the eccentric bank (MD5SUMS). Downloader: GWOSC, 12 streams, one group ahead.
# Resumable: a pass whose timing.json exists on the volume is skipped.
W=${GW_WORK:-/work}; VOLD=${GW_VOL:-/workspace}; IN=/work/in
WEEKS=${WEEKS:?}; WEEKS=${WEEKS//,/ }; G=${G:-26}; AHEAD=${AHEAD:-1}; TILE=${TILE:-512}; BANK=$VOLD/data/banks/bank_v2.hdf; O=$GW_OUT
V=$VOLD/results/o4a; ST=$W/stores; RAW=$W/gwdata; CACHE=$W/cache; HIST=$VOLD/results/analysis_history.jsonl
st() { echo "$(date -u +%FT%T) $*" | tee -a $O/job_status.txt; }
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }
mkdir -p $ST $RAW $CACHE $V
kill $(ps -eo pid,args | awk '/http.server 8000/ && !/awk/ {print $1}') 2>/dev/null; sleep 1; python3 serve.py &
# frozen code and argv (FREEZE.md)
[ "$(md5sum < fp_filter.py | cut -c1-32)" = 463abd46022e228fe6bb7baf97d8fc22 ] || { st "FAIL fp_filter md5"; exit 1; }
[ "$(md5sum < inspiral_argv_H1_000.txt | cut -c1-32)" = b2b6e958c5680b41cb04b99e8e3e68fe ] || { st "FAIL argv md5"; exit 1; }
[ "$(md5sum < $BANK | cut -c1-32)" = 217554941ad158980526c0562b96011d ] || { st "FAIL base bank md5"; exit 1; }
python3 -c "import pyseobnr" || { st "FAIL pyseobnr missing from the venv"; exit 1; }
( while true; do echo "$(date -u +%T) mem $(cat /sys/fs/cgroup/memory.current 2>/dev/null) work_free_gb $(free_gb $W) raw_mb $(du -sm $RAW | cut -f1) stores_mb $(du -sm $ST | cut -f1) vol_free_gb $(free_gb $VOLD)" >> $O/mem.txt; sleep 15; done ) &
MEMPID=$!
python3 setup_job.py $BANK $VOLD/data/stores > /dev/null || { st "FAIL setup"; exit 1; }
sed -i 's/--snr-threshold [0-9.]*/--snr-threshold 4/; s/--keep-loudest-num [0-9]*/--keep-loudest-num 30/; s/--segment-start-pad [0-9]*/--segment-start-pad 48/' argv_base.txt
grep -q -- '--low-frequency-cutoff 20 ' argv_base.txt && grep -q -- '--segment-start-pad 48 ' argv_base.txt && grep -q -- '--snr-threshold 4 ' argv_base.txt || { st "FAIL argv"; exit 1; }
st "waiting for inputs"
for i in $(seq 1200); do (cd $IN 2>/dev/null && md5sum -c --quiet $OLDPWD/MD5SUMS > /dev/null 2>&1) && break; sleep 1; done
(cd $IN && md5sum -c $OLDPWD/MD5SUMS) > $O/inputs_md5.txt 2>&1 || { st "FAIL inputs: $(grep -v ': OK' $O/inputs_md5.txt | tr '\n' ' ')"; exit 1; }
ECC=$IN/vt_45_100_mm0.85.hdf; st "inputs ok ($(wc -l < MD5SUMS) files)"
DALI=0
SG=(--sgchisq-snr-threshold 6 --sgchisq-locations "mtotal>30:20-15,20-30,20-45,20-60,20-75,20-90,20-105,20-120")
export CUPY_CACHE_DIR=$VOLD/cache/cupy; mkdir -p $CUPY_CACHE_DIR
FLAGS=(--tile $TILE --prep gpu --prep-procs 7 --gen-threads 6 --gemm-cfg 64,64,4,2 --exact-snr 5.5 --fuse-q --sg-batch --sg-kernel --exact-stage regen --consolidate --consolidate-exempt-snr 5
       --bank2 $ECC --approximant2 SEOBNRv5EHM_ecc --eps2 0.02 --history $HIST)
st "weeks: $WEEKS, G=$G, /work $(free_gb $W) GB free, volume $(free_gb $VOLD) GB free"

for w in $WEEKS; do
  mkdir -p $W/plan/$w; sed "s#/work/stores#$ST#" pieces/$w/datasets.txt > $W/plan/$w/all.txt
  split -l $G -d $W/plan/$w/all.txt $W/plan/$w/grp_
done
ls -d $W/plan/*/grp_?? | sort > $W/plan/order.txt
passes() { echo search; }
done_all() { for p in $(passes $1); do [ -e $V/$1/$p/$2/timing.json ] || return 1; done; }

( n=0
  while read g; do n=$((n+1))
    while [ $(ls $W/plan/*/grp_??.done 2>/dev/null | wc -l) -ge $(( $(ls $W/plan/*/grp_??.filtered 2>/dev/null | wc -l) + AHEAD + 1 )) ]; do sleep 5; done
    w=$(basename $(dirname $g)); done_all $w $(basename $g) && { touch $g.done; continue; }
    t0=$SECONDS
    awk '{print $1, $2, $3, $4}' $g | xargs -P 12 -L 1 bash -c '[ -f $3 ] && exit 0; python3 chunk_prep.py $0 $1 $2 $3 --cache '$RAW' > /dev/null 2>> '$g'.dl.err || echo "DL FAIL $0 $1 $2" >> '$g'.dl.err'
    mb=$(du -smc $(awk '{print $4}' $g) 2>/dev/null | tail -1 | cut -f1)
    echo "$(date -u +%T) download $w/$(basename $g) $((SECONDS-t0)) s, stores $mb MB, $(grep -c 'DL FAIL' $g.dl.err 2>/dev/null) failed" >> $O/download.log
    nxt=$(sed -n "$((n+1))p" $W/plan/order.txt); lo=$( [ -n "$nxt" ] && awk 'NR==1 || $2<m {m=$2} END {print m}' $nxt || echo 9999999999)
    for f in $RAW/*.hdf5; do [ -e "$f" ] || continue; s=$(basename $f .hdf5 | awk -F- '{print $(NF-1)}'); [ $((s + 4096)) -lt $lo ] && rm -f $f; done
    touch $g.done
  done < $W/plan/order.txt ) &
DLPID=$!
( while true; do for f in $IN/got_*; do [ -e "$f" ] || continue; w=${f##*/got_}
    if [ -f $V/$w/manifest.json ] && [ -d $V/$w/search ]; then rm -f $O/o4a/$w; rm -rf $V/$w/search; echo "$(date -u +%FT%T) $w peaks pulled, deleted from volume" >> $O/job_status.txt; fi
  done; sleep 20; done ) &

for w in $WEEKS; do
  tw=$SECONDS
  for g in $W/plan/$w/grp_??; do
    gn=$(basename $g)
    if done_all $w $gn; then st "$w $gn already on the volume, skipped"; touch $g.filtered; continue; fi
    while [ ! -e $g.done ]; do sleep 2; done
    grep -q 'DL FAIL' $g.dl.err 2>/dev/null && { st "FAIL download $w $gn: $(grep -c 'DL FAIL' $g.dl.err) pieces"; kill $DLPID; exit 1; }
    ks=$(awk '{s+=$3-$2} END {printf "%.1f", s/1000}' $g); line="$w $gn ($ks ks):"
    for p in $(passes $w); do
      R=$V/$w/$p/$gn; [ -e $R/timing.json ] && continue
      X=(); [ $p = inj_lvk ] && X=(--injection-file $IN/lvk_$w.hdf); [ $p = inj_ecc ] && X=(--injection-file $IN/inj_ecc_${w}_nyq.hdf); [ $p = inj_dali ] && X=(--injection-file $IN/inj_dali_wk02.hdf)
      [ $(free_gb $VOLD) -ge 2 ] || { st "FAIL volume under 2 GB free before $w $gn $p"; kill $DLPID; exit 1; }
      t1=$SECONDS
      PP=; [ $p = inj_dali ] && PP=/work/pylib
      PYTHONPATH=$PP python3 fp_filter.py --datasets $g --bank $BANK --approximant IMRPhenomD --argv argv_base.txt --out $W/res/$w/$p/$gn "${FLAGS[@]}" -- "${SG[@]}" "${X[@]}" > $O/${w}_${gn}_$p.log 2>&1 || { st "FAIL filter $w $gn $p"; tail -5 $O/${w}_${gn}_$p.log >> $O/job_status.txt; kill $DLPID; exit 1; }
      mkdir -p $V/$w/$p && cp -r $W/res/$w/$p/$gn $V/$w/$p/ && rm -rf $W/res/$w/$p/$gn || { st "FAIL volume copy $w $gn $p"; kill $DLPID; exit 1; }
      cp $R/timing.json $O/timing_${w}_${gn}_$p.json
      line="$line $p $((SECONDS-t1)) s"
    done
    rm -f $(awk '{print $4}' $g); rm -rf $CACHE/*
    touch $g.filtered
    st "$line, volume $(free_gb $VOLD) GB free"
  done
  cp pieces/$w/datasets.txt pieces/$w/coinc_segments.txt pieces/$w/pieces.json $V/$w/
  python3 - $V/$w $((SECONDS-tw)) "$RUNPOD_POD_ID" <<'PY'
import glob, json, os, sys
d, wall, pod = sys.argv[1:]
m = dict(week=os.path.basename(d), wall_s=int(wall), pod=pod)
for kind in ('search', 'inj_lvk', 'inj_ecc', 'inj_dali'):
    t = [json.load(open(p)) for p in sorted(glob.glob(f'{d}/{kind}/grp_*/timing.json'))]
    if t:
        m[kind] = dict(groups=len(t), n_peaks=sum(sum(x['n_peaks'].values()) for x in t), data_ks=round(sum(x['data_ks'] for x in t), 1),
                       gpu_total_s=round(sum(x['total_s'] for x in t), 1), config_md5=sorted({x['provenance']['config_md5'] for x in t}))
json.dump(m, open(f'{d}/manifest.json', 'w'), indent=1)
print(json.dumps(m))
PY
  cp $V/$w/manifest.json $O/manifest_$w.json
  mkdir -p $O/o4a && ln -sfn $V/$w $O/o4a/$w
  st "$w done in $((SECONDS-tw)) s"
done
wait $DLPID
st "all weeks filtered: waiting up to 60 min for pulls"; for i in $(seq 360); do [ -z "$(ls $O/o4a 2>/dev/null)" ] && break; sleep 10; done
cp $HIST $O/analysis_history.jsonl
kill $MEMPID
st finished
