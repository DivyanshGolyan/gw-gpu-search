#!/bin/bash
# Rank one analysis chunk (one O4 week, or several short weeks together) on a pod with gw-shared mounted.
# CPU only, single-threaded; run it in the background on the filter pod while the GPU does the next week.
#
# Usage: rank_week.sh OUTDIR WEEKDIR [WEEKDIR ...]
#   WEEKDIR  /workspace/results/o4a/wkNN with search/grp_0N/peaks/*.hdf, coinc_segments.txt and, on injection
#            weeks, inj/grp_0N/peaks/*.hdf. Several WEEKDIRs are ranked as one chunk (short weeks, < 2 days live).
#   OUTDIR   e.g. /workspace/results/o4a/wkNN/rank; full outputs stay there (background included).
# Env: BENCH (dir with coinc_rank.py, merge_peaks.py, fg_summary.py, compare_pycbc.py; default: this script's dir),
#      BANK (bank_v2.hdf), PHASETD (phasetd_H1L1.npz, default $BENCH/phasetd_H1L1.npz),
#      INJ, MATCH (LVK injection subset + aligned-22 match for this chunk; inj ranking runs when inj peaks exist),
#      SHARE_OUT (optional dir to copy the small summaries to), PY (python, default python),
#      KNOWN (catalogs/o4_known.csv) and KEEP_FAR (default 3 /yr): see keep_peaks below.
# Settings frozen 2026-10-27 (REPORT section 3, update 19:05): lnl[fits,phasetd] + sgveto, min-rank 4,
# 40,000 slides of 0.1 s, min-stat 16, fit pool 15.
set -euo pipefail
OUT=$1; shift
BENCH=${BENCH:-$(cd "$(dirname "$0")" && pwd)}; PY=${PY:-python}
PHASETD=${PHASETD:-$BENCH/phasetd_H1L1.npz}; : "${BANK:?set BANK to bank_v2.hdf}"
mkdir -p "$OUT"
ARGS="--stat lnl --terms fits,phasetd --fit-pool 15 --sgveto --max-slides 20000 --min-stat 16 --min-rank 4"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/rank.log"; }

cat $(for w in "$@"; do echo "$w/coinc_segments.txt"; done) | sort -n > "$OUT/coinc_segments.txt"
# peaks are merged in memory by coinc_rank.py and fg_summary.py: comma-separated piece lists, no merged files
declare -A PK
for pass in search inj; do
  for ifo in H1 L1; do
    PK[${pass}_$ifo]=$(for w in "$@"; do ls "$w/$pass"/grp_0*/peaks/$ifo-*.hdf 2>/dev/null || true; done | paste -sd,)
  done
done

# a restarted job keeps a finished search ranking
common="--peaks H1=${PK[search_H1]} --peaks L1=${PK[search_L1]} --segments $OUT/coinc_segments.txt --bank $BANK --top 20 --phasetd-cache $PHASETD"
[ -f "$OUT/search.hdf" ] && [ -f "$OUT/fits.npz" ] || \
  $PY -W ignore "$BENCH/coinc_rank.py" $common --out "$OUT/search.hdf" $ARGS --save-fits "$OUT/fits.npz" > "$OUT/search.log" 2>&1
log "ranked search: $(head -1 "$OUT/search.log")"
$PY "$BENCH/fg_summary.py" "$OUT/search.hdf" "$OUT/fg.hdf" $(for w in "$@"; do echo --weekdir "$w"; done) \
    --peaks H1="${PK[search_H1]}" --peaks L1="${PK[search_L1]}" >> "$OUT/rank.log"

# keep this week's peaks (WEEKDIR/keep_peaks) when a candidate below KEEP_FAR /yr (default 3) is in no catalog:
# it may need a rerun with more slides. With KNOWN unset, any candidate below KEEP_FAR keeps them.
KEEP_FAR=${KEEP_FAR:-3}
if [ -n "${KNOWN:-}" ]; then
  $PY -W ignore "$BENCH/candidates.py" --ranked base:chunk="$OUT/fg.hdf" --known "$KNOWN" --out "$OUT/candidates" \
      --far-thr "$KEEP_FAR" --no-dq >> "$OUT/rank.log" 2>&1
  keep=$($PY -c "import json; print(int(json.load(open('$OUT/candidates/candidates.json'))['n_new'] > 0))")
else
  keep=$($PY -c "import h5py; print(int((h5py.File('$OUT/fg.hdf')['foreground/far_excl_per_yr'][:] < $KEEP_FAR).any()))")
fi
if [ "$keep" = 1 ]; then for w in "$@"; do touch "$w/keep_peaks"; done; log "keep_peaks: candidate below $KEEP_FAR/yr not in a catalog"; fi

if [ -n "${PK[inj_H1]}" ]; then
  $PY -W ignore "$BENCH/coinc_rank.py" --peaks H1=${PK[inj_H1]} --peaks L1=${PK[inj_L1]} --segments $OUT/coinc_segments.txt \
      --bank $BANK --top 5 --phasetd-cache $PHASETD --out "$OUT/inj.hdf" $ARGS \
      --background-from "$OUT/search.hdf" --fits-from "$OUT/fits.npz" > "$OUT/inj.log" 2>&1
  log "ranked inj"
  if [ -n "${INJ:-}" ] && [ -n "${MATCH:-}" ]; then
    $PY "$BENCH/compare_pycbc.py" --inj "$INJ" --match "$MATCH" --ours "$OUT/inj.hdf" --out "$OUT/h2h" > "$OUT/h2h.log" 2>&1
    log "head-to-head: $(grep -A6 '"all"' "$OUT/h2h/head_to_head.json" | tr -d ' \n')"
  fi
fi

if [ -n "${SHARE_OUT:-}" ]; then
  mkdir -p "$SHARE_OUT"
  cp "$OUT"/{fg.hdf,search.json,search.log,rank.log,coinc_segments.txt} "$SHARE_OUT/"
  [ -d "$OUT/candidates" ] && cp -r "$OUT/candidates" "$SHARE_OUT/"
  [ -d "$OUT/h2h" ] && cp -r "$OUT/h2h" "$OUT/inj.json" "$SHARE_OUT/"
fi
log "DONE"
