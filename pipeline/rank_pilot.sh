#!/bin/bash
# Held-out pilot ranking job (CPU pod on gw-shared). A wrapper only: every ranking is the frozen rank_week.sh
# (FREEZE.md section 3) on a symlinked view of the pilot layout, so no frozen code changes.
#
# Usage: rank_pilot.sh WEEK [WEEK ...]        e.g. rank_pilot.sh wk02 wk06 wk18
# Env (all required unless noted):
#   ROOT     /workspace/results/o4a            (WEEK dirs: {search,inj_lvk,inj_ecc[,inj_dali]}/grp_NN/{peaks,peaks_b2}/)
#   SEGS     dir with WEEK.txt coinc segments  (default ROOT/WEEK/coinc_segments.txt if present)
#   BANK_A   bank_v2.hdf; BANK_E  vt_45_100_mm0.85.hdf (frozen eccentric bank)
#   PHASETD  phasetd_H1L1.npz; KNOWN o4_known.csv; BENCH dir with the frozen scripts; PY python
#   INJDIR   dir with lvk_WEEK.hdf and match_WEEK.npz (for the LVK head-to-head)
#   SHARE    project-share dir for small outputs, e.g. .../results/o4a (writes SHARE/WEEK/rank_{aligned,ecc}/)
#   PAR      weeks ranked at once (default 3; each ranking peaks at ~12 GB RAM)
# Per week and layer (aligned = peaks/ + BANK_A, ecc = peaks_b2/ + BANK_E): search ranking once, then each
# injection set against that search's background and fits. Outputs per WEEK/layer on SHARE:
#   fg.hdf, search.json, search.log, rank.log, coinc_segments.txt, candidates/ (from rank_week.sh), and
#   inj_<set>.hdf (foreground only), inj_<set>.json, h2h_lvk/ (LVK set only), timing.txt (CPU s per step).
set -uo pipefail
: "${ROOT:?}" "${BANK_A:?}" "${BANK_E:?}" "${PHASETD:?}" "${KNOWN:?}" "${BENCH:?}" "${INJDIR:?}" "${SHARE:?}"
PY=${PY:-python}; PAR=${PAR:-3}

one_week() {
  local W=$1 R=$ROOT/$1
  local SEG=${SEGS:+$SEGS/$W.txt}; SEG=${SEG:-$R/coinc_segments.txt}
  for layer in aligned ecc; do
    local sub=peaks bank=$BANK_A
    [ $layer = ecc ] && sub=peaks_b2 bank=$BANK_E
    local V=/tmp/view_${W}_$layer OUT=$R/rank_$layer DST=$SHARE/$W/rank_$layer
    mkdir -p "$DST" "$OUT"; : > "$DST/timing.txt"
    for set in search $(cd "$R" && ls -d inj_* 2>/dev/null); do
      rm -rf "$V"; mkdir -p "$V"; cp "$SEG" "$V/coinc_segments.txt"
      for g in "$R/search"/grp_*; do mkdir -p "$V/search/$(basename $g)"; ln -s "$g/$sub" "$V/search/$(basename $g)/peaks"; done
      if [ $set != search ]; then
        for g in "$R/$set"/grp_*; do mkdir -p "$V/inj/$(basename $g)"; ln -s "$g/$sub" "$V/inj/$(basename $g)/peaks"; done
      fi
      local inj= match=
      if [ $set = inj_lvk ]; then inj=$INJDIR/lvk_$W.hdf; match=$INJDIR/match_$W.npz; [ -f "$match" ] || match=; fi
      rm -f "$OUT/inj.hdf" "$OUT/inj.json"; rm -rf "$OUT/h2h"
      local t0=$(date +%s)
      BANK=$bank PHASETD=$PHASETD KNOWN=$KNOWN BENCH=$BENCH PY=$PY INJ=$inj MATCH=$match SHARE_OUT=$DST \
        bash "$BENCH/rank_week.sh" "$OUT" "$V" > "$OUT/rank_$set.out" 2>&1
      local rc=$?
      echo "$set $(( $(date +%s) - t0 )) s exit $rc" >> "$DST/timing.txt"
      if [ $set != search ] && [ -f "$OUT/inj.hdf" ]; then
        $PY -c "
import h5py, sys
with h5py.File('$OUT/inj.hdf') as f, h5py.File('$DST/$set.hdf', 'w') as o:
    for k, v in f.attrs.items(): o.attrs[k] = v
    for k in f['foreground']: o['foreground/' + k] = f['foreground/' + k][:]"
        cp "$OUT/inj.json" "$DST/$set.json" 2>/dev/null
        [ -d "$OUT/h2h" ] && rm -rf "$DST/h2h_lvk" && cp -r "$OUT/h2h" "$DST/h2h_lvk"
      fi
    done
    rm -rf "$V"
  done
  echo "$W DONE" >> "$SHARE/pilot_rank.log"
}
export -f one_week; export ROOT SEGS BANK_A BANK_E PHASETD KNOWN BENCH INJDIR SHARE PY
printf '%s\n' "$@" | xargs -P "$PAR" -I{} bash -c 'one_week {}'
echo "ALL DONE" >> "$SHARE/pilot_rank.log"
