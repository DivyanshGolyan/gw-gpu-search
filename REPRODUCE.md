# Reproducing the O4a/O4b search

This guide lists the steps and settings that produced `results/`. The production runs used the cloud wrappers in
`ops/`. Those wrappers are tied to our RunPod and Vast.ai setup (network volumes, an HTTP relay, kill timers), so
the steps below give the underlying commands. A generic single-machine runner is not written yet. Until it is,
read `ops/run_prod_o4a.sh` as the reference for how the pieces were chained.

## 1. Environment

- Linux, Python 3.11, an NVIDIA GPU with CUDA 12 and at least 24 GB of memory. Production ran on A40, RTX A5000,
  RTX 3090 and RTX 4090 cards. A 24 GB card needs `fp_filter.py` v8, which is the version shipped here.
- `pip install -r environment/requirements.txt`. The full production environment is in
  `environment/pip_freeze_pod.txt`.
- pyseobnr (the eccentric layer) needs GSL: `apt install libgsl-dev swig`, then
  `pip install "setuptools<70" wheel numpy==2.2.6`,
  `pip install pygsl-lite --no-build-isolation` (it may report a metadata error, ignore it), then `pip install pyseobnr==0.3.7`.
- Check the code against the frozen versions: `md5sum -c MD5SUMS`.

## 2. Data

`data/segments/{o4a,o4b}/wkNN/datasets.txt` lists the analysed GWOSC stretches for each week (`IFO START END STORE`),
and `coinc_segments.txt` lists the coincident time. Download and condition each stretch:

```
python pipeline/chunk_prep.py H1 1373583056 1373591112 /work/stores/H1-1373583056-1373591112.hdf --dataset O4a --cache /work/gwdata
```

Use `--dataset O4b` for O4b weeks. A week of H1+L1 is about 26 GB of GWOSC files. Delete the raw files once
a week is conditioned.

## 3. Filter (GPU)

Split a week's `datasets.txt` into groups of 13-26 lines (production used both sizes). Make `argv_base.txt` from
`data/inspiral_argv_H1_000.txt` with `ops/setup_job.py`, then set the frozen thresholds: `--snr-threshold 4`,
`--keep-loudest-num 30`, `--segment-start-pad 48`, and check `--low-frequency-cutoff 20`. Then, per group:

```
python pipeline/fp_filter.py --datasets GROUP.txt --bank data/banks/bank_v2.hdf --approximant IMRPhenomD \
  --argv argv_base.txt --out OUT/wkNN/search/grp_NN \
  --tile 512 --prep gpu --prep-procs 7 --gen-threads 6 --gemm-cfg 64,64,4,2 --exact-snr 5.5 --fuse-q \
  --sg-batch --sg-kernel --exact-stage regen --consolidate --consolidate-exempt-snr 5 \
  --bank2 data/banks/vt_45_100_mm0.85.hdf --approximant2 SEOBNRv5EHM_ecc --eps2 0.02 --history history.jsonl \
  -- --sgchisq-snr-threshold 6 --sgchisq-locations "mtotal>30:20-15,20-30,20-45,20-60,20-75,20-90,20-105,20-120"
```

The aligned layer's peaks go to `grp_NN/peaks/` and the eccentric layer's go to `grp_NN/peaks_b2/`. For an
injection pass, add `--injection-file FILE.hdf` and write to `inj_<set>/` instead of `search/`.
`docs/lab-notes/SELECTION_RULE.md` describes the trigger selection.

## 4. Rank (CPU)

`pipeline/rank_pilot.sh` ranks each layer of a week with the frozen `rank_week.sh`: lnl statistic with fits
and phase-time terms, sine-Gaussian veto, 40,000 time slides of 0.1 s. Example:

```
ROOT=OUT BANK_A=data/banks/bank_v2.hdf BANK_E=data/banks/vt_45_100_mm0.85.hdf \
PHASETD=data/phasetd_H1L1.npz KNOWN=data/o4_known.csv BENCH=pipeline INJDIR=INJ SHARE=RANKED PY=python \
  bash pipeline/rank_pilot.sh wk03
```

Short weeks (under about 2 days of live time) were ranked together as one chunk. `results/O4A_RESULTS.md` and
`results/O4B_RESULTS.md` list the chunks. A chunk peaks at about 12 GB of RAM.

## 5. Combine layers and cross-match

- Reported FAR = 2 x the lower of the two layer FARs (`pipeline/pool_far.py --n-layers 2`), computed per chunk.
- `pipeline/candidates.py` labels a candidate KNOWN if any entry of `data/o4_known.csv` is within 1 s, otherwise NEW.
- `pipeline/o4a_results.py RESULTS_DIR data/o4_known.csv OUT_PREFIX --skip wk01` builds the summary tables.
  Week 1 of O4a is development data and is always skipped.

## 6. Tests

- `python tests/test_coinc_rank.py lnl` checks on CPU that time-slide FARs are calibrated in fake noise and
  that a loud fake signal is recovered. It passes in a fresh environment.
- `tests/test_exact_stage.py`, `test_gpu_prep.py`, `test_interp.py`, `test_chisq_kernel_gpu.py` and
  `test_prekeep.py` need a GPU.

## Expected cost

At the measured steady rate, filtering one week of H1+L1 takes 868 GPU-seconds on an A40 (about $0.14 at
$0.59/h). The real limit is often download speed from GWOSC; see `docs/lab-notes/O4B_RUN.md`.
