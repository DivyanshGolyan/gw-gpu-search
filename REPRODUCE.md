# How to rerun the O4a and O4b search

This guide gives the commands and settings behind `results/`. We ran production through the scripts in `ops/`.
Those scripts depend on our RunPod and Vast.ai setup, so this guide gives the commands they call. There is no
single-machine runner yet. To see how the steps were chained in production, read `ops/run_prod_o4a.sh`.

## 1. Set up the environment

You need Linux, Python 3.11, and an NVIDIA GPU with CUDA 12 and at least 24 GB of memory. Production ran on
A40, RTX A5000, RTX 3090, and RTX 4090 cards.

1. Install the packages: `pip install -r environment/requirements.txt`.
   The full production environment is in `environment/pip_freeze_pod.txt`.
2. Install GSL for the eccentric templates: `apt install libgsl-dev swig`.
3. Install pyseobnr in this order:
	```
	pip install "setuptools<70" wheel numpy==2.2.6
	pip install pygsl-lite --no-build-isolation
	pip install pyseobnr==0.3.7
	```
	The `pygsl-lite` step can print a metadata error. You can ignore it.
4. Check that the code matches the production version: `md5sum -c MD5SUMS`.

## 2. Download the data

Each week has a folder in `data/segments/o4a/` or `data/segments/o4b/`. In each folder, `datasets.txt` lists the
stretches to analyse, one per line: detector, GPS start, GPS end, output file. `coinc_segments.txt` lists the
times when both detectors were recording.

Download and prepare each stretch from GWOSC:

```
python pipeline/chunk_prep.py H1 1373583056 1373591112 /work/stores/H1-1373583056-1373591112.hdf --dataset O4a --cache /work/gwdata
```

For O4b weeks, use `--dataset O4b`. One week of H1 and L1 data is about 26 GB. After a week is prepared, you can
delete the raw files in the cache folder.

## 3. Filter the data on the GPU

1. Split the week's `datasets.txt` into groups of 13 to 26 lines. Production used both sizes.
2. Make `argv_base.txt` from `data/inspiral_argv_H1_000.txt` with `ops/setup_job.py`.
3. In `argv_base.txt`, set `--snr-threshold 4`, `--keep-loudest-num 30`, and `--segment-start-pad 48`.
   Check that it has `--low-frequency-cutoff 20`.
4. Filter each group:
	```
	python pipeline/fp_filter.py --datasets GROUP.txt --bank data/banks/bank_v2.hdf --approximant IMRPhenomD \
	  --argv argv_base.txt --out OUT/wkNN/search/grp_NN \
	  --tile 512 --prep gpu --prep-procs 7 --gen-threads 6 --gemm-cfg 64,64,4,2 --exact-snr 5.5 --fuse-q \
	  --sg-batch --sg-kernel --exact-stage regen --consolidate --consolidate-exempt-snr 5 \
	  --bank2 data/banks/vt_45_100_mm0.85.hdf --approximant2 SEOBNRv5EHM_ecc --eps2 0.02 --history history.jsonl \
	  -- --sgchisq-snr-threshold 6 --sgchisq-locations "mtotal>30:20-15,20-30,20-45,20-60,20-75,20-90,20-105,20-120"
	```

The filter writes the matches for the standard templates to `grp_NN/peaks/` and the matches for the eccentric
templates to `grp_NN/peaks_b2/`. To filter injections, add `--injection-file FILE.hdf` and write to
`inj_<set>/` instead of `search/`. `docs/lab-notes/SELECTION_RULE.md` explains which matches the filter keeps.

## 4. Rank the matches on a CPU

`pipeline/rank_pilot.sh` ranks each template set of a week with the production script `rank_week.sh`. For example:

```
ROOT=OUT BANK_A=data/banks/bank_v2.hdf BANK_E=data/banks/vt_45_100_mm0.85.hdf \
PHASETD=data/phasetd_H1L1.npz KNOWN=data/o4_known.csv BENCH=pipeline INJDIR=INJ SHARE=RANKED PY=python \
  bash pipeline/rank_pilot.sh wk03
```

Ranking one week uses up to about 12 GB of memory. We ranked short weeks, with less than about 2 days of data,
together as one chunk. `results/O4A_RESULTS.md` and `results/O4B_RESULTS.md` list the chunks.

## 5. Combine the two template sets and check against known events

1. Combine the false-alarm rates: `pipeline/pool_far.py --n-layers 2`. The reported rate is twice the lower of
   the two rates, computed for each chunk.
2. Check candidates against known events: `pipeline/candidates.py`. A candidate within 1 second of an entry in
   `data/o4_known.csv` is known. Any other candidate is new.
3. Build the summary tables: `pipeline/o4a_results.py RESULTS_DIR data/o4_known.csv OUT_PREFIX --skip wk01`.
   Always skip week 1 of O4a, because we used it to develop the pipeline.

## 6. Run the tests

- To test the ranking on a CPU, run `python tests/test_coinc_rank.py lnl`. It checks that false-alarm rates are
  correct in fake noise and that a loud fake signal is found. It prints `PASS calibration signal`.
- `tests/test_exact_stage.py`, `test_gpu_prep.py`, `test_interp.py`, `test_chisq_kernel_gpu.py`, and
  `test_prekeep.py` need a GPU.

## Expected cost

At the steady rate, one week of H1 and L1 data takes 868 GPU-seconds on an A40. That is about $0.14 at $0.59 per
hour. Download speed from GWOSC is often the real limit. `docs/lab-notes/O4B_RUN.md` has the numbers.
