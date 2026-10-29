# Real PyCBC search on a GPU (A40, RunPod Secure Cloud, 2026-10-27)

**Bottom line.** The real `pycbc_inspiral` runs on the GPU after a 2-line fix to PyCBC, and its triggers match the CPU run to about 0.1%. Matched filtering got 16x faster per process. But the whole job only got 2x faster, because half of every job is CPU-side data preparation that PyCBC repeats for each of the 16 bank splits. Run as-is, a GPU pod costs *more* per template than CPU cores. Two cheap changes fix that (run fewer, bigger jobs per detector, and run ~8 processes at once on the GPU). That projects to roughly 70 CPU cores' worth of filtering per A40, about 3.5x cheaper than CPU.

Setup: same config as `baseline_spa.ini` (814-template bank, 16 splits, H1+L1, 12,112 s of analysed O4a data), one A40 ($0.59/h) with 9 vCPUs of an Intel Xeon Gold 6342. PyCBC 2.8.3, cupy-cuda12x 14.2.0, `processing-scheme = cupy`, `fft-backends` removed. Total spend: **$0.45** (three pods, all deleted).

## What happened

1. **The GPU path crashed out of the box.** On the first template it failed with `CUDA_ERROR_ILLEGAL_ADDRESS` in `pycbc/events/threshold_cupy.py` (`CUDAThresholdCluster.threshold_and_cluster`). That is the GPU kernel that finds SNR peaks above threshold. The class launches it with the grid and block sizes swapped (`fn((nt,1,1),(nb,1),...)` instead of `fn((nb,),(nt,),...)`, and the same for `fn2`). The module-level function a few lines above has the right order. That made 1024 threads index a 128-slot shared-memory array. The fix is `pod_scripts/patch_thresh.py`, a 2-line swap. PyCBC itself warns that this backend is a "prototype". This bug is worth sending upstream.
2. **The first CPU calibration run died** because the pod reports 96 cores (the host's) but only grants 9. Launching 32 jobs, each starting 96 OpenMP threads, hit the thread limit. The rerun pins `OMP_NUM_THREADS=1`. Because of this, the CPU calibration covers 2 of the 32 jobs (one H1, one L1), run one at a time on one core. The two jobs took 72.3 and 72.6 s, so the spread is small.

## Timing (per template, per 1000 s of data, per detector)

Each job is one detector with ~51 templates. "Prep" means everything before the first template: read the data, high-pass, resample 4096 to 2048 Hz, autogate, FFT into 15 overlapping segments, and estimate the PSD. "Filter" means the template loop: matched filter, peak finding, chi-squared.

| Run | seconds per job (prep + filter) | s / template / ks / det | USD per 1M template x ks |
|---|---|---|---|
| CPU, 1 core (Xeon 6342) | 72.5 (34.9 + 35.2) | 0.118 total; 0.057 filter only | 0.98 total; 0.48 filter only (at $0.03/core-h) |
| GPU, 1 job at a time | 37.6 (32.8 + 2.2) | 0.061 total; **0.0036 filter only** | 10.0 total (GPU idle ~98% of the time); 0.59 filter only |
| GPU, 8 jobs at once (whole bank, both detectors: 172.5 s wall) | 42.3 (35.3 + 4.0) | 0.0088 of pod wall time | 1.43 |
| *Projection:* GPU, 8 processes, prep amortised (big bank or 1 job per detector) | - | ~0.0008 filter | **~0.13** |

- **Calibration against our earlier number.** Our baseline of 0.245 core-s came from this sandbox's CPU. A Xeon 6342 core does the same job in 0.118, so it is 2.1x faster. Earlier "core-hour" figures in this project are in slow-core units.
- **Half of each CPU job is prep.** That prep is identical across all 16 splits of the bank, so 15/16 of it is wasted work, on CPU as well as GPU. This is a free 2x on the CPU baseline too: use fewer splits, or prepare the data once and share it.
- **GPU utilisation:** about 8% while one job filters, and with 8 jobs 21% averaged over the run (57% in the seconds when any kernel was running). `pycbc_inspiral` filters one template at a time, so the GPU mostly waits on Python and kernel launches. With 8 concurrent jobs, filtering per job only slowed from 2.2 to 4.0 s. That means the GPU absorbed about 4.4x more work, which is the basis of the projection row. GPU memory used was under 1 GB per process (7.5 GB with 8 running), so far more than 8 processes would fit.
- **Inner-loop microbenchmark vs real code:** the earlier batched kernel did 0.00098 GPU-s per template-ks. Real PyCBC, single process, needs 0.0036 (3.7x worse, and that includes chi-squared). With 8 processes it comes to about 0.0008, roughly at microbenchmark level.

## Do the GPU and CPU find the same things?

Yes. The comparison covers the same template and same time sample in the 2 jobs run on both CPU and GPU (`raw/trigger_compare.txt`):

| Job | triggers CPU / GPU | SNR > 6 CPU / GPU | loudest SNR CPU / GPU | SNR difference on matched triggers |
|---|---|---|---|---|
| H1 split 0 | 5123 / 5079 | 31 / 31 | 6.800 / 6.794 (0.08%) | median 0.04%, max 0.16% |
| L1 split 8 | 5292 / 5260 | 29 / 29 | 6.587 / 6.584 (0.05%) | median 0.04%, max 0.20% |

Chi-squared agrees to about 0.08% (median). The ~0.7% fewer triggers on GPU are all near the SNR 5 threshold, where a 0.04% difference in SNR decides whether a point gets in. All 32 GPU files were also run twice (1 job vs 8 jobs at once). The SNR > 6 count was identical (1059), and the loudest SNRs were bit-identical.

## What this means for the project

- The GPU is not the bottleneck and filtering is not the expensive part. **Data prep repeated per job, plus one-template-at-a-time Python overhead, are.** Both are engineering fixes, not physics.
- Next step for a real cost win: one prep per detector per chunk, then many templates batched on the GPU. That means either many `pycbc_inspiral` processes per GPU with large splits, or our own batched filter, which already runs at 0.001 s in the microbenchmark. Then rerun this comparison on a bank of 10^4 or more templates, where prep is negligible.
- Upstream contribution: the `threshold_cupy.py` launch-order bug (GPU path crashes on first template in PyCBC 2.8.3).

## Files

- `pod_scripts/run2.sh` (pod script), `drive.py` (runs the full-data inspiral jobs), `cmp.py`, `patch_thresh.py` (the fix)
- `raw/`: per-job ledgers (`cpu_ledger.json`, `gpu_ledger.json`, `gpu4_ledger.json` = 8-at-once), wall clocks, nvidia-smi traces at 1 s, example inspiral logs for CPU and GPU, the crash log before the fix, CPU model, pip freeze
