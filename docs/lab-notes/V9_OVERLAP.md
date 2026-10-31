# Overlapping host and GPU work in the filter loop (v9): tried, reverted (2026-10-31)

## Question

With fp_filter v7/v8, nvidia-smi showed the A40 busy 71% of the time. A megakernel review found that kernel-launch
overhead is under 1% of the time. The idle looked like host work: Python and numpy bookkeeping, the exact-stage
setup, and about eight device-wide syncs per (tile, segment) for the timers. The question was whether running that
host work while the GPU filters the next segment would lift GPU use and cut cost.

## What was built

The v9 code is at commit a5b3308 on branch claude/project-thread-ytdtvz. These files have since been removed from
the branch.

- `pipeline/seg_pipe.py`: a depth-1 pipeline. The front of segment s+1 (basis inverse FFTs, GEMM + peak kernel,
  window maxima) is queued on the default CUDA stream. The back of segment s (exact SNR and chi-squared, sg_chisq,
  appends) then runs on a second, high-priority stream, with torch and cupy bound to that stream together. Backs keep
  the serial order.
- No device-wide syncs in the loop. Timers use CUDA events. sig and norm are copied to the host once per tile instead
  of per segment. The basis scale stays on the GPU.
- Tests: the pipeline order on CPU (500 random tiles), the existing pre_keep test (0 of 300 mismatches), and a
  bitwise A/B on the GPU.

## Measurement

Setup: one Vast RTX 3090 (Maryland, $0.189/h), torch 2.4.1. The data was O4a wk09: the first 26 dataset lines, as
2 queue groups of 13. Only the base bank ran, because pyseobnr's pygsl-lite does not build on the pytorch 2.4.1
image. The four runs used the same downloaded data, in the order v8, v9, v8, v9.

| Run | Wall (s) | grp_00 total_s | grp_01 total_s (steady) | nvidia-smi mean / median busy |
|---|---|---|---|---|
| v8a (first run, cold caches) | 195 | 124.2 | 67.6 | 40% / 23% |
| v9a | 167 | 96.5 | 68.6 | 48% / 48% |
| v8b | 151 | 92.4 | 56.4 | 50% / 72% |
| v9b | 155 | 89.3 | 63.0 | 51% / 58% |

- **Output.** Every peak file was bitwise identical in all four runs, all work counters were equal, and peak memory
  was 14.6 GiB in every run.
- **Speed.** There was no gain. On the warm runs, v8b took 151 s and v9b 155 s. On the steady group, v8 took 56.4 and
  67.6 s and v9 took 63.0 and 68.6 s.
- **Why.** In v9 the GPU-side stages got slower, measured as event time per group: basis FFT 17.9 -> 26.5 s, exact
  stage 11.0 -> 13.6 s. That is what happens when exact-stage kernels run alongside cuFFT and the GEMM: they compete
  for the same memory bandwidth and SMs. The host-side select step (19-22 s per group) is mostly waiting on that
  GPU work. So the time between kernels in v8 was not wasted capacity that overlap could fill. The loop is limited
  by GPU throughput.
- **Basis scale.** A synthetic check found that keeping the scale on the device does not reproduce v8's rounding in
  a few percent of random cases: the float64-formed reciprocal mismatched in 12 of 200 checks, torch.reciprocal in
  6 of 100, and true division in 124 of 200. On the real data, every peak was still bitwise identical.

## Decision

Reverted. Production stays on fp_filter v8 (436c31dc). The 29% "idle" that nvidia-smi reports for v7/v8 cannot be
recovered by overlapping host work on this kernel mix. It sits elsewhere, in group start-up, the bank switch,
group end and prep waits, or nvidia-smi overstates it. Overlap only moves it.

Spend: about $0.15 of Vast credit over 5 boxes. Two boxes failed in the pyseobnr install. Two were stopped to fix
the scale form.

## If this comes back

Profile one v8 group with torch.profiler or nsys first. Find the gaps on the GPU timeline and attribute each to a
phase before building anything. The A/B tool `tests/cmp_runs.py` (bitwise peak files plus counters) stays in the
repository.
