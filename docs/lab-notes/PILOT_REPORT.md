# Held-out pilot: wk02, wk06, wk18 (GPU filter lane, 2026-10-28)

The pilot runs the frozen filter: fp_filter 463abd46, argv b2b6e958, exact_stage 3989f01c (CHECK_TOL 1e-4 per the
FREEZE change log). The eccentric bank is vt_45_100_mm0.85 (bffc1d39) on whole segments, with `--history` on every
pass. Ranking is the frozen rank_week.sh (8c88bae7) through the significance lane's wrapper. Manifests, the pod
status log and the download log are in pilot/.

## Data and passes

| week | groups | detector-ks | search | inj_lvk | inj_ecc | inj_dali |
|---|---|---|---|---|---|---|
| wk02 | 7 | 681.6 | 1,674 s | 1,832 s | 1,953 s | 1,668 s |
| wk06 | 6 | 540.4 | 1,375 s | 1,421 s | 1,545 s | - |
| wk18 | 5 | 480.2 | 1,134 s | 1,163 s | 1,298 s | - |

The times are A5000 wall per pass. The three weeks total 1,702.2 detector-ks, which is 1.807 weeks of 942 dks.

config_md5 per pass and group (first 8 hex digits; the full values are in pilot/manifest_*.json):

- wk02 search 378fa458 557300ff 71c9d3a0 736dd82d 78d19fb1 c2d5ab3d e2dbfe8b
- wk02 inj_lvk 05eb8212 0775407c 85c951de 897d2b35 924f9890 bc575632 ef3fdc69
- wk02 inj_ecc 232063dd 2cc7bf2e 379b6324 4953c1b1 59e94dc0 f229dd80 f2ed1c8d
- wk02 inj_dali 03de95a1 12721413 47dc77d7 56896c55 5f0cf8e2 a5554928 c05d8bb6
- wk06 search 18346a00 4fdd0159 90ab7897 dd2d9975 e959ab2c f55589b3
- wk06 inj_lvk 4da6b7b1 61767709 8185bfea 9c5fb1bb be251b75 dd9ed7f2
- wk06 inj_ecc 0d0b4892 32e23391 5f82d35b 6a557b1d a9b73c97 d9d1c898
- wk18 search 4cf60041 7ec6c4e2 914885fb d94a15f7 f66647b7
- wk18 inj_lvk 12977e5b 4eb41a90 882d625a 97674660 e3ead26a
- wk18 inj_ecc 22d16fa9 85958f8b aa6e83ca b494f485 bf6cd251

## Cost (RunPod billing API, 11:23 UTC)

| item | billed |
|---|---|
| filter pods, failed attempts included | $1.379 |
| ranking pods (wk02; one failed start) | $0.222 |
| serve/pull pods (wk06 and wk18 were ranked on the lane container) | $0.079 billed + ~$0.046 pending |
| **total** | **~$1.73** |

Per full week of 942 dks (÷1.835 as the significance lane counts it):

| | $/week |
|---|---|
| all-in, everything above | $0.94 |
| filtering, search + 3 injection sets | $0.75 |
| filtering, search only (27.8% of GPU wall) | $0.21 billed (with failures), $0.17 clean |
| ranking on its own pod | ~$0.10 (→ $0 overlapped or on a container; see CPU_WORK.md) |

Against the ≤ $0.30/week target, the production search pass with ranking off-pod comes in at $0.17-0.21. Injection
passes for sensitivity measurement cost about 3x more again per week, so they should run on a sample of weeks,
not on every week.

## Known events (candidates at FAR < 3/yr; the readout and pooling belong to the significance lane)

| week | event | aligned FAR/yr | eccentric FAR/yr |
|---|---|---|---|
| wk02 | GW230712_090405 (cWB-only in GWTC; PyCBC 255/yr) | not found | 0.56 |
| wk06 | GW230811_032116 | 0.003 (floor) | 0.003 (floor) |
| wk06 | GW230806_204041 (S230806ak) | 0.37 | - |
| wk18 | S231102w | 0.003 (floor) | 0.003 (floor) |
| wk18 | GW231104_133418 | 0.003 (floor) | - |

There are no new candidates below 3/yr in any week.

## Notes

- Ranking on the lane container reproduces the pod's foreground exactly. It took 45 min per week, against 23 on
  the pod.
- The wk06 LVK shortfall (0.912 of PyCBC) is not a gating or injection bug: see
  results/coinc_rank/pilot/wk06/missed15_README.md.
- GPU use averaged 27% on the filter pod. Caching templates and overlapping data prep with the GPU is the next
  saving (CPU_WORK.md).
