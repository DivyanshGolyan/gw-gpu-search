# gw-gpu-search

A low-cost GPU search for binary black hole mergers in public LIGO data. It searched all of the O4a and O4b
open data from H1 and L1 (232 days of time when both detectors were observing) for about **$11 of rented GPU
time, as billed**. That figure includes every restart and failed test. The search has two layers: a standard
aligned-spin template bank and an eccentric-orbit bank (SEOBNRv5EHM). Significance comes from time slides.

The search found **no new candidates** below a false-alarm rate of 1 per year. It recovered 111 of the 153
catalogued events (GWTC) that have a catalogue false-alarm rate below 1 per year. See
[results/O4_SUMMARY.md](results/O4_SUMMARY.md).

## Measured results

All numbers here are measured unless the table says otherwise. The lab notes in [docs/lab-notes](docs/lab-notes)
have the sources.

### Cost

| Run | Data | Billed | Notes |
| --- | --- | --- | --- |
| O4a filtering | 34 weeks | $5.66 (RunPod billing API) | About half of this was overhead: low GPU use before a fix, tests, and stop/relaunch rounds. At the steady rate the same data costs ~$2.9 |
| O4b filtering | 37 weeks | ~$5.42 | Vast.ai $4.53 measured from credit. The relay pod (~$0.89) is estimated from the RunPod balance drop. Includes re-filtering 4 weeks whose outputs were lost |
| Steady state | per week of H1+L1 (942 detector-ks) | $0.142 | A40 at $0.59/h, GPU-bound (868 GPU-s per week) |
| Ranking | per week | CPU only | 354 s per week on 4 vCPUs |

The O4b run missed its own target of $0.15/week (it came in at $0.26 per 942 detector-ks all-in). The GPUs spent
most of their time waiting on downloads from GWOSC. [docs/lab-notes/O4B_RUN.md](docs/lab-notes/O4B_RUN.md) has
the breakdown.

### Comparison with PyCBC

With the same 5,880 templates (4,612 aligned + 1,268 eccentric), PyCBC on CPU needs 0.118 core-seconds per
template per 1,000 s of data. We measured that on a Xeon 6342, and it matches the published 0.11-0.14
(Soni & Nitz, arXiv:2409.11317, Table 4). That is **182 core-hours per week of H1+L1**. Our filter takes 0.000157
A40-seconds for the same unit of work.

The cost ratio depends on what a CPU core-hour costs. One A40-hour does the work of about 750 core-hours.

| CPU price ($/core-hour) | PyCBC $/week | vs our filter ($0.142) | vs our all-in (~$0.20) |
| --- | --- | --- | --- |
| 0.010 | 1.82 | 13x | 9x |
| 0.015 (spot, central) | 2.73 | 19x | 14x |
| 0.030 | 5.46 | 38x | 27x |

### Sensitivity

- On three held-out O4a weeks, using LVK's own injection set, we find **0.935** of the injections PyCBC finds,
  each pipeline using its own false-alarm rates.
- The eccentric layer finds **1.29x** [1.22, 1.37] as many eccentric signals (e20 >= 0.15, total mass 45-100)
  as the aligned layer alone. The test was pre-registered and run on held-out data. A placebo second layer gave 1.01x.

### Not yet measured

- Sensitivity on O4b: no injections were run on O4b.
- How much sensitivity the coarse eccentric bank (minimal match 0.85) gives up against a standard dense bank
  (0.97). Until that test runs, the eccentric cost advantage counts only the speedup on the same templates.

## Layout

| Path | What it is |
| --- | --- |
| `pipeline/` | The code exactly as run in production (byte-identical, see `MD5SUMS`). Flat module layout: modules import each other by name |
| `pipeline/fp_filter.py` | GPU filter: SVD of the template bank, matched filtering as tensor-core matrix multiplies, peak finding, then exact SNR and chi-squared for surviving peaks (`exact_stage.py`) |
| `pipeline/coinc_rank.py`, `ranking_stat.py`, `rank_week.sh` | H1-L1 coincidence, likelihood-ratio ranking statistic, 40,000 time slides per chunk, false-alarm rates |
| `pipeline/pool_far.py`, `candidates.py` | Layer combination (2 x the lower layer's FAR) and cross-match against known events |
| `pipeline/ecc_*.py` | Eccentric template generation and bank placement |
| `data/banks/` | The aligned bank (`bank_v2.hdf`, IMRPhenomD) and the eccentric bank (`vt_45_100_mm0.85.hdf`, SEOBNRv5EHM) |
| `data/segments/` | The exact analysed time per week (GWOSC data segments) |
| `data/o4_known.csv` | 13,225 known events and candidates used for the cross-match |
| `ops/` | The scripts that ran the cloud jobs on RunPod and Vast.ai. These are specific to our setup; see REPRODUCE.md |
| `tests/` | Self-tests. `test_coinc_rank.py` runs on CPU; the others need a GPU |
| `results/` | O4a and O4b results and the PyCBC audit |
| `docs/lab-notes/` | The freeze record, pre-registrations, pilot report and bills, as written during the run |

## Reproducing

See [REPRODUCE.md](REPRODUCE.md). In short: you need Python 3.11, a CUDA 12 GPU with at least 24 GB of memory,
and the packages in `environment/`. Filter a week, rank it, then combine the layers.

## Provenance

Every production filter run wrote a `config_md5` into its output and logged itself to an analysis history.
`docs/lab-notes/FREEZE.md` lists every frozen file with its md5 and every change made during the run, with
the reason. GPU reruns agree to float32 rounding, but they are not bit-identical. Week 1 of O4a was used for
development and is excluded from all results.

## How this was built

The code was written over a few days by one engineer working with AI coding agents (Claude). Every result
above was checked by measurement, and the lab notes record each check. Please open an issue for any error you find.

## License

GPL-3.0-or-later, to match PyCBC and LALSuite, which this code imports. See [LICENSE](LICENSE).
Strain data comes from the Gravitational Wave Open Science Center (GWOSC); please cite and acknowledge
GWOSC as its terms ask.
