# gw-gpu-search

gw-gpu-search looks for merging black holes in public LIGO data, on rented GPUs. We ran it over all of the
O4a and O4b open data from the two LIGO detectors, H1 and L1. That is 232 days in which both detectors were
recording. The run cost about $11, as billed, including every failed test and restart.

The search found no new merger candidates. It found 111 of the 153 known mergers that the official catalogues
report with fewer than 1 false alarm per year. Of the 101 mergers that PyCBC, a standard LIGO pipeline,
finds with high confidence, it found 99. [results/O4_SUMMARY.md](results/O4_SUMMARY.md) has the details, and
[results/missed_events](results/missed_events/README.md) explains the misses.

## How the search works

The pipeline compares the detector data with a set of predicted signals, called templates. Each template is the
waveform of one kind of merger. The search uses two template sets:

- 4,612 standard templates for black holes on circular orbits (`data/banks/bank_v2.hdf`).
- 1,268 templates for black holes on eccentric, or oval, orbits (`data/banks/vt_45_100_mm0.85.hdf`).
  The standard LIGO template searches of O4 don't use eccentric templates.

The pipeline keeps a match only if both detectors see it within a few milliseconds of each other. To estimate how often noise alone
produces a match that loud, the pipeline shifts one detector's data in time and repeats the comparison 40,000
times. That estimate is the false-alarm rate.

## What we measured

### Cost

| Run | Data | Cost | Notes |
| --- | --- | --- | --- |
| O4a | 34 weeks | $5.66 | From the RunPod bill. About half was overhead from tests and restarts. At the steady rate, the same data costs about $2.90. |
| O4b | 37 weeks | about $5.42 | $4.53 from Vast.ai credit. The relay server (about $0.89) is estimated. Includes 4 weeks that we filtered twice after a storage failure. |
| One week, steady rate | 1 week of H1 and L1 data | $0.142 | One A40 GPU at $0.59 per hour. Filtering takes 868 GPU-seconds per week. |
| Ranking | 1 week | CPU only | 354 seconds on 4 CPU cores. |

Building and testing the pipeline cost more than running it. The whole project, from the first benchmark on
2026-10-27 to the end of O4b, cost about $33 in billed cloud time:

| Item | Cost | Source |
| --- | --- | --- |
| RunPod GPU and CPU pods, 2026-10-27 and 10-28 (development, tests, O4a) | $27.59 | RunPod billing API, 185 pods |
| RunPod network storage | $0.66 | RunPod billing API |
| Vast.ai GPUs (O4b, including tests) | $4.53 | Vast.ai credit used |
| RunPod relay pods on 2026-10-29 | about $0.25 | Estimated. The billing API had no rows for that day yet. |

The O4b run missed our target of $0.15 per week. It cost $0.26 per week, because the GPUs spent much of the
time waiting for data downloads. [docs/lab-notes/O4B_RUN.md](docs/lab-notes/O4B_RUN.md) breaks the cost down.

### Comparison with PyCBC

PyCBC is one of the standard LIGO search pipelines. With our 5,880 templates, PyCBC on CPUs needs 182 core-hours
to filter one week of data. We measured PyCBC's speed on an Intel Xeon 6342. It matches the published figure in
Soni & Nitz, arXiv:2409.11317, Table 4.

One A40 GPU-hour does about the work of 750 CPU core-hours. The saving in dollars depends on the price of
a CPU core-hour:

| CPU price per core-hour | PyCBC cost per week | Our filter is cheaper by | Our whole run (about $0.20 per week) is cheaper by |
| --- | --- | --- | --- |
| $0.010 | $1.82 | 13x | 9x |
| $0.015 (typical spot price) | $2.73 | 19x | 14x |
| $0.030 | $5.46 | 38x | 27x |

These numbers have limits:

- The PyCBC speed comes from two jobs on one CPU model. Its agreement with the published figure is the main check.
- About half of each PyCBC job prepares the data, and PyCBC repeats that step for each part of the template set.
  A PyCBC setup that avoids the repeats could cost up to half as much as the table shows.
- PyCBC can also run on GPUs. We estimate that a well-tuned PyCBC GPU setup costs about 5 times what ours does,
  not 19 times. That figure is an estimate. We haven't measured it.
- Our cost depends on the machine. On O4b, the cost per week ranged from $0.141 to $0.209.
  Download speed set the cost on most machines.

[docs/lab-notes/PYCBC_BASELINE.md](docs/lab-notes/PYCBC_BASELINE.md) describes the PyCBC measurement.

### Sensitivity

To test sensitivity, LIGO adds fake signals to real data. These fake signals are called injections. We used
LIGO's own O4a injections on three weeks that we kept out of development:

- Our search finds 0.935 of the injections that PyCBC finds. Each pipeline uses its own false-alarm rates.
- With the eccentric templates added, the search finds 1.29 times as many eccentric signals as it finds with
  the standard templates alone (90% range 1.22 to 1.37). The test applies to signals with eccentricity of at
  least 0.15 and total mass from 45 to 100 solar masses. We wrote down the pass rule before the run. As a
  control, an extra set of standard templates gave 1.01 times.

### What we haven't measured

- Sensitivity on O4b. We ran no injections on O4b.
- How much sensitivity our sparse eccentric template set loses against a standard dense one. Until we measure
  that, the eccentric cost saving counts only the speedup on the same templates.

## Repository contents

| Path | Contents |
| --- | --- |
| `pipeline/` | The code as it ran in production. `MD5SUMS` proves that each file is unchanged. The modules import each other by name, so run them from this directory. |
| `pipeline/fp_filter.py` | The GPU filter. It compresses the template set with an SVD, runs the comparison as matrix multiplies on the GPU, and computes exact values for the loudest matches in `exact_stage.py`. |
| `pipeline/coinc_rank.py`, `ranking_stat.py`, `rank_week.sh` | Matching between detectors, the ranking statistic, the time shifts, and the false-alarm rates. |
| `pipeline/pool_far.py`, `candidates.py` | Combining the two template sets, and checking candidates against known events. |
| `pipeline/ecc_*.py` | Making the eccentric templates and choosing where to place them. |
| `data/banks/` | The two template sets. |
| `data/segments/` | The exact stretches of data analysed in each week. |
| `data/o4_known.csv` | 13,225 known events and candidates from the catalogues and public alerts. |
| `ops/` | The scripts that ran the jobs on RunPod and Vast.ai. They depend on our cloud setup. |
| `tests/` | Self-tests. `test_coinc_rank.py` runs on a CPU. The others need a GPU. |
| `results/` | The O4a and O4b results, and our audit of the PyCBC comparison. |
| `docs/lab-notes/` | Notes written during the run: frozen settings, pre-registered tests, the pilot report and the bills. |

## Run it yourself

[REPRODUCE.md](REPRODUCE.md) lists the steps. You need Python 3.11, an NVIDIA GPU with CUDA 12 and at least
24 GB of memory, and the packages in `environment/`.

## How we made the results traceable

Each filter run recorded a fingerprint of its settings (`config_md5`) in its output. [docs/lab-notes/FREEZE.md](docs/lab-notes/FREEZE.md)
lists every frozen file, and every change made during the run with its reason. Runs on a GPU repeat to within
float32 rounding, but not bit for bit. We used week 1 of O4a to develop the pipeline, so no result uses it.

## How we built it

One software engineer wrote this code in a few days with AI coding agents (Claude). We checked each result by
measurement, and the lab notes record each check. If you find an error, open an issue.

## License

GPL-3.0-or-later, because the code imports PyCBC and LALSuite, which use the GPL. See [LICENSE](LICENSE).
The data comes from the Gravitational Wave Open Science Center (GWOSC). If you use it, cite GWOSC as its
terms ask.
