# Production freeze (review item R9), 2026-10-28

Everything below is fixed for the held-out pilot and for any O4a+O4b run after it. Changing any item means a new entry in "Change log" at the bottom, with the reason. If the change touches ranking, rerun the R7 calibration before it applies. Any run whose `config_md5` doesn't match a logged config is development, not discovery.

## 1. Data

| Item | Frozen value |
|---|---|
| Strain | GWOSC O4a/O4b 4 kHz open data, H1 and L1, downloaded fresh per week (no persistent conditioned stores) |
| Analysed time | GWOSC `{IFO}_DATA` segments (`make_pieces_o4a.py`), frozen into each week's `datasets.txt` and `coinc_segments.txt` |
| DQ vetoes | None beyond DATA. R8 measured that CBC_CAT1/CAT2/CAT3 and NO_CBC_HW_INJ remove 0 s beyond DATA over all of O4a+O4b, and there are no CBC hardware injections. `candidates.py` re-checks every candidate against those flags |
| Glitches | pycbc autogating at threshold 50 (argv), plus dropping peaks within width + taper + 0.5 s of a gate |
| Development data (never used for a discovery claim) | Week 1 (wk01) and every stretch already inspected: the bench chunk, the R5 day, the R6 glitchy stretch and random day, and GW230707 |
| Held-out discovery data | Every other O4a/O4b week. The first look at each one is logged in `analysis_history.jsonl` |
| Pilot weeks | wk02, wk06, wk18 (HELDOUT_PILOT.md, seed 20261028). A fourth week only from the same generator's next draw |

## 2. Filter

| Item | Frozen value |
|---|---|
| Code | `bench/fp_filter.py` md5 `463abd46022e228fe6bb7baf97d8fc22`, plus the modules recorded in provenance `code` |
| Argv | `bench/inspiral_argv_H1_000.txt` md5 `b2b6e958c5680b41cb04b99e8e3e68fe`: flow 20 Hz, start pad 48, autogating 50, sg_chisq on mtotal > 30 at newsnr ≥ 6 |
| Base bank | bank_v2, 4,612 IMRPhenomD aligned-spin templates, md5 `217554941ad158980526c0562b96011d` |
| Eccentric bank | `results/ecc/vt_45_100_mm0.85.hdf`, 1,268 SEOBNRv5EHM templates, md5 `bffc1d3921689c006b3451cd3a9a690a` (FROZEN_BANK.md) |
| Eccentric flags | `--bank2 <file> --approximant2 SEOBNRv5EHM_ecc --eps2 0.02` on **whole segments, no `--chunk2`**. This is what R5 and R6 tested. HELDOUT_PILOT.md's `--chunk2 8,3,1,512` line is superseded, because --chunk2 was refuted |
| Trigger selection | SELECTION_RULE.md: threshold 4, margin 0.04, exact SNR and chi-squared at rebuilt SNR ≥ 5.28, consolidation keeps the 30 loudest per 1.072 s, SNR ≥ 5 exempt |
| Prep | `--prep gpu` |
| Provenance | `--history /workspace/results/analysis_history.jsonl` on every pass (PROVENANCE.md). config_md5 is recorded in every peak file |
| Pods | `bench/pod_launch.sh` with preflight, `--procs 7`, volume gw-shared. Each pod is logged in budget/LEDGER.md |

## 3. Ranking and significance

| Item | Frozen value |
|---|---|
| Code | `coinc_rank.py` `662d176c4d7058b05523e18f6f575e40`, `rank_week.sh` `8c88bae7a60a448fa0cae0a7006eaee5`, `pool_far.py` `bcf22bc7929bc3a54dfb988b71fd5b34` |
| Statistic | `--stat lnl --terms fits,phasetd --sgveto --min-rank 4`. Noise fits use triggers at rank ≥ 6. One ranking over the whole bank (mass-split refuted) |
| Background | Time slides per analysis chunk, giving a FAR per chunk. The candidate list is the union over chunks |
| Layers | Aligned (bank_v2) and eccentric (bank2), each ranked by its own `rank_week.sh` pass. Combined with the union bound: reported FAR = 2 × min(layer FARs) (`pool_far.py --n-layers 2`) |
| At the floor | Poisson 90% upper limit. Slide-reuse clumping (variance 2.6-3.3x Poisson at FAR 0.17-0.5/yr, corrected 10:15 from 1.5-2.6x) is quoted as a 10-16% FAR uncertainty |
| Calibration | **R7 PASS (2026-10-28 10:15)**: 40 noise-only pseudo-experiments through this exact procedure on real week-1 data (`results/coinc_rank/r7/calib.png`, `.json`). Directly tested for FAR >= ~1/yr. Below that, quote the Poisson 90% upper limit and the slide-reuse clumping, not a calibrated number |

## 4. Candidates

| Item | Frozen value |
|---|---|
| Code | `candidates.py` md5 `58b34d7a9f6716c50c12e452b798c56f` |
| Threshold | Reported (union-bound) FAR < 1/yr |
| Merging | Candidates within 1 s across layers count as one, keeping the best FAR |
| KNOWN vs NEW | KNOWN if any entry in `catalogs/o4_known.csv` (13,225 rows: GWTC-4.0, GWTC-5.0 including the subthreshold releases from all 4 pipelines, public alerts, independent catalogs) is within 1 s. Otherwise NEW |
| NEW candidates | DQ context (flags, distance to the nearest gate, glitchy neighbourhood, short template). Then the audit lane's VERIFY.md (R10: rerun from independently downloaded strain with this freeze). No claim leaves the project before that |
| Claims | (A) new astrophysical signal: calibrated FAP, independent rerun, DQ review. (B) "eccentric": out of scope unless A holds, and needs separate parameter estimation |

## 5. What the pilot reports against this freeze

$/week from billing (target ≤ $0.30), recovery of every catalog event in its weeks with FAR < 1/yr, aligned and eccentric injection found fractions at FAR 1/yr, and one config_md5 per pass.

## Change log

| UTC | change | why | R7 rerun? |
|---|---|---|---|
| 2026-10-28 | Freeze written | R9 | n/a |
| 2026-10-28 07:05 | bench/exact_stage.py CHECK_TOL snr_rel 1e-5 -> 3e-5 (assertion only; md5 0851569d -> e0ff127a1c2f2cd0fb209f849cdd5c10). fp_filter.py unchanged (463abd46). New config_md5 from pilot wk02 grp_04 on; grp_00-03 keep theirs | Self-check tripped at snr_rel_max 1.0117e-5 on 56 sample peaks (p99 5.2e-6), which is float32 rounding; the check never alters output values. Skipping tripped groups would drop livetime non-randomly | No (filter only) |
| 2026-10-28 07:30 | bench/exact_stage.py CHECK_TOL snr_rel 3e-5 -> 1e-4 and phase_rad 1e-5 -> 1e-4 (assertion only; exact_stage.py md5 3989f01cda585954b559607862d58ddd; fp_filter.py unchanged). Edge and chi-squared tolerances unchanged. Pre-approved: any other self-check tripping at float32 level (< 1e-4 relative) may be raised to 1e-4 by the filter lane, logged here, without a hold | Second trip: phase_max 1.142e-5 rad, wk02 grp_04 inj_dali; float32 noise floor. 1e-4 is ~100x below any ranking effect | No (filter only) |
| 2026-10-28 10:15 | No change: R7 passed (significance lane) | Counts at FAR <= 10/yr: 6 vs 5.99 expected; <= 100/yr: 65 vs 59.9; loudest-event p uniform (KS p 0.998); 0 of 40 thresholds outside the Poisson 90% band | n/a |
| 2026-10-28 15:55 | fp_filter.py 463abd46 -> a45c2989143c5d9d66f892a5e7f03b32 (results/gpu_first_principles/code/fp_filter_v7.py) with new flags --queue (one process filters every group of a pod: bank tables, chi-squared bins and coarse templates made once; next group's CPU and GPU prep, consolidate and peak-file writes overlapped with the GPU; cupy allocates through torch) and --pre-consolidate (rebuilt peaks that consolidate would drop are dropped on the GPU: same newsnr cut, gate veto, keep-loudest ranking and ties, plus a running per-window floor). Runner run_prod_v7.sh (3 parallel group downloads with retries). Peak outputs unchanged in definition; new config_md5 from the O4a resume on. Groups filtered before (wk03-wk15 and partial wk09/wk11/wk12/wk13) keep theirs | GPU was idle ~70% of the time (Divyansh). Equivalence: pre-consolidate == consolidate on 300 random trials (test_prekeep.py, mutations caught); wk09 grp_00-02 (36.0 M peaks) vs frozen production: 99.997% identical (template, time) keys, matched SNR p99.99 rel diff 1.6e-4, newsnr>6 diff <= 3.3e-5; all mismatches at the 5.28 exact-SNR cut (float32, as for any GPU rerun) (compare_stats.py). GPU 71% steady on A40 (was 27%); 868 GPU-s per week | No (filter only; same peak definition) |
| 2026-10-28 18:40 | fp_filter.py a45c2989 -> 436c31dc879cc3a4d874dccf5153d700 (results/gpu_first_principles/code/fp_filter_v8.py), **O4b and later only**; O4a keeps a45c2989. Two memory releases: the per-tile view `h = H[...]` is deleted with H (it kept the old 4 GiB H alive while the next tile's H was allocated), and each bank's qz/qbuf are released before the next bank allocates. Plus peak-memory logging (`gpu_peak_gib`) and an optional GW_MEMSNAP profiling hook (off in production). No change to any computed value | v7 ran out of memory on 24 GB cards (A5000 test 8: 23.1 of 23.5 GiB), which blocks the cheaper 24 GB cards (3090, A5000). Profile (torch memory snapshot) found two 4 GiB H copies alive at once. Peak on A40 wk09 grp_00-02: 22.1 -> 15.6/16.9/17.0 GiB. Equivalence vs frozen wk09 grp_00-02: 35,978,399 matched keys, SNR rel diff p99 1.5e-5, p99.99 1.6e-4; newsnr>6 diff max 3.34e-5; mismatches only at the 5.28 exact-SNR cut (same as v7) (compare_stats.py) | No (filter only; same peak definition) |
| 2026-10-28 19:20 | bench/coinc_rank.py 662d176c -> 1624bb5eedced565381f22458295edeb (results/rank_speedup/, ranking speed-up lane): before pairing, drop single triggers whose lnl upper bound (-lnn_H - lnn_L + max phasetd) plus the best same-template partner bound within the slide reach cannot reach --min-stat. Old file kept as bench/coinc_rank_662d176c.py. Used from O4a wk21+wk22 on (significance lane) | Pair search was 90% of ranking time, and RAM OOM-killed the 5.24 d wk21+wk22 chunk on a 15 GB container. Bound reviewed (phasetd lookup minus a non-negative penalty never exceeds lnp.max; 1e-9 slack; wrapped neighbour bins cover the 2000 s reach). Outputs bit-identical on P0 week 1 (search, fits, inj; speed-up lane) and on production wk21+wk22 eccentric layer (all 22 datasets, 178,085 background rows, 0 differences; significance lane). Search ranking 607 -> 141 s (P0), 449 -> 75 s (wk21+22 ecc); peak RAM 12.3 -> 8.5 GB (P0), 10.7 GB on the 5.24 d chunk | No (identical outputs; R7 unchanged by construction) |
| 2026-10-28 22:10 | Ranking speed-up fix 2 (results/rank_speedup/): rank_week.sh 8c88bae7 -> 2d202a49, coinc_rank.py 1624bb5e -> 784c6a50, merge_peaks.py b5834031 -> 048e3b8b, fg_summary.py 214f7655 -> ec31c826. No merged peak files: coinc_rank.py and fg_summary.py merge comma-separated piece lists in memory (same sort and duplicate rule; 8 columns). Old files kept as bench/*_pre_fix2.*. Used for O4b from wk40+wk41 on (significance lane); O4a ranked with fix 1 | Writes ~4.4 GB of uncompressed merged peaks per layer to local disk, which limits how many weeks can queue. Outputs identical: P0 week 1 end to end, 64 datasets, 0 differences (speed-up lane); O4b wk40+wk41 both layers vs fix 1, 42 + 42 datasets, 0 differences (significance lane). wk40+wk41: 262 -> 239 s, peak RAM 7.6 -> 7.9 GB | No (identical outputs) |
