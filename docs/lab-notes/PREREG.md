# Eccentric add-on: pre-registered day-1 test (R5) and replication (R6)

Written 2026-10-28 ~03:05 UTC, before any eccentric search or injection pass has run. Follows the external review (Divyansh's project-chat message of 02:53) and cascade/O4_RUN_PLAN.md R5/R6. Any change after the first R5 result must be written below as a dated amendment with its reason, and the original rule still gets reported.

## Question

Does adding a coarse eccentric template bank to the aligned search, at the same total false-alarm rate, find more eccentric signals than the aligned search alone, without losing circular ones? This is an engineering go/no-go for spending ~$13 on O4a+O4b. It does not measure astrophysical sensitivity to a real eccentric population, and it says nothing yet about whether any candidate is actually eccentric.

## Fixed inputs

| Item | Value |
|---|---|
| Data | Day 1 of week 1, H1+L1: datasets_week1 lines starting within 86.4 ks of the earliest, coincident segments from results/slice1/coinc_segments.txt clipped to that span (job_vt1/job.sh) |
| Filter | Production base config (job_p1 flags, --prep gpu, 20 Hz). Eccentric bank via --bank2, --eps2 0.02, --chunk2 8,3,1,512 |
| Aligned bank | bank_v2.hdf (4,612 IMRPhenomD) |
| Eccentric bank | vt_45_100_mm0.85.hdf: SEOBNRv5EHM (2,2), 1,268 templates, Mtot 45-100, q ≤ 4, m2 ≥ 8, chi ±0.5 (equal), e20 0.15-0.45, MM 0.85; md5 bffc1d3921689c006b3451cd3a9a690a. Coverage checked before launch (R2-lite, see below) |
| Injections | injections/ecc_day1/inj_ecc_day1.hdf, md5 c64d210e6ef5c88847c8ee191dff3da5, 1,310 SEOBNRv5EHM all-mode, frozen (SPEC.md) |
| Ranking | bench/rank_week.sh frozen settings, run separately on bank-1 peaks and on bank-2 peaks (each with its own time-slide background) |
| Readout | bench/ecc/vt_or.py, FAR in per year, match window 0.1 s on H1 arrival |

**Near-threshold weighting.** Injection distances follow expected network SNR log-uniform in 5-30, so most injections sit near threshold. Each carries the volume weight vt_weight = d³ ln(d_hi/d_lo)/3. VT is the weighted found count.

**Combination.** OR: an injection is found if aligned FAR ≤ x/2 or eccentric FAR ≤ x/2. Compared against aligned alone at FAR ≤ x. Primary x = 1/yr, secondary x = 0.1/yr.

## Strata, all reported

- **Primary:** Mtot 45-100 (the eccentric bank's range), e20 ≥ 0.15.
- **Target bins:** e20 0.15-0.25, 0.25-0.35, 0.35-0.45, each for Mtot 45-100.
- **Quasi-circular control:** e20 < 0.10, Mtot 45-100.
- **Incremental bin:** e20 0.10-0.15.
- **Outside-bank control:** Mtot 30-45, all e20. Expect OR/aligned ≈ 0.95-1.0 here, because the eccentric bank adds trials but no coverage.

Each stratum reports: n; paired outcomes on the same injections and weights (aligned only, eccentric only, both, neither); weighted VT for each search; the OR/aligned ratio and OR minus aligned, each with a paired bootstrap 90% CI (2,000 resamples over injections).

## Decision table (day 1, FAR 1/yr)

| Outcome | Rule |
|---|---|
| **GO** | Primary OR/aligned ≥ 1.10 **and** its bootstrap 5th percentile > 1.00 **and** control (e20 < 0.10) OR/aligned ≥ 0.95 |
| **NO-GO** | Primary OR/aligned bootstrap 95th percentile < 1.05, **or** control OR/aligned < 0.90 |
| **INCONCLUSIVE** | Anything else. Next step is R6 (more data), not a different statistic |

Secondary checks (reported, not gating): FAR 0.1/yr version of the table; the trend across target bins (expected gain rising with e20, from the 50-55 box FFs: ~1.1, 1.4, 1.8 on signal strength alone); eccentric-only VT/aligned.

## Background breakdown (reported, with one flag)

For the eccentric ranking's time-slide background above the stat at FAR 1/yr, give counts by template Mtot (45-60/60-80/80-100), e20_est (three target bins), template duration (IMRPhenomD duration terciles) and chi-squared (reduced chi-squared < 1.5 / 1.5-3 / > 3). Do the same for bank-1 over the same masses.

**Flag:** more than half the eccentric background above that threshold comes from one e20 bin, or the eccentric bank's background rate above FAR-1/yr-equivalent stat is more than 3x the aligned bank's per template. Either means glitch-matching by bursty templates. It doesn't change the day-1 decision, but it must be addressed (veto or chi-squared tuning) before R6.

## Coverage prerequisite (R2-lite, before the pod)

ff_ecc_bank.py on ≥ 80 held-out SEOBNRv5EHM all-mode targets in Mtot 45-100. These are drawn at fixed physical e20 at 20 Hz, with relativistic anomaly uniform, independent of the bank's base shapes. **Launch only if** median FF ≥ 0.88 and ≤ 10% of targets fall below 0.80. Otherwise densify the bank first. The full R2 (several thousand targets, by region, fraction below 0.85/0.80) runs on CPU in parallel and gates production, not this test.

## R6 replication (defined now)

- **Stretch A (glitchy):** GPS 1372787464-1372797722 (week 1, where the precessing lane's gain disappeared).
- **Stretch B (random held-out day):** day index 6 of week 1, GPS 1372867552-1372953952. It was drawn with `random.Random(20261028).choice([1, 2, 3, 4, 6])`, which excludes day 1 (used) and day 5 (contains stretch A).
- Each gets a new frozen injection set with the same generator and population (new seed, new file name), the same banks and the same readout.
- **Replication failure:** in either stretch, primary OR/aligned point estimate < 1.00, or control OR/aligned < 0.90. Pooled over day 1 + A + B, the GO rule must still hold. If day 1 is GO and R6 fails, the add-on does not go to production and the failure is reported as the result.

## What a GO does and does not establish

A GO says the coarse bank plus split FAR adds detectable volume for SEOBNRv5EHM signals on this data. It does not establish sensitivity to SEOBNRv6EHM or v6EPHM signals (R4 checks that). It also doesn't establish that a found candidate is eccentric: precession can mimic eccentric morphology, so that needs separate parameter estimation with v6EPHM. And it doesn't calibrate the whole-search FAR across O4a+O4b (R1).

## Results log

- 2026-10-28 03:05, R2-lite coverage (measured, `ff_vt_45_100.json`): 80 targets, Mtot 45-100. Eccentric-bank FF median 0.913 (p5 0.855); 1.3% below 0.80 and 3.8% below 0.85. Aligned bank median 0.830. By e20 bin, ecc/aligned median FF is 0.907/0.895, 0.917/0.840 and 0.912/0.772, and SNR³ ratio is 1.03, 1.29 and 1.60. **Prerequisite met; launch allowed.**
- 2026-10-28 03:2x, full R2 coverage (measured, `r2_vt_45_100_n2000.json`, seed 202, held out from the bank and from the 80-target gate). 1,972 SEOBNRv5EHM all-mode targets with random orientation, at fixed physical e20 at 20 Hz, anomaly uniform; 28 failed to generate. FF median 0.916, 4.4% below 0.85, **0.3% below 0.80**. Flat in relativistic anomaly: by quadrant, median 0.915-0.916 and 3-5% below 0.85. That answers the review's doubt that mass steps cover anomaly. Weak spots: q 3-4 (median 0.892, 11% below 0.85), e20 0.35-0.45 (8% below 0.85, 0.9% below 0.80) and Mtot 45-55 (8% below 0.85). High e plus low mass: 172 targets, median 0.900, 2.9% below 0.80. SNR³ gain over aligned by e20 bin: 1.04 / 1.28 / 1.62. **R2 passes for the day-1 bank; the day-1 result is not provisional on coverage.** A production bank should densify q > 3.

## Amendments

- 2026-10-28 03:20. Run 1 (pod 1wiml9ppmbhxvt) crashed before producing any result. The cause was the eccentric bank's band-limited chunks (`--chunk2 8,3,1,512`), which trip the exact stage's chi-squared bin-width assertion (min 8 < 32 bins). The filter lane owns this. The eccentric bank now runs on whole segments (drops `--chunk2`). That affects filter cost only, not statistics or sensitivity, and the decision table is unchanged.
- 2026-10-28 03:30. Run 2 (pod zj0i1tzwro7vvh) completed the search pass: 328 s on an A5000, of which the eccentric bank added +80 s, with r/N 0.147. The injection pass then crashed. pyseobnr's Nyquist check rejects the (4,4) ringdown of light systems at the 2048 Hz injection rate; this was reproduced locally. The job now injects from a copy of the frozen file with the static arg `lmax_nyquist=1` added. That only switches off the check: parameters and modes are unchanged, and the frozen file itself is unchanged. The R6 sets get the same patch. The decision table is unchanged.
- 2026-10-28 03:45, **R5 day-1 result: GO** (measured, `vt1_day1/vt_day1.json`, `vt1_day1/bg_breakdown.txt`; run 3, pod igcp4d1hvvunc4). At FAR 1/yr, Mtot 45-100:

| Stratum | n | found aligned / OR | OR/aligned VT [90% CI] |
|---|---|---|---|
| **Primary, e20 ≥ 0.15** | 631 | 355 / 397 | **1.38 [1.19, 1.63]** |
| e20 0.15-0.25 | 184 | 113 / 119 | 1.12 [1.02, 1.27] |
| e20 0.25-0.35 | 238 | 134 / 148 | 1.53 [1.09, 2.26] |
| e20 0.35-0.45 | 209 | 108 / 130 | 1.49 [1.20, 1.92] |
| **Control, e20 < 0.10** | 174 | 100 / 102 | **1.04 [1.00, 1.10]** |
| Incremental, e20 0.10-0.15 | 91 | 45 / 47 | 1.01 [1.00, 1.02] |
| Outside bank, Mtot 30-45 | 414 | 224 / 233 | 0.89 [0.68, 1.10] (weighted; counts +9) |

  - At FAR 0.1/yr the primary is 1.38 [1.18, 1.64] and the control is 1.00.
  - GO rule: primary ≥ 1.10 with p5 > 1.00 **and** control ≥ 0.95. **Met.**
  - Background flag not raised. The largest share of eccentric background above its 1/yr stat comes from e20 0.25-0.35, at 35 of 83 (42%). Per template above a common stat (the aligned 1/yr stat), the eccentric bank's rate is 1.27x the aligned bank's. Against aligned templates of the same mass, 45-100, it is about 1.0x.
  - Chi-squared breakdown is not available: background rows in search.hdf carry no chi-squared, and this gets added for R6.
  - Filter cost on an A5000 (run 2): the eccentric bank adds +80 s per day of data to 247 s.
  - Status: one day of data. R6 replication on stretches A and B is next, as pre-registered.
- 2026-10-28 04:15, **R6 replication: PASS; pooled GO holds** (measured, `r6/pooled.txt`, `r6/A/vt_A.json`, `r6/B/vt_B.json`; pod 4drmoh3tynifv7). Values at FAR 1/yr, Mtot 45-100. Paired counts are on the same injections: "OR only" means found only by the combination, "aligned only" means found only by aligned.

| Set | Primary e20 ≥ 0.15: OR/aligned [90% CI] | paired OR only / aligned only | Control e20 < 0.10 |
|---|---|---|---|
| Day 1 | 1.38 [1.19, 1.62] | 43 / 1 | 1.04 |
| A, glitchy (10.7 ks) | 1.24 [1.00, 1.99] | 4 / 0 | 1.00 |
| B, random day 6 (74.7 ks) | 1.07 [1.01, 1.15] | 25 / 2 | 1.00 |
| **Pooled** | **1.22 [1.13, 1.33]** | 72 / 3 | **1.01 [0.99, 1.03]** |

  - Pooled by e20 bin: 1.15 / 1.19 / 1.32 at 1/yr, and 1.08 / 1.44 / 1.27 at 0.1/yr. Pooled primary at 0.1/yr is 1.26 [1.15, 1.39].
  - Replication-failure test: in either stretch, primary < 1.00 or control < 0.90. **Not triggered.** The pooled GO rule (≥ 1.10, p5 > 1.00, control ≥ 0.95) **holds**.
  - Honest reading: the day-1 effect (1.38) was on the high side, and day 6 gives 1.07. Use the pooled ~1.2x (90% CI 1.13-1.33) as the expected VT gain for e20 ≥ 0.15 over Mtot 45-100. On counts, the combination finds 72 injections that aligned alone misses and loses 3.
- 2026-10-28 05:00, R4 (independent-waveform check, TEOBResumS-Dali as stand-in for unreleased v6): **pass**, see R4_TABLE.md.

## Placebo-layer control (pre-registered 2026-10-28 ~10:10Z, before launch)

**Question.** Is the R5/R6 gain eccentricity, or just "a second, heavy-mass layer with its own keep-loudest budget helps" (GW230712_note.md §3)?

**Placebo bank.** bank_v2 rows with Mtot 45-100 (detector frame): 681 of 4,612 templates, md5 8f93d20907c9df7e8a7534131aec9c5c, approximant IMRPhenomD (aligned, quasi-circular). Same `--bank2` path, whole segments, `--eps2 0.001` (the IMR bank's base tolerance; the 0.02 eccentric tolerance is for the rescaled SEOBNRv5EHM solve). Same day-1 data, same `inj_ecc_day1.hdf` (+ lmax_nyquist patch), same rank_week.sh settings, same OR at FAR/2 readout via vt_or.py. Known asymmetry: 681 placebo templates vs 1,268 eccentric (fewer trials, so if anything it favours the placebo).

**Readout (FAR 1/yr, Mtot 45-100; 0.1/yr reported alongside).**
1. Placebo OR/aligned with 90% CI on: primary e20 ≥ 0.15; the three e20 bins; control e20 < 0.10.
2. Paired comparison on the same injections: found by eccentric-OR but not placebo-OR, and the reverse; bootstrap CI of VT(OR_ecc)/VT(OR_placebo).

**Decision (missed-mergers lane's rule, verbatim).** "If placebo OR/aligned on e20 ≥ 0.15 is within the CI of the eccentric layer's 1.38, the eccentric claim is reduced to 'a second, heavy-mass layer helps'." The eccentric day-1 CI is [1.19, 1.63], so: placebo primary ≥ 1.19 → **reduced claim**. Placebo primary < 1.19 → the eccentric claim stands, with the placebo's gain reported as the part of the effect that is layer/thinning rather than eccentricity. Secondary (not decisive): a placebo gain flat in e20 vs the eccentric layer's rising gain supports eccentricity as the cause.

**GW230712 check.** The placebo bank on the wk02 group containing GPS 1373187863.93, same ranking: report whether it forms the coincidence and at what FAR (eccentric layer: 0.56/yr exclusive). Run by whichever pod has the wk02 stores.

**Cap.** $0.30, single pod, logged in budget/LEDGER.md.
- 2026-10-28 09:55, **Placebo-layer control, day 1: eccentric claim stands** (measured, `placebo/day1/compare.txt`, `placebo/day1/vt_placebo_day1.json`; pod 2mz2rkrp51a8c7, A40, ~$0.15). Same injections and same aligned ranking (aligned FAR identical on 100% of injections). FAR 1/yr, Mtot 45-100:

| Stratum | OR/aligned, eccentric layer | OR/aligned, placebo layer | paired ecc-OR only / placebo-OR only | VT ecc-OR / placebo-OR |
|---|---|---|---|---|
| **Primary e20 ≥ 0.15** | 1.38 [1.19, 1.64] | **1.01 [0.94, 1.08]** | 40 / 2 | **1.37 [1.18, 1.62]** |
| e20 0.15-0.25 | 1.12 | 1.05 | 5 / 0 | 1.07 [1.01, 1.15] |
| e20 0.25-0.35 | 1.53 | 1.00 | 14 / 2 | 1.53 [1.08, 2.29] |
| e20 0.35-0.45 | 1.49 | 0.97 | 21 / 0 | 1.53 [1.22, 2.05] |
| Control e20 < 0.10 | 1.04 | 1.03 | 1 / 1 | 1.01 |

  - Pre-registered rule: placebo primary ≥ 1.19 → reduced claim. Placebo primary is 1.01 (p95 1.08). **Not triggered; the eccentric claim stands.** At 0.1/yr: placebo 1.04 [1.00, 1.10] vs eccentric 1.38.
  - Secondary: the placebo's small gain is flat in e20 (and appears equally in the control), the eccentric layer's rises with e20. Supports eccentricity as the cause.
  - Side note: the placebo layer alone (its own ranking at full FAR) finds 6-7% more than aligned alone (ecc/al 1.07), which is the thinning/own-budget effect from GW230712_note.md §3. It does not survive the FAR/2 split, so it is not what drives the OR gain.
  - wk02 GW230712 placebo check (GPU search lane, ~$0.03, `results/o4a/wk02/rank_placebo/README.md`): the placebo layer forms the coincidence (quadsum newsnr 8.40 vs eccentric 9.37; aligned none). Group-only ranking puts neither layer above min-stat 16, so FARs are not comparable to the week's 0.56/yr. Week-level placebo FAR not run (cap). Total placebo spend ~$0.18 of $0.30.
