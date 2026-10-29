# Independent audit of the PyCBC head-to-head (REPORT.md sections 2 and 3)

Date: 2026-10-27, about 17:40 UTC. Auditor: a separate thread that did not write the pipeline, plus a code-review helper that read the code cold. Scripts and logs are in `results/coinc_rank/audit/`. All numbers below are measured unless marked inferred.

## Verdict

**The headline holds as stated.** On these 3,318 injections at FAR < 1/yr, our search finds 1,574 and PyCBC's LVK entry finds 1,627. I got the same counts from the raw files, through three routes:
1. The LVK release file, re-downloaded from zenodo, for PyCBC's side.
2. The raw A8 peak files, with the pipeline rerun from scratch.
3. A coincidence, time-slide and FAR implementation I wrote myself, which reproduces the old-statistic count of 507 exactly.

There is no leakage, no unit or time-convention error, no difference in injection loudness, and no FAR bug worth more than about 0.1%.

**What "0.97" should be read as.** It is not a clean like-for-like ratio. PyCBC's published FAR is not the same quantity as ours: in O4a, PyCBC adds the false-alarm rates of every candidate type (H1L1, H1 alone, L1 alone) at the candidate's score. That makes PyCBC's FAR larger than a plain two-detector FAR, so the cut "PyCBC FAR < 1/yr" is stricter than ours. If that sum is 2 to 3 times the plain H1L1 FAR, the like-for-like ratio is **about 0.91 to 0.93**. The data can't pin down where in the range 0.91 to 0.97 the truth lies. Counting noise on top of that is ±0.009 (1σ).

The 7 event FARs in section 3 reproduce exactly. They carry two caveats: the statistic was partly tuned while looking at these events, and the top three sit at or near the background floor, so they are uncertain by a factor of about 2.

## Issues, ranked by how much they could change the headline

| # | issue | direction | size on the ratio 0.967 | status |
|---|---|---|---|---|
| 1 | PyCBC's FAR adds the FARs of single-detector candidate types; ours is H1L1 only | flatters us | down to 0.93 (if ×2) or 0.91 (if ×3) | confirmed in the GWTC-4 methods paper; size not measurable from the release |
| 2 | Counting noise on 3,318 injections | either way | ±0.009 (1σ); 95% range 0.949–0.985 | measured (bootstrap) |
| 3 | PyCBC can detect single-detector events and has a lower per-detector threshold; we need both detectors at newsnr ≥ 5 | flatters PyCBC | 19 of PyCBC's 1,627 are single-detector finds (1.2%); 33% of the 134 we miss have one detector below SNR 5 in PyCBC | measured. This is a real capability gap, not an accounting error |
| 4 | Injections were limited to our bank's mass range (m2 ≥ 8, m1 ≤ 200, q ≤ 8, detector frame) | flatters us relative to "PyCBC's VT" | none on this subset. The claim does not carry over to PyCBC's full mass range | by design; should be stated with the headline |
| 5 | sgveto was chosen after seeing these injections | flatters us | at most about 0.3% | measured: the sgveto gain holds on every half split (see below) |
| 6 | Gated time is not removed from live time | flatters us | about 0.2% of live time → about 0.1% of the count | measured: ≤ 488 s (H1) and ≤ 420 s (L1) of 470,819 s |
| 7 | Our background keeps loud real events; PyCBC removes events below 1/100 yr from its background | flatters PyCBC | +3 found for us if removed (1,577) | measured |
| 8 | `merge_peaks.py` misses most near-duplicate peaks (sorts by time first) | neither | none: duplicates hit zero lag and every slide the same way, and clustering removes them | measured: my de-duplication drops 3,645 vs its 540 in search H1, and the found counts are identical |
| 9 | Nothing checks that the injection run's settings match the search background | latent risk | none in this run (checked) | confirmed by reading; recommend an assert and logging all CLI args |

### 1. What PyCBC's FAR is (the one that matters)

GWTC-4.0 methods paper (arXiv 2508.18081, §3.4) says of the PyCBC offline search in O4a: "In times when multiple detectors are active, FAR estimates from all candidate types at the ranking statistic value of the candidate are added to give the final FAR estimate." In week 1 that means H1L1 coincidences plus H1-only plus L1-only candidates (Virgo was not observing). In engineering terms, PyCBC charges each candidate for three alarm channels and we charge for one. So at the same nominal 1/yr threshold, PyCBC is being held to a stricter bar.

How much stricter depends on how loud single-detector noise is at a given score. The release has no per-type breakdown, so I can only bracket it. Counting PyCBC at a looser threshold stands in for removing the extra channels:

| threshold | us | PyCBC | ratio |
|---|---|---|---|
| both at FAR < 0.5/yr | 1,500 | 1,558 | 0.963 |
| both at FAR < 1/yr (headline) | 1,574 | 1,627 | 0.967 |
| both at FAR < 2/yr | 1,649 | 1,696 | 0.972 |
| us < 1/yr, PyCBC < 2/yr (PyCBC FAR is 2× a plain H1L1 FAR) | 1,574 | 1,696 | 0.928 |
| us < 1/yr, PyCBC < 3/yr (3×) | 1,574 | 1,738 | 0.906 |

The ratio is flat when both thresholds move together, so the comparison is not balanced on a knife edge. The open question is only the offset between the two FAR definitions. The paper also says O4a PyCBC was a single analysis (two-detector background), not the O3-style broad + BBH + IMBH split, so there is no separate trials factor across searches.

**To settle it:** run PyCBC's own two-detector background on this week, or ask LVK for per-type FARs. Without that, quote "0.91–0.97 of PyCBC".

### 2. Counting noise

The ratio comes from about 1,600 found injections on each side. A paired bootstrap over injections gives 0.959–0.976 (68%) and 0.949–0.985 (95%). Half splits (measured):

| split | lnl | lnl + sgveto | PyCBC | ratio |
|---|---|---|---|---|
| even rows | 755 | 773 | 803 | 0.963 |
| odd rows | 787 | 801 | 824 | 0.972 |
| first half in time | 762 | 777 | 794 | 0.979 |
| second half in time | 780 | 797 | 833 | 0.957 |

### 5. Leakage: was anything tuned on the test injections?

- The noise fits come from the search run only. The injection run used the search run's fits and background (checked: matching fit files, 596.77 yr, 40,000 slides, zero background pairs built from the injection run).
- Nothing in the statistic reads injection or catalog times. The phasetd histogram is a Monte Carlo over isotropic sources, not over these injections.
- Fit pool 15 was picked by event-blind cross-validation on noise.
- The one choice made on these injections is `--sgveto`. It gains 34 and loses 2. The gain shows up in all four half splits (+14 to +18 each), so it is a real effect rather than a fit to noise. The selection bias from picking the best of 2–3 variants is at most a few injections (inferred).

## Checked and found correct

- **PyCBC flags.** All 3,318 rows of `lvk_week1_xphm.hdf` match the zenodo file exactly in parameters, `pycbc_far` and weights. 1,627 have `pycbc_far` < 1/yr (units 1/yr, inf = not found, per the release README). The same 1,627 pass with `<=`. The 32,843 injections in our segments and the 25,495 in-range candidates both reproduce. The random subset's PyCBC-found fraction (0.490) matches the pool's (0.479) within 1.4σ.
- **Times and segments.** Our H1 arrival time agrees with LVK's `time_H` to 1.3 µs. PyCBC's recovered time sits 2 ms from `time_geocenter`, so `time_geocenter` is the merger time and our ±0.1 s window is generous. All subset injections lie inside our coincident segments, ≥ 10 s from the edges.
- **Inf rows.** 121 injections have no PyCBC FAR at all (88 of them have none from any search). We find none of those, and none of our 81 "us only" are inf rows. So we gain nothing from times PyCBC didn't analyse.
- **Injection loudness.** On injections both searches found, our per-detector SNR over PyCBC's recovered SNR has median 1.004 (H1) and 0.993 (L1); network 0.9995. Our injections are not louder than LVK's. A 3% excess would have been worth about 9% in found count, so this was the biggest thing that could have gone wrong. Our f_lower of 15 Hz vs LVK's 16 Hz makes no measurable difference.
- **FAR arithmetic.** Recounting the background myself (stat ≥ candidate, divided by 596.77 yr) matches the code to 1e-14. The FAR = 1/yr threshold sits at stat ≈ 22.0. The background stored with `--min-stat 16` is exact there. Both searches use strict `<` and the same 1/yr.
- **Independent pipeline.** My own same-template coincidence, ±20,000 slides of 0.1 s on a gap-squeezed clock, 10 s clustering and FAR, run on the raw per-chunk peak files, gives **507 found with quadsum, identical to the report**. It also puts GW230707, GW230709, GW230702 and GW230708_230935 on top, at 3.7–6.0/yr, as the report's old statistic does. This confirms the plumbing (segments, slides, live time, association). It does not re-derive the lnl statistic itself.
- **Code review** (`audit/code_review.md`): slides match a brute-force circular search with no double counting; slide 0 is excluded from the background time; foreground and background are clustered identically; newsnr and sgveto match PyCBC's formulas; no sign error; the exclusive FAR is slightly conservative.

## The 7 real events (week-1 FARs)

Reproduced exactly from the raw A8 peaks with lnl + sgveto (exclusive FAR, /yr): GW230709_122727 < 0.0017, GW230702_185453 0.0017, GW230707_124047 0.005, GW230708_230935 0.28, GW230708_053705 4.4, GW230706_104333 4.4, GW230704_021211 37. Without sgveto: < 0.0017 / 0.005 / 0.013 / 0.31 / 5.7 / 4.4 / 25.

Caveats, largest first:
1. **Tuning on the same events.** Several choices were made while watching these events' FARs: fit threshold 6, rejecting `sens` and `lrate`, and keeping sgveto. Pool size was later confirmed blind. Treat these FARs as somewhat optimistic, more for the quiet four, where any one choice moves the FAR by about 2× (inferred from the ablation table). The injection head-to-head is the cleaner number, because those injections were used for only one choice.
2. **Floor-limited.** The top three are decided by 0–3 background coincidences out of 597 yr. A count of 1 is uncertain by a factor of about 2–3 either way.
3. **Background handling is not the issue.** Removing the three loud events from everyone's background, as PyCBC does, changes the others by under 2%.
4. **Do not compare with catalog FARs.** The catalog's come from PyCBC/GstLAL statistics with population and data-quality terms, plus far more background. The three events above 1/yr are catalog ≈ 0.2/yr events, which is consistent with the injection result that we lose about 3–9% of sensitivity at that level.

## What I'd change before quoting this outside the project

1. Quote the result as "0.91–0.97 of PyCBC's found count on in-bank-range injections, week 1", not "0.97 of PyCBC's VT".
2. Freeze the configuration and confirm the count on a second LVK subset (`lvk_inj_subset.py --seed 2`, different 100 s picks). That costs one more injection pass (about $0.8).
3. Make `coinc_rank.py` assert that the background file's settings match the injection run, and log all CLI args to the JSON.
4. Remove gated time from live time, and fix the `merge_peaks.py` sort order. Both are cosmetic for this result.
