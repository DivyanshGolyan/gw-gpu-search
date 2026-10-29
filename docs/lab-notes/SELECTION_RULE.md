# Single-detector trigger selection rule (review item R3)

This is the rule that `bench/fp_filter.py` and `bench/coinc_rank.py` apply with the production flags. The production flags are those of P0 week 1, `job_p1/run_weeks.sh` and `job_vt1`. Code lines refer to `bench/fp_filter.py` as of 2026-10-28.

## Production parameters

| Parameter | Value | Where set |
|---|---|---|
| SNR threshold `thr` | 4.0 | argv `--snr-threshold`, which the job's sed sets to 4 |
| Margin | 0.04 | `--margin` default |
| Exact cutoff | `--exact-snr 5.5` | job flags |
| Exact cutoff on the rebuilt SNR | 5.5 × (1 − 0.04) = **5.28** | |
| Chi-squared bins | `0.72*get_freq('fSEOBNRv4Peak',...)**0.7` | argv |
| sg_chisq | newsnr ≥ 6, templates with mtotal > 30 | `--sgchisq-snr-threshold 6`, `--sgchisq-locations "mtotal>30:..."` |
| Consolidation | drop newsnr < 4, then keep the 30 loudest by newsnr per 1.072 s. Peaks with snr ≥ 5 are exempt from the cap | `--newsnr-threshold 4 --keep-loudest-num 30 --keep-loudest-interval 1.072 --consolidate --consolidate-exempt-snr 5` |
| Coincidence | `--min-rank 4 --sgveto --stat lnl --terms fits,phasetd`, noise fits above newsnr 6 | `rank_week.sh` |

## What happens to each peak

Steps 1-3 happen per detector, per template tile and per segment.

1. **Candidate search (compressed filter).** The SVD+GEMM filter gives a rebuilt SNR, called `approx`. In each window of 64 samples × templates, the maximum becomes a candidate if `approx > thr × (1 − margin) = 3.84` (line 609).

2. **Rebuilt band: `4 < approx < 5.28`** (lines 628-643).
   - The SNR and coa_phase are the rebuilt values from the SVD reconstruction.
   - `chisq = 0`, `chisq_dof = 0` and `exact = False` are stored.
   - `sg_chisq = 1` and `sg_chisq_computed = False` are stored. This is PyCBC's "not computed" convention.
   - So **newsnr = snr**, with no chi-squared penalty and no sine-Gaussian penalty.
   - These peaks can never reach the noise-fit threshold of newsnr 6 or the sg_chisq threshold of 6, because their snr is below 5.28.

3. **Exact band: `approx ≥ 5.28`** (lines 644-742).
   - The exact template's complex SNR is recomputed at the peak sample, from the regenerated template.
   - The peak is kept if the exact `snr > 4`.
   - The power chi-squared is computed with PyCBC's bins: `chisq = nbins·Σ|z_bin|² − |z|²`, `dof = 2·nbins − 2`.
   - `newsnr` is PyCBC's: snr / ((1 + rchisq³)/2)^(1/6) where rchisq > 1.
   - If newsnr ≥ 6 and the template has sg locations (mtotal > 30), sg_chisq is computed. Otherwise sg_chisq stays 1.
   - `exact = True`.

4. **Consolidation** (lines 535-560, after every tile). This is pycbc_inspiral's `consolidate_events`. Peaks with newsnr < 4 are dropped, then the 30 loudest by newsnr are kept per 1.072 s. Peaks with snr ≥ 5 are always kept and do not count toward the 30. Rebuilt peaks enter this step with newsnr = their rebuilt snr.

5. **Autogating veto at write** (lines 766-768). Peaks within width + taper + 0.5 s of an autogate are dropped.

6. **Coincidence** (`coinc_rank.py`).
   - Single-detector rank: newsnr, then the sine-Gaussian veto, `rank /= sqrt(max(1, sg_chisq/4))`. That is a no-op for rebuilt peaks and for any peak without an sg computation.
   - Singles with rank < 4 are dropped.
   - lnL uses per-template exponential noise fits made from triggers with rank ≥ 6. Below 6 the fitted exponential is extrapolated (`ranking_stat.ln_noise`). Rebuilt peaks therefore never shape the fits, only use them.

## Foreground, background and injections use the identical rule

- Steps 1-5 run on one detector's data at a time, with no knowledge of the other detector or of time shifts. The `--gate` follow-up gate is also per detector (`make_gate.py`: the same newsnr ≥ thr rule on every detector and every time).
- Time-slide background in `coinc_rank.py` shifts the same single-detector trigger files that zero-lag uses, so every slid coincidence is built from triggers selected by exactly the foreground rule.
- The injection pass runs `fp_filter.py` with identical flags (only `--injection-file` is added). Found injections are ranked against the search pass's background and fits (`--background-from`, `--fits-from`).
- The eccentric second bank (`--bank2`, `peaks_b2`) runs through the same code path with the same thresholds. It gets its own `rank_week.sh` run, so it has its own background.

## Known properties of the boundary (by design; the same in foreground and background)

- **The 5.28 cut is on the rebuilt SNR, not the exact SNR.** P0 week 1 measured rebuilt/exact for 6.09M exact candidates: median 1.000, p1 0.985, p0.1 ≈ 0.958, min 0.79. So nearly every peak with an exact SNR ≥ 5.5 gets the exact treatment. About 1 in 1,000 is reported with its rebuilt SNR and no chi-squared, because its rebuilt value fell below 5.28. Peaks with an exact SNR between about 5.2 and 5.5 can land on either side.
- **Rebuilt peaks carry no chi-squared penalty.** A clean rebuilt peak at 5.2 ranks above an exact peak at 5.4 whose chi-squared is poor. This changes sensitivity, not calibration: background is measured with the same mixture. A rebuilt peak can never reach a single-detector rank of 6, so the loud tail that sets FAR at 1/yr and below consists entirely of exact, chi-squared-tested triggers.
- **Low edge.** Rebuilt peaks need `approx > 4` strictly. A true SNR-4.05 peak whose rebuilt value reads 3.99 is lost, while the exact path keeps anything with an exact snr > 4. At most this trims the 4.0-4.2 band by the rebuilt error (median 1e-3, p1 1.5%).
- **Volume of each band (P0 week 1, injection pass, all templates).** 337M rebuilt peaks and 6.09M exact candidates went in; 325M were removed by consolidation.

## How to change it

Every value above is a command-line flag. `--exact-snr 0` makes every peak exact (PyCBC-equivalent; its cost is not measured here). The rule is recorded per run in the provenance block (see `PROVENANCE.md`, review item R9).
