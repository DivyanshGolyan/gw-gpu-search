# O4a results: aligned + eccentric search, frozen procedure (2026-10-28)

Everything here is measured, except where marked inferred. Sources: results/coinc_rank/o4a/o4a_results.json (bench/o4a_results.py) and per-chunk results/o4a/CHUNK/rank_{aligned,ecc}/.

## Summary

- **Coverage.** 31 chunks ranked, which is all of O4a except week 1. Week 1 (wk01) is development data under FREEZE.md and is never used for a claim. Total coincident H1+L1 live time is **119.8 days**, against 13,124 years of time-slide background.
- **New candidates.** None below 1/yr: every one of the **51** candidates below 1/yr matches a catalogued event or GraceDB superevent.
- **Catalog recovery.** 48 of 68 GWTC-4.0/4.1 events with catalog FAR below 1/yr are found below 1/yr. The table below lists the 20 that are not.
- **Loudest uncatalogued candidate.** 7.3/yr (wk12). Over the whole run that is a false-alarm probability of 0.91, so it is consistent with noise.
- **Reported FARs.** Each is 2 x the lower of the two layers' FARs (frozen union bound), with one FAR per chunk. Backgrounds are not pooled: the pooling test failed in the pilot, so FARs stay per chunk.

## Sensitivity notes (no injections in production)

- **Held-out pilot (wk02, wk06, wk18; measured).** On LVK O4a sensitivity injections we find 0.935 as many as PyCBC, counted at FAR <= 1/yr with each pipeline's own FARs. By week: 0.976 in a clean week, 0.91 in glitchier ones. The gap is borderline signals, plus PyCBC's singles channel and our keep-loudest cap (about 1%).
- **Eccentric signals (pilot, measured).** With e20 >= 0.15, found volume is 1.29x [1.22, 1.37] that of the aligned search alone.
- **Code changes during the run.** The ranking speed-ups were adopted with identical outputs (FREEZE.md change log 19:20). The filter runner changes kept the same peak definition (15:55).

## Catalog events not recovered below 1/yr

| chunk | event | catalog FAR (/yr) | ours (union, /yr) | layers with a trigger |
|---|---|---|---|---|
| wk14 | GW231008_142521 | 0.0016 | 3.09 | aligned, ecc |
| wk21+wk22 | GW231127_165300 | 0.01 | 4.21 | aligned, ecc |
| wk02 | GW230712_090405 | 0.018 | 1.13 | ecc |
| wk20 | GW231119_075248 | 0.019 | 6.01 | aligned |
| wk14 | GW231005_091549 | 0.04 | 73 | aligned, ecc |
| wk21+wk22 | GW231129_081745 | 0.056 | 30.8 | aligned, ecc |
| wk08 | GW230825_041334 | 0.1 | 4.27 | aligned, ecc |
| wk14 | GW231004_232346 | 0.16 | - | none |
| wkm01 | GW230630_125806 | 0.16 | 24.1 | aligned, ecc |
| wk13 | GW230930_110730 | 0.17 | 64.1 | aligned, ecc |
| wk15 | GW231014_040532 | 0.21 | 3.68 | aligned, ecc |
| wk20 | GW231113_122623 | 0.28 | 19.9 | aligned, ecc |
| wk09 | GW230831_015414 | 0.29 | 44 | aligned, ecc |
| wk05 | GW230803_033412 | 0.31 | 7.44 | aligned, ecc |
| wk26 | GW231230_170116 | 0.42 | 1.86e+03 | aligned, ecc |
| wkm01 | GW230630_070659 | 0.47 | - | none |
| wk20 | GW231113_150041 | 0.5 | 308 | ecc |
| wk25 | GW231221_135041 | 0.54 | 25.5 | ecc |
| wk16 | GW231018_233037 | 0.68 | 27.3 | aligned |
| wk17 | GW231026_130704 | 0.76 | 30 | aligned, ecc |

- Most of these are marginal in the catalog (FAR 0.1-0.8/yr).
- The three that are louder in the catalog and that we put at 3-6/yr (GW231008_142521, GW231127_165300, GW231119_075248) need a per-pipeline check. GW231127_165300, for example, is 0.010/yr in cWB but only 0.73/yr in PyCBC, so the matched-filter pipelines also see it as weak. Inferred from the catalog candidate tables; not yet checked for the other two.
- GW230712_090405 reports at 1.13/yr, just over 1/yr because of the union factor. Its eccentric layer alone gives 0.565/yr.

## Every candidate below 1/yr

All are KNOWN. FAR is the union FAR, clamped at the chunk floor.

| chunk | GPS (H1) | union FAR (/yr) | aligned | ecc | H1 / L1 SNR | catalog match |
|---|---|---|---|---|---|---|
| wkm06+wkm05 | 1369694512.100 | 0.00398 | 0.00199 | 0.125 | 8.6 / 9.3 | GW230601_224134 |
| wkm04 | 1369983241.498 | 0.00435 | 0.00218 | 16.8 | 7.1 / 8.8 | GW230605_065343 |
| wkm04 | 1370047403.784 | 0.0174 | 0.0457 | 0.00871 | 8.8 / 5.4 | GW230606_004305 |
| wkm04 | 1370328616.269 | 0.00435 | 0.00218 | 0.0196 | 5.1 / 8.5 | GW230609_064958 |
| wkm02 | 1371641481.749 | 0.00487 | 0.00243 | 0.00974 | 7.8 / 6.4 | GW230624_113103 |
| wkm01 | 1371866035.820 | 0.00413 | 0.00206 | - | 14.2 / 10.3 | GW230627_015337 |
| wkm01 | 1372029138.361 | 0.00413 | 0.00206 | 0.00206 | 12.1 / 8.6 | GW230628_231200 |
| wkm01 | 1372203950.396 | 0.00413 | 0.00206 | - | 7.1 / 6.4 | GW230630_234532 |
| wk03 | 1374142732.443 | 0.00475 | 0.00238 | - | 7.8 / 6.7 | GW230723_101834 |
| wk04 | 1374654215.201 | 0.168 | 0.084 | - | 6.7 / 6.2 | GW230729_082317 |
| wk05 | 1375242187.967 | 0.0958 | 0.0479 | 0.132 | 5.9 / 7.1 | GW230805_034249 |
| wk06 | 1375389659.937 | 0.747 | 0.374 | - | 6.9 / 5.8 | GW230806_204041 |
| wk06 | 1375759294.281 | 0.00584 | 0.00292 | 0.00292 | 6.7 / 9.9 | GW230811_032116 |
| wk07 | 1376029178.459 | 0.0177 | 0.223 | 0.00883 | 7.1 / 6.6 | GW230814_061920 |
| wk08 | 1376601933.967 | 0.418 | 0.209 | 45 | 7.3 / 5.5 | GW230820_212515 |
| wk08 | 1376883065.740 | 0.00324 | 0.00162 | 0.0891 | 7.2 / 7.9 | GW230824_033047 |
| wk09 | 1377524799.743 | 0.481 | 0.24 | - | 5.4 / 7.2 | GW230831_134621 |
| wk10 | 1377839431.130 | 0.00422 | 0.00211 | - | 6.2 / 8.3 | GW230904_051013 |
| wk11 | 1378725259.719 | 0.00363 | 0.00181 | 0.00181 | 8.2 / 10.2 | GW230914_111401 |
| wk12 | 1379195850.020 | 0.0043 | 0.00215 | 0.00215 | 11.3 / 10.9 | GW230919_215712 |
| wk12 | 1379229102.721 | 0.0043 | 0.00215 | 0.00215 | 7.7 / 6.5 | GW230920_071124 |
| wk12 | 1379383442.893 | 0.0043 | 0.00215 | 0.00215 | 6.1 / 9.9 | GW230922_020344 |
| wk12 | 1379390836.089 | 0.0043 | 0.00215 | 0.0624 | 7.6 / 8.7 | GW230922_040658 |
| wk12 | 1379594711.831 | 0.0043 | 0.00215 | 0.00215 | 8.6 / 8.5 | GW230924_124453 |
| wk13 | 1379824667.113 | 0.0039 | 0.00195 | 0.00195 | 8.4 / 5.9 | GW230927_043729 |
| wk13 | 1379864330.915 | 0.0039 | 0.00195 | 0.00195 | 11.5 / 15.4 | GW230927_153832 |
| wk13 | 1379973525.129 | 0.0195 | 0.00975 | 0.0897 | 6.8 / 6.5 | GW230928_215827 |
| wk13 | 1380204158.723 | 0.0039 | 0.00195 | 0.117 | 7.4 / 7.3 | GW231001_140220 |
| wk14 | 1380507048.976 | 0.236 | 0.118 | - | 6.7 / 7.4 | GW231005_021030 |
| wk16 | 1381847405.583 | 0.00368 | 0.00184 | - | 10.2 / 6.5 | GW231020_142947 |
| wk17 | 1382542224.228 | 0.00591 | 0.00296 | 0.00296 | 12.2 / 18.0 | GW231028_153006 |
| wk18 | 1382944674.373 | 0.00657 | 0.00329 | 0.00329 | 9.4 / 10.3 | GW231102_071736 |
| wk18 | 1383140076.944 | 0.00657 | 0.00329 | - | 7.7 / 8.7 | GW231104_133418 |
| wk19 | 1383309965.896 | 0.56 | 0.28 | - | 8.0 / 5.5 | GW231106_124547 |
| wk19 | 1383483120.457 | 0.00431 | 0.00215 | 0.00215 | 7.5 / 8.8 | GW231108_125142 |
| wk19 | 1383624218.219 | 0.00431 | 0.00215 | 0.00215 | 6.5 / 8.0 | GW231110_040320 |
| wk20 | 1383941075.908 | 0.00559 | 0.00279 | - | 7.7 / 6.7 | GW231113_200417 |
| wk20 | 1383971549.249 | 0.00559 | 0.00279 | 0.081 | 7.6 / 6.5 | GW231114_043211 |
| wk20 | 1384304204.840 | 0.00559 | 0.00279 | 0.00279 | 7.5 / 6.7 | GW231118_005626 |
| wk20 | 1384326860.231 | 0.598 | 0.299 | 55.4 | 6.8 / 5.8 | GW231118_071402 |
| wk20 | 1384333580.008 | 0.00559 | 0.00279 | - | 7.6 / 7.5 | GW231118_090602 |
| wk21+wk22 | 1384482081.031 | 0.23 | 0.115 | - | 6.1 / 7.6 | GW231120_022103 |
| wk21+wk22 | 1384782888.620 | 0.188 | 0.0941 | 1.37 | 11.6 / 15.2 | GW231123_135430 |
| wk23 | 1385940712.519 | 0.007 | 0.0035 | 0.0035 | 7.0 / 8.7 | GW231206_233134 |
| wk23 | 1385941159.492 | 0.007 | 0.0035 | 0.0035 | 11.2 / 16.4 | GW231206_233901 |
| wk24 | 1386501275.915 | 0.00708 | 0.00354 | 0.00708 | 6.3 / 7.7 | GW231213_111417 |
| wk25 | 1387337334.048 | 0.0506 | 0.0253 | 0.0366 | 6.8 / 6.6 | GW231223_032836 |
| wk25 | 1387353073.316 | 0.0338 | 0.0169 | - | 7.0 / 6.3 | GW231223_075055 |
| wk25 | 1387421019.974 | 0.00563 | 0.00281 | - | 8.6 / 10.3 | GW231224_024321 |
| wk26 | 1387620938.299 | 0.00521 | 0.0026 | 0.0026 | 24.8 / 19.1 | GW231226_101520 |
| wk27 | 1388626353.719 | 0.0211 | 0.0106 | 30 | 8.0 / 5.7 | GW240107_013215 |

## Top 10 uncatalogued candidates

DQ columns:
- **gate_dt:** seconds to the nearest autogate.
- **env:** H1/L1 counts of SNR >= 8 peaks within ±8 s. -1 means not computed for that chunk.
- GWOSC CAT1/2/3 and hardware-injection flags remove 0 s of O4a (R8), so they flag nothing here.

| chunk | GPS (H1) | union FAR (/yr) | whole-run FAP | layer | H1 / L1 SNR | masses (det.) | gate_dt (s) | env H1/L1 |
|---|---|---|---|---|---|---|---|---|
| wk12 | 1379103856.239 | 7.29 | 0.91 | aligned | 6.2 / 6.1 | 24.6 + 8.6 | 985 | 0/0 |
| wk21+wk22 | 1385217680.752 | 8.65 | 0.94 | aligned | 6.0 / 6.1 | 10.7 + 9.1 | 239 | 0/0 |
| wk16 | 1381579885.080 | 16.1 | 0.99 | ecc | 5.7 / 6.6 | 53.1 + 18.0 | 82 | 0/0 |
| wk06 | 1375598432.646 | 20.1 | 1.00 | ecc | 5.3 / 6.5 | 35.5 + 16.2 | 1865 | -1/-1 |
| wk08 | 1376989505.557 | 25.6 | 1.00 | ecc | 5.6 / 6.4 | 36.4 + 10.4 | 141 | -1/-1 |
| wk13 | 1379944446.611 | 32.2 | 1.00 | aligned | 6.3 / 5.8 | 27.3 + 8.6 | 93 | -1/-1 |
| wk07 | 1376204312.424 | 49 | 1.00 | ecc | 5.5 / 6.0 | 28.8 + 17.2 | 724 | -1/-1 |
| wk03 | 1373869491.565 | 55 | 1.00 | ecc | 5.9 / 6.1 | 50.1 + 15.8 | 88 | -1/-1 |
| wkm02 | 1371614002.390 | 59.7 | 1.00 | ecc | 6.5 / 5.2 | 38.0 + 11.4 | 332 | -1/-1 |
| wk02 | 1373027927.491 | 60.6 | 1.00 | ecc | 4.7 / 6.7 | 31.7 + 14.6 | 929 | -1/-1 |

None is significant. The loudest, 7.3/yr, would be expected about 2.4 times by chance in 0.33 yr of data.

## Chunks

| chunk | live (d) | background (yr) | FAR floor (/yr) | gates |
|---|---|---|---|---|
| wk02 | 3.94 | 432 | 0.0023 | 326 |
| wk03 | 3.84 | 421 | 0.0024 | 370 |
| wk04 | 3.91 | 429 | 0.0023 | 336 |
| wk05 | 3.24 | 355 | 0.0028 | 392 |
| wk06 | 3.13 | 342 | 0.0029 | 250 |
| wk07 | 4.14 | 453 | 0.0022 | 329 |
| wk08 | 5.64 | 618 | 0.0016 | 425 |
| wk09 | 4.33 | 474 | 0.0021 | 316 |
| wk10 | 4.32 | 474 | 0.0021 | 282 |
| wk11 | 5.04 | 552 | 0.0018 | 355 |
| wk12 | 4.25 | 465 | 0.0022 | 283 |
| wk13 | 4.68 | 513 | 0.0020 | 326 |
| wk14 | 4.72 | 516 | 0.0019 | 348 |
| wk15 | 3.83 | 419 | 0.0024 | 243 |
| wk16 | 4.96 | 543 | 0.0018 | 376 |
| wk17 | 3.09 | 338 | 0.0030 | 197 |
| wk18 | 2.78 | 304 | 0.0033 | 201 |
| wk19 | 4.24 | 464 | 0.0022 | 284 |
| wk20 | 3.27 | 358 | 0.0028 | 226 |
| wk21+wk22 | 5.24 | 574 | 0.0017 | 389 |
| wk23 | 2.61 | 286 | 0.0035 | 215 |
| wk24 | 2.58 | 282 | 0.0035 | 151 |
| wk25 | 3.25 | 356 | 0.0028 | 198 |
| wk26 | 3.51 | 384 | 0.0026 | 266 |
| wk27 | 2.60 | 284 | 0.0035 | 277 |
| wk28+wk29 | 2.39 | 262 | 0.0038 | 552 |
| wkm01 | 4.42 | 484 | 0.0021 | 824 |
| wkm02 | 3.75 | 411 | 0.0024 | 524 |
| wkm03 | 3.37 | 369 | 0.0027 | 359 |
| wkm04 | 4.20 | 459 | 0.0022 | 937 |
| wkm06+wkm05 | 4.59 | 502 | 0.0020 | 909 |
