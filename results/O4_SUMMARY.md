# O4a + O4b search summary (2026-10-29)

All numbers are measured. Details: [O4A_RESULTS.md](O4A_RESULTS.md) and [O4B_RESULTS.md](O4B_RESULTS.md).

| | O4a | O4b | total |
|---|---|---|---|
| chunks ranked | 31 | 28 | 59 |
| coincident live time (days) | 119.8 | 112.2 | 232.0 |
| candidates below 1/yr | 51 | 64 | 115 |
| of which NEW | 0 | 0 | **0** |
| GWTC events (catalog FAR < 1/yr) recovered below 1/yr | 48 / 68 | 63 / 85 | 111 / 153 |
| loudest uncatalogued (union FAR, /yr) | 7.3 | 4.7 | |

- **No new detection.** Nothing uncatalogued comes in below 1/yr.
- **Loudest uncatalogued trigger.** wk45 at GPS 1399562563.836, 4.7/yr (eccentric layer, 33+13 Msun detector frame). Over the 0.64 yr analysed here, its false-alarm probability is 0.95, so it is consistent with noise.
- **Procedure.** Aligned plus eccentric layers, combined as 2 x the lower layer FAR. FARs are per chunk, from 40,000 time slides. The cross-match covers GWTC-4.0/4.1/5.0, the candidate releases of all pipelines, and GraceDB O4a/b/c. Week 1 of O4a is development data and is excluded.
- **Sensitivity.** Measured only on the held-out O4a pilot: 0.935x PyCBC's injection recovery, and 1.29x the aligned-only found volume for eccentric signals with e20 >= 0.15. O4b sensitivity is not measured.
- **wk80-wk83 rerun.** The first ranked outputs of O4b chunks wk80+wk81 and wk82+wk83 were lost on 2026-10-29: the project share was full and dropped the copies, and a pull-loop bug wiped the local ones. Both causes are fixed. The four weeks were re-filtered (~$0.47) and re-ranked. The O4b totals match the readout made before the loss.
