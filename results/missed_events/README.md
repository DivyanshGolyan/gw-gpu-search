# Why the search misses 42 catalogued mergers

The search finds 111 of the 153 catalogued mergers (GWTC-4.1 and GWTC-5.0) that the catalogues list with a
false-alarm rate below 1 per year. Most of the 42 misses are events that the other matched-filter pipelines
also found weak. For the mergers that PyCBC finds with high confidence, the search finds 99 of 101.

## Recovery depends on how confident PyCBC was

PyCBC is the standard pipeline closest to ours in method. The GWOSC event pages list each pipeline's own
false-alarm rate. The table groups the 153 events by PyCBC's rate.

| PyCBC false-alarm rate (per year) | Events | We find | Share |
| --- | --- | --- | --- |
| below 0.01 | 101 | 99 | 98% |
| 0.01 to 0.1 | 14 | 6 | 43% |
| 0.1 to 1 | 20 | 4 | 20% |
| 1 or more | 12 | 2 | 17% |
| PyCBC did not list it | 6 | 0 | 0% |

The catalogue includes an event if any one of four pipelines (PyCBC, GstLAL, MBTA, cWB) finds it below
1 per year. The same pattern shows when we count how many pipelines agree:

| Pipelines below 1 per year | Events | We find |
| --- | --- | --- |
| 4 | 71 | 69 |
| 3 | 44 | 37 |
| 2 | 14 | 3 |
| 1 | 24 | 2 |

So 33 of the 42 misses are events that only one or two pipelines found below 1 per year.

## Causes of the misses

- **Weak events.** 32 of the misses have a PyCBC rate of 0.1 per year or more, or no PyCBC entry. Several were
  found mainly by cWB, which looks for bursts of power without templates.
- **The factor of 2 for two template sets.** Our reported rate is twice the lower rate of the two template sets.
  Three events (GW230712_090405, GW240531_040326, GW241116_151753) fall below 1 per year in one template set,
  and only that factor puts them above 1 per year.
- **No Virgo data.** We use only the two LIGO detectors. In O4b, 79 of the 85 events have Virgo data, and the
  catalogue pipelines can use it. O4a had no Virgo data.
- **Ten events that PyCBC found below 0.1 per year.** These ten are real shortfalls against PyCBC:
  GW230630_125806, GW230803_033412, GW230831_015414, GW230930_110730, GW231008_142521, GW231113_122623,
  GW231119_075248, GW240621_214041, GW241116_151753, GW241229_155844. Eight of the ten were found below 1 per
  year by only one or two pipelines. Each has a trigger in our search, so none was outside the data we analysed.

We haven't checked these ten one by one. Two differences in method can explain part of the gap. PyCBC's O4a
false-alarm rate adds up the rates of all detector combinations, including single-detector events. PyCBC also
estimates its background over the whole run, while we estimate it for each week.

All 42 events fall inside our analysed time. 41 of them fall inside the mass range of our templates. The catalogue gives no masses for the other one, GW230630_070659.

## Files

- `explain_misses.py` makes the tables above. To fetch fresh data from GWOSC, run it with `--cacert BUNDLE`.
  To reuse the saved copy, run it with `--cached catalog_with_pipelines.json`.
- `catalog_with_pipelines.json` has, for each event, our false-alarm rate and the rate and SNR from each
  LIGO pipeline, fetched from the GWOSC event API on 2026-10-29.
