# O4b pieces (CPU, 2026-10-28)

All numbers are measured from the GWOSC H1_DATA and L1_DATA timelines.
The script is the unchanged `bench/make_pieces_o4a.py`, run with the O4b GPS range from the GWOSC runs API: `--start 1396417050 --end 1422118818`.
Same rules as O4a:
- max piece 11,000 s, min span 1,100 s, 176 s overlap
- pads 48/152/24 s
- `--store-dir /work/stores`

Week names continue the O4a numbering: wk40-wk83. They never collide with O4a names.
The gaps wk55-59 and wk68-69 are O4b breaks with no coincident data.

| | O4a | O4b |
|---|---|---|
| coincident spans | 700 | 849 |
| weeks with data | 35 | 37 |
| pieces | - | 1,395 |
| analysed coincident time | 125.2 d | 112.2 d |
| data per detector | 10,990,815 s | 9,860,332 s |
| **total detector-ks (H1+L1)** | **21,982** | **19,721** (0.897x O4a) |

Pricing O4b at 0.90x the measured O4a cost per detector-ks is inferred: it assumes the same filter rate.

Ranking chunks (chunks.json): 28 chunks. Weeks with under 2 days of live time are ranked with the smaller adjacent week in the same stretch, as for O4a. The merged pairs are wk40+41, 49+50, 60+61, 66+67, 70+71, 72+73, 77+78, 80+81 and 82+83. The smallest chunk is 2.31 d.

Files: weeks.json, chunks.json, wkNN/{datasets.txt, coinc_segments.txt, pieces.json}.
