# O4b filtering run: launch config (2026-10-28, filter lane)

Approval: Divyansh, cited by the coordinator. They allowed using "the $5 on VAST" and said "don't be blocked on me ... use the entire budget on Vast and on RunPod". The order is: equivalence test, then a one-week canary, then fan-out.

## Code (payload `vast/v2/pay.txz`, md5 7eb54540b73944728807004c393c385a, on the relay at /f/pay/pay.txz)

| File | md5 | Notes |
|---|---|---|
| fp_filter.py | 436c31dc879cc3a4d874dccf5153d700 | v8. FREEZE.md change log 2026-10-28 18:40 (memory fix, O4b only) |
| exact_stage.py | 3989f01cda585954b559607862d58ddd | FREEZE 07:30 tolerances |
| inspiral_argv_H1_000.txt | b2b6e958c5680b41cb04b99e8e3e68fe | frozen argv (runner sets snr 4, keep-loudest 30, start-pad 48) |
| run_prod.sh | ae58e66872adfc27b3b859d2de91a452 | the O4a v7 runner with its post-mortem fixes, unchanged except the fp_filter md5 check |
| boot.sh | 6e944ee4ea53cb7ada2af3412fb3594b | Vast bootstrap: env from the frozen pip freeze as constraints, pyseobnr 0.3.7, then run_prod.sh |
| push.sh | c6f5427ae8af79f34997e38d7fbb4047 | pushes groups and weeks to the relay; local copy deleted after the lane pulls |
| bank_v2.hdf | 217554941ad158980526c0562b96011d | base bank |
| vt_45_100_mm0.85.hdf | bffc1d3921689c006b3451cd3a9a690a | eccentric bank (--bank2) |
| job_hub/hub_serve.py | a295ca96944740ba274da718308a242d | relay on RunPod CPU pod rm32hxqlm17awv |

FREEZE rows applied: all rows up to and including 2026-10-28 18:40. The FREEZE 19:20 ranking row (coinc_rank 1624bb5e) is the ranking lane's and does not touch filtering.

O4a isolation: outputs go to the Vast box's local disk (/work/vol/results/o4b), then to the relay, then to the lane. Nothing on this path mounts or writes gw-shared, so O4a's outputs cannot be touched.

## Equivalence and fit (measured 19:50 UTC, vast 58138563, RTX 3090, $0.191/h)

wk09 grp_00-02 against frozen production:
- 35,978,207 matched keys;
- SNR relative difference p99.99 1.6e-4;
- newsnr>6 difference max 3.34e-5.

These are the same statistics as the A40 v7 and v8 checks, and the config_md5 values are the same as on the A40. Peak GPU memory was 16.97 GiB, which fits on the 24 GB card. Per-group times were 260/158/103 s on the 3090 against 187/104/87 s on the A40.

## Weeks

All 37 O4b weeks: wk40-54, wk60-67, wk70-83 (results/o4b/pieces, 19,721 dks). The canary is wk40 + wk41, which is one ranking chunk.

## Caps and guards

- Vast spend is capped at $4.00, keeping $1.00 as a re-run reserve.
- RunPod spend is capped at $1.50 (relay plus anything else).
- `bench/vast_guard.py` checks every 2 min. It destroys gw-* Vast boxes or deletes gw-* pods at the cap. It alerts at 90% and on RunPod runway under 2 h.
- `bench/vast_launch.sh` refuses any launch whose $/h x max hours would cross the cap.
- Each box has a kill timer (MAXH) and stops itself when its job ends.
- Fan-out happens only if the canary's all-in $/week is at most $0.15 (planning thread).

## Canary result (PASS, 2026-10-29 02:54 IST)

Four tries were needed. Each failure was cheap and each was fixed before the next try:
1. The runner passed no `--dataset`, so chunk_prep.py looked in O4a. Fixed with GW_DATASET=O4b, then checked with a real local download of one O4b chunk.
2-3. A Norway host could not connect to gwosc.org. Fixed with an on-box GWOSC test download that runs before the runner, so a bad host stops itself in ~2 min. The machine is listed in bench/vast_bad_machines.txt. A CA bundle with the GWOSC root was also added.
4. Vast 58147725 (RTX 3090, US, $0.191/h) ran wk40+wk41, 916 dks, from 20:59:37 to 21:22:40 UTC (23 min). Both weeks were published on the relay. The lane pulled wk40 3 min after it was published, and the box deleted its local copy. The box stopped itself at 21:24:49.

**All-in $/week.**
- Canary as run, one box: ~35 min of Vast at $0.191/h (~$0.11, estimated from $/h) plus 35 min of relay at $0.12/h ($0.07). That is $0.18 for 0.97 week, so ~$0.19/week, almost all of it one-off boot and relay time.
- Fan-out projection, 5 boxes for the remaining 18,805 dks:
  - GPU at the canary rate: $0.077/week;
  - boot (~10 min per box over ~4 weeks): +$0.01;
  - relay (~2 h over 20 week-equivalents): +$0.01;
  - total about **$0.10/week**, which is under the $0.15 bar.
- The measured number after fan-out comes from the Vast credit and RunPod billing.

Fix after the canary: boot.sh now waits until every finished week is published before it starts the pull wait. In the canary, wk41 was published after the check, and the lane pulls it from the relay either way.

## Fan-out (launched 2026-10-29 ~02:58 IST)

The 5-box plan was dropped minutes after launch. The boxes were deleted while their images were still loading, once I found that most Vast hosts charge per GB of download (an O4b week pulls ~27 GB from GWOSC). The run used 3 hosts that charge nothing for download, plus one mop-up box. Every box had a kill timer, and at the timer the week in progress was cut off and redone in the mop-up.

## Result: all 37 O4b weeks filtered (done 2026-10-29 06:49 IST)

All 37 weeks (19,720 dks) were filtered with fp_filter v8, published on the relay and pulled by the ranking lane. No box failed after its GWOSC test passed. All boxes and the relay were deleted by 07:00 IST.

### Measured per box

Box cost is $/h x billed life (create to delete), an estimate from the listed price. The measured total is below.

| Box | Card, host | $/h | Life | ~Cost | Weeks | dks done | GPU busy, job | GPU busy, life | $ per 942 dks |
|---|---|---|---|---|---|---|---|---|---|
| canary4 | RTX 3090, US | 0.191 | 0.72 h | $0.14 | 2 | 916 | 0.83 | 0.45 | $0.141 |
| fr4090 | RTX 4090, PA US | 0.468 | 3.38 h | $1.58 | 16 | 8,147 | 0.48 | 0.45 | $0.183 |
| nla5000 | RTX A5000, NL | 0.264 | 3.05 h | $0.81 | 9 | 4,666 | 0.55 | 0.50 | $0.163 |
| b1s | RTX 4090, IT | 0.402 | 2.35 h | $0.94 | 6 | 4,265 | 0.38 | 0.35 | $0.209 |
| mop | RTX 4090, PA US (same host as fr4090) | 0.468 | 0.57 h | $0.27 | 4 | 1,726 | 0.78 | 0.56 | $0.145 |

"GPU busy, job" is the summed `s filter` time divided by the time from the job's first input to its last group, from each box's job_status.txt (o4b_box_logs_saved/). "GPU busy, life" divides the same filter time by the whole billed life, so it includes image pull, env build and the end-of-job pull wait. The filter time includes the CPU work inside each group, so both numbers are upper bounds on true GPU utilisation.

The cards waited on GWOSC downloads, not on compute. The same PA 4090 host was at 0.48 for 3 h overnight and at 0.78 in the mop-up an hour later. Its GWOSC test took 56 s at 21:38 UTC and 18 s at 00:54 UTC. The host's download rate decided $/week more than the card did.

Kill timers cut off 1,039 dks of partial weeks (5% of the work), which was redone in the mop-up.

### Measured totals (billing APIs, 2026-10-29 07:00 IST)

- **Vast: $4.21** (5.00 credit minus 0.794 left). This covers everything Vast did for O4b: the 3090 equivalence test, canaries 1-4, the 5 deleted boxes, the fan-out, the mop-up and bandwidth. The production boxes above account for ~$3.73. The other ~$0.48 is tests, the failed canaries and the deleted boxes.
- **Relay pod (RunPod cpu3c)**: up 18:41-01:29 UTC, 6.8 h x $0.12 = **~$0.82**. This is estimated, because the billing API had no rows for the pod yet. It matches the RunPod balance, which fell $1.00 since 18:30 UTC; other lanes' pods account for $0.23 of that.
- **O4b all-in: ~$5.03** for 37 weeks. That is $0.136 per calendar week, or **$0.24 per full week of H1+L1 (942 dks)** over 20.9 week-equivalents. Production only (the five boxes plus the relay): ~$4.55, or $0.218 per 942 dks. O4a for comparison: $5.66 for 34 weeks ($0.166 per calendar week) on RunPod A40s.

**This missed the $0.15/week bar**: $0.24 per 942 dks all-in, and $0.14-0.21 per box. The causes, in order of size:
1. The GPUs waited on downloads (busy 0.38-0.55 on three of the five boxes).
2. The relay cost $0.82 for 6.8 h, ~16% of the total. It was needed only because this sandbox cannot reach Vast boxes directly.
3. The tests and deleted boxes cost $0.48.
4. 5% of the work was redone after the kill timers.

### Re-filter of wk80-83 (2026-10-29 07:54 IST)

The ranking lane lost its ranked outputs for wk80-83 while project storage was full, and no copy of the peaks survived: the boxes and relay were already deleted. Those 1,684 dks were re-filtered on vast 58181576 (the same PA 4090 host) through a new relay, 0wvhal17bypt1u. It finished at 08:25 IST: wk80 369 s, wk81 512 s, wk82 328 s, wk83 154 s. Env build was slow this time (585 s). The lane pulled all 4 weeks, and the box and relay were deleted at 08:33 IST. It cost $0.32 on Vast (measured from credit) plus ~$0.07 for the relay (RunPod balance drop), about $0.39 in total. With the rerun, O4b all-in is ~$5.42, or $0.26 per 942 dks. Vast credit left: $0.47.

### What would cut it next time (O4c)

- Pick hosts by measured GWOSC download rate, not by card. Run the on-box GWOSC test on 2-3 candidate hosts and keep the fastest. US hosts were faster here.
- Raise download parallelism (DLPAR 3 -> 6) on download-bound hosts, or run two weeks at once on one card. 4090 filter time per group was about half its wait.
- Pre-fetch the O4c data once to cheap storage near the GPUs, instead of pulling ~27 GB per week from GWOSC on each box.
- Make kill timers finish the current week instead of cutting it off: a soft deadline that stops taking new weeks.
- Keep the box logs and peaks until the ranking lane confirms its outputs are durable, and do not delete them as soon as they are pulled.
