#!/bin/bash
# Rent one Vast.ai GPU instance for the filter runner (no volume; payload and results through the RunPod relay).
# Usage: vast_launch.sh OFFER_ID INST MAXH ENVFILE      ENVFILE: lines KEY=VALUE for boot.sh / run_prod.sh (WEEKS, G, ...)
# Needs: the relay up (HUB in ENVFILE), pay.txz uploaded to HUB/f/pay/pay.txz with md5 in $PAY_MD5, vast_guard.py running.
# Refuses (exit 4) if the offer's $/h x MAXH plus bandwidth (BW_GB of download, inet_down_cost) would take Vast spend past the
# guard's cap (budget/guard_state.json, < 5 min old), or if the host charges more than MAX_DOWN_COST $/GB download (default
# 0.001: an O4b week pulls ~27 GB from GWOSC, so a $0.01/GB host adds $0.27/week, more than the GPU).
set -euo pipefail
OFFER=$1; INST=$2; MAXH=$3; ENVF=$4; D=$(cd "$(dirname "$0")" && pwd); G=$D/../budget/guard_state.json; TPL=$D/vast_onstart.sh
: "${PAY_MD5:?}"
python3 - "$G" "$OFFER" "$MAXH" <<'PY' || exit 4
import json, os, subprocess, sys, time
g, offer, h = sys.argv[1], sys.argv[2], float(sys.argv[3])
s = json.load(open(g)); age = time.time() - os.path.getmtime(g)
if age > 300: sys.exit(f'guard_state.json is {age:.0f} s old: is vast_guard.py running?')
q = json.dumps({'ask_contract_id': {'eq': int(offer)}})
r = subprocess.run(['curl', '-s', '--max-time', '30', '-G', 'https://console.vast.ai/api/v0/bundles/', '--data-urlencode', 'q=' + q], capture_output=True, text=True).stdout
offers = json.loads(r).get('offers') or []
if not offers: sys.exit(f'offer {offer} not found or not rentable')
dph = offers[0]['dph_total']; down = offers[0].get('inet_down_cost') or 0; up = offers[0].get('inet_up_cost') or 0
gb = float(os.environ.get('BW_GB', 30))      # GWOSC download for the job (~27 GB per 942 dks week of H1+L1); set it per job
if down > float(os.environ.get('MAX_DOWN_COST', 0.001)): sys.exit(f'host charges ${down:.4f}/GB download: {gb:.0f} GB would add ${down * gb:.2f}')
need = s['vast_spend'] + s['vast_burn_per_h'] * h + dph * h + down * gb + up * gb / 15
print(f'spend ${s["vast_spend"]:.2f} + running ${s["vast_burn_per_h"]:.2f}/h x {h} h + new ${dph:.3f}/h x {h} h = ${need:.2f} vs cap ${s["vast_cap"]}', file=sys.stderr)
sys.exit(0 if need < s['vast_cap'] else 'would cross the cap: not launched')
PY
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
python3 - "$TPL" "$ENVF" "$INST" "$MAXH" "$PAY_MD5" > "$TMP/body.json" <<'PY'
import json, sys
tpl, envf, inst, maxh, md5 = sys.argv[1:]
env = open(envf).read().strip() + f'\nINST={inst}\nMAXH={maxh}\n'
s = open(tpl).read().replace('__ENV__', env).replace('__MD5__', md5)
assert len(s) < 16000, len(s)
print(json.dumps(dict(client_id='me', image='pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel', env={}, disk=100, onstart=s,
                      runtype='ssh_direct', label=f'gw-{inst}')))
PY
R=$(curl -s -X PUT --max-time 30 -H 'Content-Type: application/json' "https://console.vast.ai/api/v0/asks/$OFFER/" -d @"$TMP/body.json")
ID=$(printf '%s' "$R" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('new_contract') or '')")
[ -n "$ID" ] || { echo "create failed: $(printf '%s' "$R" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('msg') or d.get('error'))")" >&2; exit 1; }
echo "$ID"
