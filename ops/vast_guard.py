"""Hard money stop for Vast.ai + RunPod (2026-10-28, after O4a's bill ran $0.66 past its cap).

Every 2 min: Vast spend = BASE_CREDIT - current credit, RunPod spend = RP_BASE - clientBalance. Appends to
budget/pod_alerts.log (the file every session Monitors):
  ALERT vast-90%    Vast spend + 15 min of the running burn >= 90% of VAST_CAP
  ALERT vast-CAP    ... >= VAST_CAP: every Vast instance labelled gw-* is DESTROYED (its outputs should already be on the relay)
  ALERT runpod-90% / runpod-CAP   same for RunPod; at the cap every pod named gw-* is deleted
  ALERT runway      RunPod balance / spend per hour < 2 h
and writes budget/guard_state.json (spend, burn, caps) which vast_launch.sh reads to refuse launches that would cross a cap.
Usage: python3 bench/vast_guard.py [--once]     (curl only: python urllib POST to RunPod is blocked here)"""
import json, os, subprocess, sys, time
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'budget')
VAST_BASE, VAST_CAP = 5.00, 4.80       # credit at 17:38 UTC 2026-10-28; 4.00 -> 4.50 at 22:25 UTC; -> 4.80 at 00:52 UTC 2026-10-29 for the 4-week mop-up (Divyansh allowed the whole budget); keeps $0.20
RP_BASE, RP_CAP = 1.9885, 1.50         # RunPod clientBalance at 18:30 UTC 2026-10-28; O4b RunPod cap (planning thread)
V = 'https://console.vast.ai/api'


def curl(*a):
    r = subprocess.run(['curl', '-s', '--max-time', '30', *a], capture_output=True, text=True)
    return json.loads(r.stdout) if r.stdout.strip().startswith(('{', '[')) else {}


def alert(kind, msg):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} ALERT {kind} {msg}'
    open(os.path.join(B, 'pod_alerts.log'), 'a').write(line + '\n'); print(line, flush=True)


def check(sent):
    rk = os.environ.get('RUNPOD_KEY', '')
    u = curl(f'{V}/v0/users/current/'); inst = curl(f'{V}/v1/instances/').get('instances', [])
    gw = [i for i in inst if (i.get('label') or '').startswith('gw-')]
    vburn = sum(i.get('dph_total', 0) for i in inst if i.get('intended_status') != 'stopped')   # a just-created box has no actual_status yet
    vspend = round(VAST_BASE - u['credit'], 4) if 'credit' in u else None
    rp = curl('-H', f'Authorization: Bearer {rk}', '-H', 'Content-Type: application/json', 'https://api.runpod.io/graphql',
              '-d', '{"query":"query { myself { clientBalance currentSpendPerHr pods { id name } } }"}').get('data', {}).get('myself', {})
    rbal, rburn = rp.get('clientBalance'), rp.get('currentSpendPerHr') or 0
    rspend = round(RP_BASE - rbal, 4) if rbal is not None else None
    st = dict(as_of=time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()), vast_spend=vspend, vast_burn_per_h=round(vburn, 3), vast_cap=VAST_CAP,
              vast_instances=[(i['id'], i.get('label'), i.get('actual_status'), round(i.get('dph_total', 0), 3)) for i in inst],
              runpod_spend=rspend, runpod_balance=rbal, runpod_burn_per_h=rburn, runpod_cap=RP_CAP)
    json.dump(st, open(os.path.join(B, 'guard_state.json'), 'w'), indent=1)
    if vspend is not None:
        proj = vspend + vburn / 4
        if proj >= VAST_CAP:
            for i in gw:
                curl('-X', 'DELETE', f'{V}/v0/instances/{i["id"]}/')
            alert('vast-CAP', f'spend ${vspend:.2f} + 15 min burn >= cap ${VAST_CAP}: destroyed {[i["id"] for i in gw]}')
        elif proj >= 0.9 * VAST_CAP and 'v90' not in sent:
            sent.add('v90'); alert('vast-90%', f'spend ${vspend:.2f}, burn ${vburn:.2f}/h, cap ${VAST_CAP}')
    if rspend is not None:
        proj = rspend + rburn / 4
        if proj >= RP_CAP:
            for p in rp.get('pods') or []:
                if (p.get('name') or '').startswith('gw-'):
                    curl('-X', 'DELETE', '-H', f'Authorization: Bearer {rk}', f'https://rest.runpod.io/v1/pods/{p["id"]}')
            alert('runpod-CAP', f'spend ${rspend:.2f} + 15 min burn >= cap ${RP_CAP}: deleted gw-* pods')
        elif proj >= 0.9 * RP_CAP and 'r90' not in sent:
            sent.add('r90'); alert('runpod-90%', f'spend ${rspend:.2f}, burn ${rburn:.2f}/h, cap ${RP_CAP}')
        if rburn > 0 and rbal / rburn < 2 and 'rw' not in sent:
            sent.add('rw'); alert('runway', f'RunPod balance ${rbal:.2f} lasts {rbal / rburn:.1f} h at ${rburn:.2f}/h')
    return st


if __name__ == '__main__':
    sent = set()
    while True:
        try:
            st = check(sent)
            if '--once' in sys.argv:
                print(json.dumps(st, indent=1)); break
        except Exception as e:
            alert('guard-error', repr(e)[:200])
        time.sleep(120)
