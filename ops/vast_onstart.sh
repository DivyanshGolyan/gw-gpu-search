# Vast onstart (no inbound needed): fetch the payload from the relay HUB, write env.sh, run pay/boot.sh.
mkdir -p /work/out /work/in; cd /work
cat > /work/in/env.sh <<'ENV'
__ENV__
ENV
. /work/in/env.sh
for i in $(seq 120); do curl -sf --max-time 120 -o /work/in/pay.txz $HUB/f/pay/pay.txz && [ "$(md5sum < /work/in/pay.txz | cut -c1-32)" = __MD5__ ] && break; sleep 15; done
tar xJf /work/in/pay.txz -C /work && nohup bash /work/pay/boot.sh > /work/out/boot.log 2>&1 &
