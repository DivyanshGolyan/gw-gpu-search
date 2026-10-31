"""seg_pipe.SegmentPipe keeps the serial order of fp_filter's per-segment work (CPU only; run: python test_seg_pipe.py).

The serial (v8) order per tile is, for each dataset m: for each segment s: front(s), collect(s), back(s); then
finish(m). The pipeline may move front(s+1) before back(s), and nothing else: backs and finishes keep their order,
each finish runs after its dataset's last back and before the next dataset's first back, and every collect follows
its own front."""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pipeline'))
import torch
from seg_pipe import SegmentPipe, BackStream, EventClock


def run(plan):
    """plan: list of (dataset, [segments]). Returns the event log of one tile through SegmentPipe."""
    log = []
    bs = BackStream(torch)
    pipe = SegmentPipe(lambda c: log.append(('back', c)), bs)
    for m, segs in plan:
        log.append(('cwait', m))
        for s in segs:
            pipe.step(lambda m=m, s=s: log.append(('front', (m, s))) or (m, s),
                      lambda f: log.append(('collect', f)) or f)
        pipe.then(lambda m=m: log.append(('finish', m)))
    pipe.drain()
    return log


def check(plan, log):
    serial = [(k, x) for k, x in log if k in ('back', 'finish')]
    want = []
    for m, segs in plan:
        want += [('back', (m, s)) for s in segs] + [('finish', m)]
    assert serial == want, (serial, want)
    pos = {e: i for i, e in enumerate(log)}
    for m, segs in plan:
        for s in segs:
            assert pos[('front', (m, s))] < pos[('collect', (m, s))] < pos[('back', (m, s))]
        assert pos[('cwait', m)] < min([pos[('front', (m, s))] for s in segs] + [pos[('finish', m)]])
    fronts = [x for k, x in log if k == 'front']
    assert fronts == [(m, s) for m, segs in plan for s in segs]
    # depth 1: front(k+1) comes before back(k), and back(k) before collect(k+1)
    for a, b in zip(fronts, fronts[1:]):
        assert pos[('front', b)] < pos[('back', a)] < pos[('collect', b)]


rng = random.Random(3)
for trial in range(500):
    plan = [(f'm{i}', list(range(rng.randint(0, 5)))) for i in range(rng.randint(1, 5))]
    check(plan, run(plan))

# an exception in a back propagates and leaves nothing pending
bs = BackStream(torch)
def bad(c):
    raise ValueError(c)
pipe = SegmentPipe(bad, bs)
pipe.step(lambda: 1, lambda f: f)
try:
    pipe.step(lambda: 2, lambda f: f)
    raise AssertionError('no exception')
except ValueError as e:
    assert e.args == (1,)
assert pipe.pending is None

# EventClock is a no-op without CUDA
ck = EventClock(torch); T = {}
with ck.span('x'):
    pass
ck.flush(T)
assert T == {} or not torch.cuda.is_available()
print('seg_pipe: 500 random tiles in serial order; exception path ok')
