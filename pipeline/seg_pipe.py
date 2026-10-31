"""Overlap of the host work of one data segment with the GPU work of the next (fp_filter.py v9).

fp_filter's per-segment loop has a GPU-heavy front (basis inverse FFTs, the GEMM + peak kernel, window maxima,
candidate selection) and a host-heavy back (numpy bookkeeping, the exact SNR / chi-squared stage, sg_chisq,
appends). Run one after the other, the GPU waits while the host runs the back. SegmentPipe runs them as a depth-1
pipeline:

    host:  launch front(s+1) | back(s) on the back stream | collect(s+1): selection, copies to the host
    GPU:   default stream: front(s+1) ...........           back stream: exact(s), sg(s)

Rules that keep this equal to the serial order:
- Fronts run on the default stream, backs on BackStream (torch and cupy bound to it together), in step order, so
  every append happens in the serial order.
- Only host (numpy) data passes from collect to back. Tensors used on both streams live for a whole tile; the back
  stream waits for the default stream once per tile (after_front) and the default stream waits for the back stream
  before they are freed (drain).
"""
import contextlib


class BackStream:
    """One non-blocking stream for the back, with torch's and cupy's current stream set to it together inside `with`.
    Inside it, cupy kernels, cupy's allocations through torch's allocator, and cp.from_dlpack of torch tensors all use
    this stream, so no hidden wait on the default stream is added. On a machine without CUDA (CPU tests) every method
    does nothing."""

    def __init__(self, torch, cp=None, priority=-1):
        self.torch, self.cp = torch, cp
        self.on = torch.cuda.is_available()
        if self.on:
            self.S = torch.cuda.Stream(priority=priority)
            self.cps = cp.cuda.ExternalStream(self.S.cuda_stream) if cp is not None else None
        self._stack = None

    def __enter__(self):
        st = contextlib.ExitStack()
        if self.on:
            st.enter_context(self.torch.cuda.stream(self.S))
            if self.cps is not None:
                st.enter_context(self.cps)
                assert self.cp.cuda.get_current_stream().ptr == self.torch.cuda.current_stream().cuda_stream
        self._stack = st
        return self

    def __exit__(self, *exc):
        st, self._stack = self._stack, None
        st.close()
        return False

    def after_front(self):
        """The back stream waits for everything enqueued so far on the default stream (the tile's tensors, the data)."""
        if self.on:
            self.S.wait_stream(self.torch.cuda.default_stream())

    def join_front(self):
        """The default stream waits for the back stream (before tensors used by both are freed)."""
        if self.on:
            self.torch.cuda.default_stream().wait_stream(self.S)

    def synchronize(self):
        """Host waits for the back stream only (never a device-wide sync)."""
        if self.on:
            self.S.synchronize()


class EventClock:
    """GPU time per T key from event pairs on the current stream; no host sync until flush()."""

    def __init__(self, torch):
        self.torch, self.spans = torch, []
        self.on = torch.cuda.is_available()

    @contextlib.contextmanager
    def span(self, key):
        if not self.on:
            yield
            return
        e0 = self.torch.cuda.Event(enable_timing=True); e0.record()
        yield
        e1 = self.torch.cuda.Event(enable_timing=True); e1.record()
        self.spans.append((key, e0, e1))

    def flush(self, T):
        """Add the spans to T (seconds). Call after the events are complete (after drain)."""
        for k, e0, e1 in self.spans:
            e1.synchronize()
            T[k] = T.get(k, 0.0) + e0.elapsed_time(e1) / 1e3
        self.spans = []


class SegmentPipe:
    """Depth-1 pipeline: front(s+1) runs on the GPU while back(s) runs. step(launch, collect): launch() enqueues the
    front and returns its handle without a host sync; collect(handle) does the selection (may sync the default stream)
    and returns what back() takes. Backs run in step order (FIFO); then(fn) runs fn after the back of the last step,
    or at once when nothing is pending."""

    def __init__(self, back, bstream):
        self.back, self.bs = back, bstream
        self.pending, self.after = None, []

    def step(self, launch, collect):
        fa = launch()
        self._run_back()
        self.pending = (collect(fa),)

    def then(self, fn):
        if self.pending is None:
            fn()
        else:
            self.after.append(fn)

    def drain(self):
        self._run_back()
        self.bs.join_front()

    def _run_back(self):
        if self.pending is not None:
            (c,), self.pending = self.pending, None
            with self.bs:
                self.back(c)
        after, self.after = self.after, []
        for fn in after:
            fn()
