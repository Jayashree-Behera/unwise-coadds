"""
Per-stage wall-time accounting for the main coadd pipeline.

destripe_utils has its own copy of this for the --destripe path, which is how
we learned that --destripe is compute-bound in mask_stars rather than blocked
on I/O. The plain (no --destripe) path had no equivalent, so there was no way
to tell which part of round 1 / round 2 a run actually spends its time in.

Nothing here runs unless $UNWISE_TIMING is set to a positive integer, which is
also the report interval in items. When it is unset, `stage()` returns a shared
do-nothing object and no clock is read, so instrumenting a hot loop costs one
attribute lookup per call.

Round 1 runs in worker processes, so each process accumulates and reports its
own totals; round 2 and the accumulation loops run in the parent. Reports are
tagged with the pid so the lines can be told apart and summed.

Usage:

    from pipeline_timing import stage, report, tick

    with stage('round1_read'):
        ...
    tick('round1')        # count one item, report every $UNWISE_TIMING items
"""
import os
import sys
import time

_EVERY = 0
try:
    _EVERY = int(os.environ.get('UNWISE_TIMING', '0') or 0)
except ValueError:
    _EVERY = 0

ENABLED = _EVERY > 0

_TIME = {}
_N = {}
_COUNT = {}


class _Stage(object):
    """Context manager accumulating wall time under `name`."""
    __slots__ = ('name', 't0')

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        _TIME[self.name] = _TIME.get(self.name, 0.0) + dt
        _N[self.name] = _N.get(self.name, 0) + 1
        return False


class _Null(object):
    """Stand-in used when timing is off; reads no clock."""
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL = _Null()


def stage(name):
    if not ENABLED:
        return _NULL
    return _Stage(name)


_OPEN = {}


def start(name):
    """Open an interval under `name`; close it with stop(name)."""
    if not ENABLED:
        return
    _OPEN[name] = time.time()


def stop(name):
    """Close the interval opened by start(name) and accumulate it."""
    if not ENABLED:
        return
    t0 = _OPEN.pop(name, None)
    if t0 is None:
        return
    _TIME[name] = _TIME.get(name, 0.0) + (time.time() - t0)
    _N[name] = _N.get(name, 0) + 1


def report(label='pipeline'):
    """One-line summary of this process's accumulated stage times."""
    if not _TIME:
        return '%s timing: nothing recorded' % label
    total = sum(_TIME.values())
    parts = ['%s %.1fs/%d' % (k, _TIME[k], _N[k])
             for k in sorted(_TIME, key=lambda k: -_TIME[k])]
    return ('%s timing pid=%d total=%.1fs | %s'
            % (label, os.getpid(), total, '  '.join(parts)))


def tick(label='pipeline'):
    """Count one processed item; print a report every $UNWISE_TIMING items."""
    if not ENABLED:
        return
    n = _COUNT.get(label, 0) + 1
    _COUNT[label] = n
    if (n % _EVERY) == 0:
        print('%s [%d items] %s' % (label, n, report(label)))
        sys.stdout.flush()


def report_now(label='pipeline'):
    """Print the accumulated totals immediately (end of a phase)."""
    if not ENABLED:
        return
    print(report(label))
    sys.stdout.flush()
