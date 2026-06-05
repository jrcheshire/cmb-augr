"""augr.parallel -- process-pool helpers.

Consolidates the multiprocessing + BLAS-env + AUGR_DELENS_WORKERS
boilerplate that augr scripts (and external callers like
JPL_CMBprobe_2026's `sweep_dmirror.py` / `aperture_sweep.py`)
reimplement piecewise. Two entry points:

* :func:`process_pool` -- context manager yielding a spawn-context
  ``multiprocessing.Pool``, or ``None`` when ``n_workers <= 1`` so the
  caller can fall back to a serial loop without a parallel branch in
  outer code.
* :func:`parallel_map` -- convenience wrapping :func:`process_pool` for
  the "just map a callable over args" case; always returns a list.

Plus helpers callers reach for directly: :func:`cpu_count`,
:func:`workers_for_outer`, :func:`pin_blas_env`,
:func:`kill_orphan_workers`.

When this module is enough on its own:

* ``scripts/broom_residual_template.py`` MC sim loop.
* ``scripts/validate_pico.py`` per-case forecast sweeps.
* JPL-side ``aperture_sweep.py`` / ``sweep_dmirror.py``.

When this module is **not** the right tool:

* The per-L Wigner-3j parallelism in ``iterate_delensing`` and
  ``compute_n0_mv``. That one is a module-level lazy
  ``concurrent.futures.ProcessPoolExecutor`` rather than a
  context-managed Pool, because the outer routine calls into it from
  many call sites within a single forecast and the pool's startup cost
  is non-trivial. ``augr/delensing.py`` imports :func:`cpu_count` from
  here for its env-policy default but keeps its own pool object.
* Anything inside a ``jax.jit``-traced compute graph -- use
  ``jax.vmap`` / ``jax.pmap``. See :mod:`augr.sweep` for ready-made
  vmap wrappers over the differentiable forward path.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from multiprocessing.pool import Pool

_BLAS_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def cpu_count() -> int:
    """``os.cpu_count()`` with a 1-fallback."""
    return os.cpu_count() or 1


def workers_for_outer(n_outer: int) -> int:
    """Inner worker count for a nested-pool caller.

    Returns ``max(1, cpu_count() // n_outer)`` -- the recurring math
    when an outer ``multiprocessing.Pool`` wants each child to run
    with N inner workers without oversubscribing all-cores × N.
    ``n_outer <= 0`` returns the full ``cpu_count()``.
    """
    if n_outer <= 0:
        return cpu_count()
    return max(1, cpu_count() // n_outer)


def resolve_delens_workers(n_outer: int) -> int:
    """Recommended ``AUGR_DELENS_WORKERS`` value for a parent with pool size ``n_outer``.

    Identical math to :func:`workers_for_outer`; named for clarity at
    call sites that read like "compute the delens-workers env value".
    Used by :func:`process_pool` as the default when the caller does
    not pass ``delens_workers`` explicitly.
    """
    return workers_for_outer(n_outer)


def pin_blas_env() -> dict[str, str | None]:
    """Set BLAS thread counts to 1 in the parent process env.

    Sets ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` /
    ``OPENBLAS_NUM_THREADS`` via :func:`os.environ.setdefault`, so
    values the user already configured are preserved. Spawn-context
    children inherit the parent env at process creation, so this
    pinning propagates into pool workers.

    Returns the prior values (``None`` for vars that were unset
    before the call) for any caller that wants to restore them.
    Idempotent: a second call is a no-op.
    """
    prior: dict[str, str | None] = {}
    for var in _BLAS_ENV_VARS:
        prior[var] = os.environ.get(var)
        os.environ.setdefault(var, "1")
    return prior


@contextmanager
def process_pool(
    n_workers: int,
    *,
    pin_blas: bool = True,
    delens_workers: int | None = None,
    maxtasksperchild: int | None = None,
) -> Iterator[Pool | None]:
    """Spawn-context :class:`multiprocessing.pool.Pool` with BLAS + delens-workers policy.

    Yields a ``Pool`` when ``n_workers > 1``, or ``None`` when
    ``n_workers <= 1`` so callers can branch on a single ``with`` block
    rather than maintaining two parallel code paths.

    Parameters
    ----------
    n_workers
        Number of pool workers. Values ``<= 1`` disable the pool.
    pin_blas
        If True (default), calls :func:`pin_blas_env` before pool
        creation so spawn children inherit OMP/MKL/OPENBLAS=1. Set
        False if you know your worker is JAX-only and the pinning is
        unwanted; in practice pinning is harmless either way.
    delens_workers
        Sets ``AUGR_DELENS_WORKERS`` in the parent env before pool
        creation so workers inherit it. ``None`` (default) means:
        leave the parent's existing value alone if it is already set,
        else use ``resolve_delens_workers(n_workers)`` so
        outer × inner ≈ ``cpu_count()``. Explicit ``int`` always wins.
        Restored on exit to the prior state.
    maxtasksperchild
        Forwarded to :class:`multiprocessing.pool.Pool`. ``None``
        (default) keeps the standard behaviour: each worker lives for
        the whole pool and is reused for every task it pulls. A finite
        value recycles a worker (clean ``exit`` + fresh ``spawn``)
        after it has completed that many tasks, which **bounds RSS
        growth across tasks** -- the failure mode that OOM-kills a
        long-lived worker when per-task allocations (JAX/XLA
        retention, ducc0 buffers, large transient arrays) accumulate
        over hundreds of tasks. Set ``1`` for the tightest ceiling (a
        fresh process per task) at the cost of re-paying per-process
        import / warm-up; a small value (2-4) amortises that.
        Recycling happens on *clean* task completion only -- a worker
        SIGKILLed mid-task (e.g. by the OS OOM-killer) is not recycled
        and will hang ``map`` / ``imap``; see
        :func:`parallel_imap_unordered`'s ``result_timeout`` watchdog
        for the fail-fast on that path.

    Yields
    ------
    multiprocessing.pool.Pool or None
        Use as ``with process_pool(n) as pool:``. If ``pool`` is
        ``None``, fall back to a serial Python loop; else call
        ``pool.map(fn, args)``.

    Notes
    -----
    Worker functions must be picklable for spawn semantics:
    module-level (not closures), with module-level globals re-bound
    inside the worker if the worker reads them (see
    ``scripts/broom_residual_template.py:_run_one_sim``).

    On normal exit the pool is closed and joined; on exception inside
    the ``with`` it is terminated and joined before the exception
    propagates. Worker subprocesses outliving a hard parent kill
    (Ctrl-C that bypasses ``__exit__``, SIGKILL) need
    :func:`kill_orphan_workers`.
    """
    if n_workers <= 1:
        yield None
        return

    if pin_blas:
        pin_blas_env()

    prior_delens = os.environ.get("AUGR_DELENS_WORKERS")
    if delens_workers is None:
        if prior_delens is None:
            os.environ["AUGR_DELENS_WORKERS"] = str(resolve_delens_workers(n_workers))
    else:
        os.environ["AUGR_DELENS_WORKERS"] = str(int(delens_workers))

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(n_workers, maxtasksperchild=maxtasksperchild)
    try:
        yield pool
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()
    finally:
        if prior_delens is None:
            os.environ.pop("AUGR_DELENS_WORKERS", None)
        else:
            os.environ["AUGR_DELENS_WORKERS"] = prior_delens


def parallel_map(
    fn: Callable,
    args: Sequence,
    *,
    workers: int = 1,
    chunksize: int = 1,
    pin_blas: bool = True,
    delens_workers: int | None = None,
    maxtasksperchild: int | None = None,
) -> list:
    """``pool.map`` convenience: serial when ``workers <= 1``, parallel otherwise.

    Always returns a list in input order. Wraps :func:`process_pool`
    so the BLAS-pin / ``AUGR_DELENS_WORKERS`` / ``maxtasksperchild``
    policy is identical to the context-manager path.
    """
    with process_pool(
        workers,
        pin_blas=pin_blas,
        delens_workers=delens_workers,
        maxtasksperchild=maxtasksperchild,
    ) as pool:
        if pool is None:
            return [fn(a) for a in args]
        return list(pool.map(fn, args, chunksize=chunksize))


def parallel_imap_unordered(
    fn: Callable,
    args: Sequence,
    *,
    workers: int = 1,
    chunksize: int = 1,
    pin_blas: bool = True,
    delens_workers: int | None = None,
    maxtasksperchild: int | None = None,
    result_timeout: float | None = None,
) -> Iterator:
    """Streaming ``pool.imap_unordered``: yields results **as they complete**.

    Unlike :func:`parallel_map` (which blocks until every task is done and
    returns one list), this is a generator: each result is yielded the moment
    a worker finishes it. That lets the caller act on partial progress --
    checkpoint each completed task to disk, print a heartbeat, update an ETA --
    so a crash or kill loses at most the in-flight tasks rather than the whole
    run. Results arrive in **completion order, not input order**; the worker
    must therefore return enough identity in its payload for the caller to
    place it (e.g. return ``(key, value)``).

    Serial fallback (``workers <= 1``) yields in input order, one ``fn(a)`` at
    a time, so the streaming-consumer code path is identical with and without a
    pool.

    Parameters
    ----------
    fn, args, workers, chunksize, pin_blas, delens_workers, maxtasksperchild
        As for :func:`parallel_map` / :func:`process_pool`.
    result_timeout
        Watchdog, in seconds. ``None`` (default) waits indefinitely for the
        next result (standard ``imap_unordered`` behaviour). A finite value is
        the maximum time to wait for *any* one task to complete; if no result
        arrives within it, a :class:`multiprocessing.TimeoutError` is raised
        and the pool is terminated. This is the fail-fast for a deadlocked
        pool -- e.g. a worker SIGKILLed by the OS OOM-killer, which leaves
        ``imap_unordered`` waiting forever on a result that will never arrive
        while the surviving workers go idle. Size it well above a single
        task's wall time (it also bounds the gap between the *last* few
        results, when fewer tasks than workers remain) so a slow-but-live task
        does not trip it. Pairs with caller-side checkpointing: a tripped
        watchdog becomes a clean, resumable stop instead of a silent hang.
    """
    with process_pool(
        workers,
        pin_blas=pin_blas,
        delens_workers=delens_workers,
        maxtasksperchild=maxtasksperchild,
    ) as pool:
        if pool is None:
            for a in args:
                yield fn(a)
            return
        # IMapUnorderedIterator.next(timeout) raises multiprocessing.TimeoutError
        # if no result lands within `timeout`, and StopIteration when drained.
        # Letting the TimeoutError propagate out of this `with` triggers
        # process_pool's `except BaseException: pool.terminate()` path.
        it = pool.imap_unordered(fn, args, chunksize)
        while True:
            try:
                yield it.next(result_timeout)
            except StopIteration:
                return


def kill_orphan_workers(name_filter: str = "spawn_main") -> int:
    """SIGKILL processes whose command line matches ``name_filter``.

    Recipe for the case documented in
    ``feedback_pool_workers_orphan_on_kill``: after a parent holding a
    :class:`multiprocessing.Pool` (spawn) is hard-killed (external
    SIGKILL, Ctrl-C that bypasses ``__exit__``), spawn worker
    subprocesses can survive and keep ~2 GB of JAX memory each.

    This function is intentionally narrow: it only kills processes
    whose ``pgrep -f`` output matches the user-supplied pattern,
    minus the calling process itself. It deliberately does **not**
    auto-include ``multiprocessing.resource_tracker`` -- the
    resource tracker is a singleton owned by the active Python
    process and killing it from inside a live program breaks any
    further multiprocessing use. If you want to clean up dead
    parents' resource trackers separately, call
    ``kill_orphan_workers(name_filter="multiprocessing.resource_tracker")``
    explicitly after you've verified the trackers in question are
    orphans (parent PID is 1 / init).

    Parameters
    ----------
    name_filter
        ``pgrep -f`` pattern matching worker processes. Default
        ``"spawn_main"`` matches Python multiprocessing spawn-mode
        worker entry points.

    Returns
    -------
    int
        Number of processes killed.

    Notes
    -----
    POSIX only (uses ``pgrep`` + ``SIGKILL``). On platforms without
    ``pgrep`` this returns 0 silently.
    """
    try:
        out = subprocess.run(
            ["pgrep", "-f", name_filter],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 0
    if out.returncode != 0 or not out.stdout.strip():
        return 0
    n_killed = 0
    for pid_str in out.stdout.strip().split("\n"):
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            n_killed += 1
        except (ProcessLookupError, PermissionError):
            pass
    return n_killed
