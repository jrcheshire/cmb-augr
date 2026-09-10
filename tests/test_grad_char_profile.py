"""Gates for the ``--mode profile`` kernel-gap parser in
``scripts/mapbased_grad_characterization.py``.

The parser's job is to split device time into "inside a kernel" and "between
kernels", which is the measurement that separates a launch-latency-bound forward
from an arithmetic-bound one. It can only be exercised for real on a GPU, so its
arithmetic and its stream selection are pinned here against a SYNTHETIC trace
with known durations and known gaps. Without this, a naming change in the
profiler and a genuinely gap-free run both come back as "no device stream" and
a GPU job's worth of time is spent finding out which.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mapbased_grad_characterization.py"


@pytest.fixture(scope="module")
def gc_mod():
    """Load the script as a module (it is not an importable package member)."""
    spec = importlib.util.spec_from_file_location("_gc_profile", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_trace(tmp_path: Path, events) -> Path:
    """Write ``events`` where ``jax.profiler.trace`` would have put them."""
    d = tmp_path / "plugins" / "profile" / "2026_01_01_00_00_00"
    d.mkdir(parents=True)
    path = d / "host.trace.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump({"traceEvents": events}, fh)
    return path


def _meta(pid, name, tid=None, tname=None):
    out = [{"ph": "M", "pid": pid, "name": "process_name", "args": {"name": name}}]
    if tid is not None:
        out.append(
            {"ph": "M", "pid": pid, "tid": tid, "name": "thread_name", "args": {"name": tname}}
        )
    return out


# Three kernels: 10 us, then a 5 us gap, 5 us, then a 10 us gap, 20 us.
# kernel = 35 us, gap = 15 us, span = 50 us, gap fraction = 0.30.
_KERNELS = [
    {"ph": "X", "pid": 1, "tid": 7, "name": "fusion.1", "ts": 0.0, "dur": 10.0},
    {"ph": "X", "pid": 1, "tid": 7, "name": "fusion.2", "ts": 15.0, "dur": 5.0},
    {"ph": "X", "pid": 1, "tid": 7, "name": "fusion.1", "ts": 30.0, "dur": 20.0},
]


def test_gap_histogram_recovers_known_kernels_and_gaps(gc_mod, tmp_path):
    """Durations, gaps, span and the gap fraction come back exactly."""
    events = [
        *_meta(1, "/device:GPU:0", 7, "XLA Ops"),
        *_meta(2, "/host:CPU", 3, "tf_XLAEigen/3"),
        *_KERNELS,
        # Decoys: a module-level span on the same device (double-counts the whole
        # run if admitted) and a host-side op (not a device kernel at all).
        {"ph": "X", "pid": 1, "tid": 8, "name": "jit_loss", "ts": 0.0, "dur": 50.0},
        {"ph": "X", "pid": 2, "tid": 3, "name": "eigen_matmul", "ts": 0.0, "dur": 40.0},
    ]
    _write_trace(tmp_path, events)

    h = gc_mod._kernel_gap_histogram(str(tmp_path))

    assert h is not None
    assert h["stream"] == "XLA Ops"
    assert h["n_kernels"] == 3
    assert h["kernel_us"] == pytest.approx(35.0)
    assert h["gap_us"] == pytest.approx(15.0)
    assert h["span_us"] == pytest.approx(50.0)
    assert h["gap_fraction"] == pytest.approx(0.30)
    assert h["gap_median_us"] == pytest.approx(7.5)
    assert h["kernel_median_us"] == pytest.approx(10.0)
    # Per-op totals aggregate repeats of the same kernel name.
    assert dict(h["top_ops"])["fusion.1"] == pytest.approx(30.0)
    assert np.allclose(np.sort(h["gaps"]), [5.0, 10.0])


def test_gap_histogram_needs_a_device_stream(gc_mod, tmp_path):
    """A host-only trace returns None -- the CPU case, and not an error."""
    events = [
        *_meta(2, "/host:CPU", 3, "tf_XLAEigen/3"),
        {"ph": "X", "pid": 2, "tid": 3, "name": "eigen_matmul", "ts": 0.0, "dur": 40.0},
    ]
    _write_trace(tmp_path, events)
    assert gc_mod._kernel_gap_histogram(str(tmp_path)) is None


def test_gap_histogram_falls_back_when_the_thread_is_renamed(gc_mod, tmp_path):
    """A renamed kernel thread still measures, and says which stream it used.

    The profiler's "XLA Ops" name is not contractual. Silently returning None on a
    rename would read as "no gaps", so the fallback must fire AND be visible.
    """
    events = [
        *_meta(1, "/device:GPU:0", 7, "XLA Kernels"),
        *_meta(1, "/device:GPU:0", 8, "XLA Modules"),
        *_KERNELS,
        {"ph": "X", "pid": 1, "tid": 8, "name": "jit_loss", "ts": 0.0, "dur": 50.0},
    ]
    _write_trace(tmp_path, events)

    h = gc_mod._kernel_gap_histogram(str(tmp_path))

    assert h is not None
    assert h["stream"] == "fallback: XLA Kernels"
    assert h["n_kernels"] == 3
    assert h["gap_fraction"] == pytest.approx(0.30)


def test_gap_histogram_is_absent_without_a_trace(gc_mod, tmp_path):
    """No trace file at all is None, not a crash."""
    assert gc_mod._kernel_gap_histogram(str(tmp_path)) is None


def test_script_imports_without_optax():
    """The script must import in the slim aarch64 ``gpu`` env, which has no optax.

    ``design_opt.stochastic_design_descent`` already imports optax lazily for this
    reason; a module-level import in the driver undid it, and made every mode --
    including the GPU-only ones -- unimportable on the cluster. That failure costs
    a queue wait to discover, so it is pinned here instead.
    """

    class _NoOptax:
        def find_spec(self, name, path=None, target=None):
            if name == "optax" or name.startswith("optax."):
                raise ImportError("No module named 'optax' (simulated slim gpu env)")
            return None

    blocker = _NoOptax()
    saved = sys.modules.pop("optax", None)
    sys.meta_path.insert(0, blocker)
    try:
        spec = importlib.util.spec_from_file_location("_gc_no_optax", _SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(mod.run_profile)
        # Anti-vacuity: the Adam path still needs optax, so it must raise rather
        # than silently degrade -- otherwise this test would pass on a script that
        # had quietly dropped the optimizer.
        with pytest.raises(ImportError, match="optax"):
            mod._descent_adam(None, None, None, None, None, None, None)
    finally:
        sys.meta_path.remove(blocker)
        if saved is not None:
            sys.modules["optax"] = saved
