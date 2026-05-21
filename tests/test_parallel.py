"""Tests for augr.parallel -- process-pool helpers."""

from __future__ import annotations

import os

import pytest

from augr.parallel import (
    cpu_count,
    kill_orphan_workers,
    parallel_map,
    pin_blas_env,
    process_pool,
    resolve_delens_workers,
    workers_for_outer,
)

# Module-level worker functions: must be picklable for spawn-mode workers.


def _square(x: int) -> int:
    return x * x


def _read_delens_workers(_x: int) -> str:
    return os.environ.get("AUGR_DELENS_WORKERS", "<unset>")


def _read_blas_env(_x: int) -> tuple[str | None, str | None, str | None]:
    return (
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
    )


# -----------------------------------------------------------------------
# Worker-count arithmetic
# -----------------------------------------------------------------------

class TestWorkerArithmetic:
    def test_workers_for_outer_1_returns_cpu_count(self):
        assert workers_for_outer(1) == cpu_count()

    def test_workers_for_outer_cpu_count_returns_1(self):
        assert workers_for_outer(cpu_count()) == 1

    def test_workers_for_outer_oversize_clamps_to_1(self):
        # n_outer larger than cpu_count must still return >= 1 (clamp).
        assert workers_for_outer(cpu_count() * 4) == 1

    def test_workers_for_outer_zero_returns_cpu_count(self):
        # Degenerate input: treat 0 as "no outer" -> full machine.
        assert workers_for_outer(0) == cpu_count()

    def test_workers_for_outer_negative_returns_cpu_count(self):
        assert workers_for_outer(-1) == cpu_count()

    def test_resolve_delens_workers_matches(self):
        for n in (1, 2, 4, cpu_count(), cpu_count() * 2):
            assert resolve_delens_workers(n) == workers_for_outer(n)


# -----------------------------------------------------------------------
# BLAS env pinning
# -----------------------------------------------------------------------

class TestPinBlasEnv:
    def test_sets_all_three(self, monkeypatch):
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            monkeypatch.delenv(var, raising=False)
        pin_blas_env()
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"

    def test_idempotent(self, monkeypatch):
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            monkeypatch.delenv(var, raising=False)
        pin_blas_env()
        pin_blas_env()
        assert os.environ["OMP_NUM_THREADS"] == "1"

    def test_preserves_user_set_value(self, monkeypatch):
        # setdefault semantics: a value the user already chose must survive.
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
        pin_blas_env()
        assert os.environ["OMP_NUM_THREADS"] == "4"
        assert os.environ["MKL_NUM_THREADS"] == "1"

    def test_returns_prior_values(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
        prior = pin_blas_env()
        assert prior["OMP_NUM_THREADS"] == "2"
        assert prior["MKL_NUM_THREADS"] is None
        assert prior["OPENBLAS_NUM_THREADS"] is None


# -----------------------------------------------------------------------
# process_pool context manager
# -----------------------------------------------------------------------

class TestProcessPool:
    def test_yields_none_for_n_workers_le_1(self):
        with process_pool(1) as pool:
            assert pool is None
        with process_pool(0) as pool:
            assert pool is None

    @pytest.mark.slow
    def test_yields_pool_for_n_workers_ge_2(self):
        with process_pool(2) as pool:
            assert pool is not None
            results = pool.map(_square, [1, 2, 3, 4])
        assert results == [1, 4, 9, 16]

    @pytest.mark.slow
    def test_sets_augr_delens_workers_in_child(self, monkeypatch):
        # Child should see AUGR_DELENS_WORKERS = workers_for_outer(2).
        monkeypatch.delenv("AUGR_DELENS_WORKERS", raising=False)
        expected = str(workers_for_outer(2))
        with process_pool(2) as pool:
            child_vals = pool.map(_read_delens_workers, list(range(2)))
        assert all(v == expected for v in child_vals), child_vals

    @pytest.mark.slow
    def test_respects_explicit_delens_workers(self, monkeypatch):
        monkeypatch.delenv("AUGR_DELENS_WORKERS", raising=False)
        with process_pool(2, delens_workers=7) as pool:
            child_vals = pool.map(_read_delens_workers, [0, 0])
        assert child_vals == ["7", "7"]

    @pytest.mark.slow
    def test_respects_parent_set_delens_workers(self, monkeypatch):
        # Parent already set the env; default delens_workers=None must not clobber.
        monkeypatch.setenv("AUGR_DELENS_WORKERS", "3")
        with process_pool(2) as pool:
            child_vals = pool.map(_read_delens_workers, [0, 0])
        assert child_vals == ["3", "3"]

    @pytest.mark.slow
    def test_restores_delens_env_on_exit(self, monkeypatch):
        monkeypatch.delenv("AUGR_DELENS_WORKERS", raising=False)
        with process_pool(2) as pool:
            assert "AUGR_DELENS_WORKERS" in os.environ
            pool.map(_square, [1])
        # After exit, env must be back to "unset".
        assert "AUGR_DELENS_WORKERS" not in os.environ

    @pytest.mark.slow
    def test_restores_delens_env_on_exit_when_parent_set(self, monkeypatch):
        monkeypatch.setenv("AUGR_DELENS_WORKERS", "9")
        with process_pool(2, delens_workers=2) as pool:
            assert os.environ["AUGR_DELENS_WORKERS"] == "2"
            pool.map(_square, [1])
        assert os.environ["AUGR_DELENS_WORKERS"] == "9"

    @pytest.mark.slow
    def test_pin_blas_in_child(self, monkeypatch):
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            monkeypatch.delenv(var, raising=False)
        with process_pool(2, pin_blas=True) as pool:
            child_vals = pool.map(_read_blas_env, [0])
        assert child_vals[0] == ("1", "1", "1")

    @pytest.mark.slow
    def test_terminate_on_exception(self):
        class BoomError(RuntimeError):
            pass
        with pytest.raises(BoomError), process_pool(2):
            raise BoomError("kaboom")
        # If terminate() didn't run, this test would hang or zombie -- pytest
        # will simply hang the suite. Reaching here means clean shutdown.


# -----------------------------------------------------------------------
# parallel_map convenience
# -----------------------------------------------------------------------

class TestParallelMap:
    def test_serial_fallback(self):
        result = parallel_map(_square, [1, 2, 3, 4], workers=1)
        assert result == [1, 4, 9, 16]

    @pytest.mark.slow
    def test_parallel_path(self):
        result = parallel_map(_square, [1, 2, 3, 4, 5], workers=2)
        assert result == [1, 4, 9, 16, 25]

    def test_empty_input(self):
        assert parallel_map(_square, [], workers=1) == []
        assert parallel_map(_square, [], workers=2) == []


# -----------------------------------------------------------------------
# kill_orphan_workers smoke
# -----------------------------------------------------------------------

class TestKillOrphanWorkers:
    def test_runs_without_raising(self):
        # Best-effort smoke: pgrep against a marker pattern that no real
        # process matches, plus the unconditional "multiprocessing.resource_tracker"
        # pass. We don't assert any specific count -- the runner / xdist may
        # have legitimate resource_tracker processes from other tests -- only
        # that the call returns an int cleanly.
        n = kill_orphan_workers(name_filter="augr_no_such_process_marker_xyzzy")
        assert isinstance(n, int)
        assert n >= 0
