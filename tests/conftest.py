"""Shared pytest configuration -- the fast-gate network guard.

The per-PR gate (``pytest -m 'not slow'``) must never touch the network. A
cold-cache PySM / astropy template download blocks in a C-level socket wait that
``--timeout-method=thread`` cannot interrupt, so it bloats CI and dumps
faulthandler stacks instead of failing. Every test that legitimately downloads
(builds a PySM ``Sky`` / calls ``get_emission``) is marked ``slow`` and runs only
in the weekly suite.

This autouse fixture enforces that invariant structurally: for any non-``slow``
test it disables astropy internet access (PySM fetches its templates through
``astropy.utils.data.download_file``, which honors ``conf.allow_internet`` and
caches to ``~/.astropy/cache``). An accidental downloader in the fast gate then
raises immediately with a clear error instead of hanging forever. ``slow`` tests
keep internet enabled so the weekly suite still fetches templates. Already-cached
templates are served from disk regardless, so this never breaks a warm-cache run.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_network_in_fast_gate(request):
    """Disable astropy internet downloads for non-``slow`` tests (fail fast, not hang)."""
    if request.node.get_closest_marker("slow") is not None:
        yield
        return
    try:
        from astropy.utils.data import conf
    except ImportError:  # pragma: no cover - astropy always present via healpy
        yield
        return
    prev = conf.allow_internet
    conf.allow_internet = False
    try:
        yield
    finally:
        conf.allow_internet = prev
