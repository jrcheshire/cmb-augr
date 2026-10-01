"""Does jax.profiler.trace destabilize kernel launch on this node?

Three arms of increasing complexity, each run N times OUTSIDE the profiler and
then N times INSIDE it. The arm that first fails inside-but-not-outside says how
much graph is needed to provoke it -- which is the difference between "the
profiler is broken here" (arm 1 fails) and "our workload provokes it" (only arm 3).

  pixi run -e gpu python profiler_smoke.py --repeat 5
"""
from __future__ import annotations

import argparse
import tempfile

import jax
import jax.numpy as jnp


def arm_matmul(n=4096):
    a = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=jnp.float64)
    f = jax.jit(lambda x: (x @ x).sum())
    return f, (a,)

def arm_scan(n=256, steps=512):
    x = jax.random.normal(jax.random.PRNGKey(1), (n, n), dtype=jnp.float64)
    def body(c, _):
        return jnp.tanh(c @ c.T) * 0.5, None
    f = jax.jit(lambda z: jax.lax.scan(body, z, None, length=steps)[0].sum())
    return f, (x,)

def arm_scan_grad(n=192, steps=256):
    x = jax.random.normal(jax.random.PRNGKey(2), (n, n), dtype=jnp.float64)
    def body(c, _):
        return jnp.tanh(c @ c.T) * 0.5, None
    def loss(z):
        return jax.lax.scan(body, z, None, length=steps)[0].sum()
    f = jax.jit(jax.grad(loss))
    return f, (x,)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=5)
    a = p.parse_args()
    print(f"jax {jax.__version__}  backend {jax.default_backend()}", flush=True)
    for d in jax.devices():
        print(f"  device {d} kind={getattr(d, 'device_kind', '?')}", flush=True)
    for name, build in (("matmul", arm_matmul), ("scan", arm_scan), ("scan+grad", arm_scan_grad)):
        f, args = build()
        jax.block_until_ready(f(*args))  # compile
        try:
            for _ in range(a.repeat):
                jax.block_until_ready(f(*args))
            print(f"  {name:10s} outside profiler: OK x{a.repeat}", flush=True)
        except Exception as e:
            print(f"  {name:10s} outside profiler: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        try:
            with tempfile.TemporaryDirectory() as d, jax.profiler.trace(d):
                for _ in range(a.repeat):
                    jax.block_until_ready(f(*args))
            print(f"  {name:10s} INSIDE  profiler: OK x{a.repeat}", flush=True)
        except Exception as e:
            print(f"  {name:10s} INSIDE  profiler: FAILED {type(e).__name__}: {e}", flush=True)

if __name__ == "__main__":
    main()
