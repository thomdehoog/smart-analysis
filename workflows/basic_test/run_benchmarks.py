"""
Performance benchmarks for the v4 pipeline engine.

Measures real wall-clock performance across key scenarios:
  1. Concurrency scaling      1, 2, 4, 8 concurrent jobs
  2. Warm vs cold workers     first call vs subsequent calls
  3. Scope collection         overhead of scope completion
  4. max_workers scaling      parallelism with max_workers
  5. Data serialization       cost of large data through workers
  6. Multi-pipeline           two pipelines sharing one engine

Usage:
    python run_benchmarks.py
"""

import sys
import time
import textwrap
import tempfile
import shutil
import atexit
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

WIDTH = 70
_TEMP = tempfile.mkdtemp(prefix="bench_")
atexit.register(shutil.rmtree, _TEMP, True)
_counter = 0


def _next_id():
    global _counter
    _counter += 1
    return _counter


def _temp_step(code, name=None):
    path = Path(_TEMP) / (f"{name}.py" if name else f"step_{_next_id()}.py")
    path.write_text(textwrap.dedent(code))
    return str(path)


def _temp_yaml(content):
    text = textwrap.dedent(content)
    if "functions_dir" not in text:
        d = Path(_TEMP).as_posix()
        header = f'metadata:\n  functions_dir: "{d}"\n'
        if "metadata:" in text:
            text = text.replace("metadata:", header.rstrip("\n"), 1)
        else:
            text = header + text
    path = Path(_TEMP) / f"pipeline_{_next_id()}.yaml"
    path.write_text(text)
    return str(path)


def _fmt(seconds):
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def _bar(value, max_value, width=30):
    filled = int(width * value / max_value) if max_value > 0 else 0
    return "#" * filled + "." * (width - filled)


def _wait_results(engine, name, expected, timeout=60):
    t0 = time.monotonic()
    collected = []
    while time.monotonic() - t0 < timeout:
        collected.extend(engine.results(name))
        if len(collected) >= expected:
            return collected
        time.sleep(0.1)
    return collected


# ---- Benchmarks ------------------------------------------------------


def bench_concurrency_scaling():
    """Measure throughput scaling with concurrent jobs."""
    _temp_step("""
        import time
        def run(pd, state, **p):
            time.sleep(0.1)
            pd["done"] = True
            return pd
    """, name="bs_work")

    from engine import Engine

    n_jobs = 16
    levels = [1, 2, 4, 8, 16]
    results = {}

    for n_threads in levels:
        yaml = _temp_yaml("wf:\n  - bs_work:")
        with Engine(max_concurrent=n_threads) as e:
            e.register("test", yaml)
            t0 = time.perf_counter()
            for i in range(n_jobs):
                e.submit("test", {})
            _wait_results(e, "test", n_jobs, timeout=60)
            elapsed = time.perf_counter() - t0
            results[n_threads] = elapsed

    baseline = results[1]
    for n_threads in levels:
        elapsed = results[n_threads]
        speedup = baseline / elapsed
        bar = _bar(speedup, max(baseline / v for v in results.values()))
        print(f"  {n_threads:2d} threads: {_fmt(elapsed):>8s}  "
              f"{speedup:5.1f}x  {bar}")


def bench_warm_worker():
    """Measure cold start vs warm calls on persistent workers."""
    from engine._worker import Worker

    path = _temp_step("def run(pd, state, **p): pd['ok'] = True; return pd")

    w = Worker(environment=None, connect_timeout=10)
    try:
        # Cold start
        t0 = time.perf_counter()
        w.execute(path, {}, {}, timeout=10)
        cold = time.perf_counter() - t0

        # Warm calls
        n = 20
        t0 = time.perf_counter()
        for _ in range(n):
            w.execute(path, {}, {}, timeout=10)
        warm = (time.perf_counter() - t0) / n

        print(f"  Cold start:  {_fmt(cold)}")
        print(f"  Warm call:   {_fmt(warm)} (avg of {n})")
        if warm > 0:
            print(f"  Speedup:     {cold / warm:.1f}x")
    finally:
        w.shutdown()


def bench_scope_collection():
    """Measure scope completion latency."""
    _temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="bsc_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["n"] = len(pd["results"])
            return pd
    """, name="bsc_b")
    yaml = _temp_yaml("""
        wf:
          - bsc_a:
          - bsc_b:
              scope: group
    """)

    from engine import Engine

    for n_jobs in [5, 20, 50]:
        with Engine(max_concurrent=32) as e:
            e.register("test", yaml)

            t0 = time.perf_counter()
            for i in range(n_jobs):
                is_last = (i == n_jobs - 1)
                e.submit("test", {"v": i},
                         scope={"group": "X"},
                         complete="group" if is_last else None)

            # Wait for scoped result
            results = _wait_results(e, "test", n_jobs + 1, timeout=60)
            elapsed = time.perf_counter() - t0

            scoped = [r for r in results if r.get("_phase") == 1]
            n = scoped[0]["n"] if scoped else 0
            print(f"  {n_jobs:3d} jobs: {_fmt(elapsed):>8s} total "
                  f"(scoped step got {n} results)")


def bench_max_workers_scaling():
    """Measure parallelism benefit of max_workers."""
    from engine import Engine

    for mw in [1, 2, 4, 8]:
        _temp_step(f"""
            import time
            METADATA = {{"max_workers": {mw}}}
            def run(pd, state, **p):
                time.sleep(0.1)
                pd["done"] = True
                return pd
        """, name=f"bmw_{mw}")
        yaml = _temp_yaml(f"wf:\n  - bmw_{mw}:")

        with Engine(max_concurrent=16) as e:
            e.register("test", yaml)
            t0 = time.perf_counter()
            for i in range(16):
                e.submit("test", {})
            _wait_results(e, "test", 16, timeout=60)
            elapsed = time.perf_counter() - t0

        print(f"  max_workers={mw}: {_fmt(elapsed):>8s} "
              f"for 16 jobs (each 0.1s)")


def bench_data_serialization():
    """Measure cost of moving large data through subprocess workers."""
    from engine._worker import Worker

    path = _temp_step("def run(pd, state, **p): return pd")

    sizes = [100, 1_000, 10_000, 50_000]
    w = Worker(environment=None, connect_timeout=10)
    try:
        # Warm up
        w.execute(path, {}, {}, timeout=10)

        for n in sizes:
            data = {"big": list(range(n))}
            t0 = time.perf_counter()
            r = w.execute(path, data, {}, timeout=30)
            elapsed = time.perf_counter() - t0
            assert len(r["big"]) == n
            print(f"  {n:>6,d} items: {_fmt(elapsed):>8s}  "
                  f"(round-trip through subprocess)")
    finally:
        w.shutdown()


def bench_multi_pipeline():
    """Measure throughput of two pipelines sharing one engine."""
    _temp_step("""
        import time
        def run(pd, state, **p):
            time.sleep(0.05)
            pd["pipeline"] = pd["input"]["pipeline"]
            return pd
    """, name="bmp")

    from engine import Engine

    n_per = 10

    # Single pipeline baseline
    yaml_a = _temp_yaml("wf_a:\n  - bmp:")
    with Engine(max_concurrent=16) as e:
        e.register("a", yaml_a)
        t0 = time.perf_counter()
        for i in range(n_per):
            e.submit("a", {"pipeline": "A"})
        _wait_results(e, "a", n_per, timeout=60)
        single = time.perf_counter() - t0

    # Two pipelines interleaved
    yaml_a2 = _temp_yaml("wf_a2:\n  - bmp:")
    yaml_b = _temp_yaml("wf_b:\n  - bmp:")
    with Engine(max_concurrent=16) as e:
        e.register("a", yaml_a2)
        e.register("b", yaml_b)
        t0 = time.perf_counter()
        for i in range(n_per):
            e.submit("a", {"pipeline": "A"})
            e.submit("b", {"pipeline": "B"})
        _wait_results(e, "a", n_per, timeout=60)
        _wait_results(e, "b", n_per, timeout=60)
        dual = time.perf_counter() - t0

    print(f"  Single pipeline ({n_per} jobs):  {_fmt(single)}")
    print(f"  Two pipelines ({n_per * 2} jobs): {_fmt(dual)}")
    print(f"  Overhead: {((dual / single) - 1) * 100:.0f}% "
          f"(for 2x the work)")


# ---- Main ------------------------------------------------------------


BENCHMARKS = [
    ("Concurrency scaling", bench_concurrency_scaling),
    ("Warm worker advantage", bench_warm_worker),
    ("Scope collection latency", bench_scope_collection),
    ("max_workers scaling", bench_max_workers_scaling),
    ("Data serialization cost", bench_data_serialization),
    ("Multi-pipeline interleaving", bench_multi_pipeline),
]


def main():
    import engine
    print()
    print("=" * WIDTH)
    print("  SMART Analysis v4 -- Performance Benchmarks")
    print("=" * WIDTH)
    print()
    print(f"  Engine:   {engine.__version__}")
    print(f"  Python:   {sys.version.split()[0]}")
    print()

    t_total = time.perf_counter()

    for name, func in BENCHMARKS:
        print("-" * WIDTH)
        print(f"  {name}")
        print("-" * WIDTH)
        try:
            func()
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    elapsed = time.perf_counter() - t_total
    print("=" * WIDTH)
    print(f"  Total benchmark time: {_fmt(elapsed)}")
    print("=" * WIDTH)


if __name__ == "__main__":
    main()
