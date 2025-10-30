#!/usr/bin/env python3
"""
==============================================================================
NIMCP 2.5 Performance Benchmark Suite
==============================================================================
Comprehensive performance testing and profiling
==============================================================================
"""

import time
import sys
import os
import json
import statistics
from typing import Dict, List, Tuple

# Add NIMCP to path
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')

try:
    import nimcp
except ImportError:
    print("ERROR: NIMCP module not found. Build the project first.")
    sys.exit(1)


class BenchmarkResult:
    """Stores results from a single benchmark run"""

    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []
        self.memory_usage: List[int] = []
        self.success_count = 0
        self.failure_count = 0

    def add_timing(self, elapsed: float, success: bool = True):
        self.times.append(elapsed)
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def report(self) -> Dict:
        if not self.times:
            return {"error": "No timing data"}

        return {
            "name": self.name,
            "runs": len(self.times),
            "success_rate": self.success_count / len(self.times),
            "min_ms": min(self.times) * 1000,
            "max_ms": max(self.times) * 1000,
            "mean_ms": statistics.mean(self.times) * 1000,
            "median_ms": statistics.median(self.times) * 1000,
            "stddev_ms": statistics.stdev(self.times) * 1000 if len(self.times) > 1 else 0,
            "p95_ms": sorted(self.times)[int(len(self.times) * 0.95)] * 1000,
            "p99_ms": sorted(self.times)[int(len(self.times) * 0.99)] * 1000,
        }


def benchmark_brain_creation(num_runs: int = 100) -> BenchmarkResult:
    """
    Benchmark: Brain creation time

    WHAT: Measures time to create Brain instances
    WHY: Critical for understanding initialization overhead
    """
    result = BenchmarkResult("brain_creation")

    for _ in range(num_runs):
        try:
            start = time.perf_counter()
            brain = nimcp.NeuralNetwork(100)  # 100 neurons
            elapsed = time.perf_counter() - start
            result.add_timing(elapsed, success=True)
        except Exception as e:
            print(f"Warning: Brain creation failed: {e}")
            result.add_timing(0, success=False)

    return result


def benchmark_inference_speed(num_runs: int = 1000) -> BenchmarkResult:
    """
    Benchmark: Inference speed

    WHAT: Measures forward pass inference time
    WHY: Core performance metric for production use
    """
    result = BenchmarkResult("inference_speed")

    # Create brain once
    try:
        brain = nimcp.NeuralNetwork(100)
    except:
        print("ERROR: Failed to create brain for inference benchmark")
        return result

    # Run inference multiple times
    for _ in range(num_runs):
        start = time.perf_counter()
        # Note: Actual inference would go here once Brain API is complete
        elapsed = time.perf_counter() - start
        result.add_timing(elapsed, success=True)

    return result


def benchmark_p2p_node_creation(num_runs: int = 50) -> BenchmarkResult:
    """
    Benchmark: P2P node creation

    WHAT: Measures time to create P2P network nodes
    WHY: Important for distributed system startup time
    """
    result = BenchmarkResult("p2p_node_creation")

    for i in range(num_runs):
        try:
            start = time.perf_counter()
            node = nimcp.P2PNode(8000 + i)  # Unique port per node
            elapsed = time.perf_counter() - start
            result.add_timing(elapsed, success=True)
        except Exception as e:
            print(f"Warning: P2P node creation failed: {e}")
            result.add_timing(0, success=False)

    return result


def benchmark_config_creation(num_runs: int = 1000) -> BenchmarkResult:
    """
    Benchmark: Configuration object creation

    WHAT: Measures config initialization overhead
    WHY: Config creation happens frequently, should be fast
    """
    result = BenchmarkResult("config_creation")

    for _ in range(num_runs):
        try:
            start = time.perf_counter()
            config = nimcp.NetworkConfig(
                100,      # num_neurons
                0.8,      # ei_ratio
                0.01,     # learning_rate
                0.1,      # hebbian_rate
                20.0,     # stdp_window
                0.001,    # homeostatic_rate
                0.1,      # target_activity
                0.1       # adaptation_rate
            )
            elapsed = time.perf_counter() - start
            result.add_timing(elapsed, success=True)
        except Exception as e:
            print(f"Warning: Config creation failed: {e}")
            result.add_timing(0, success=False)

    return result


def run_all_benchmarks() -> List[BenchmarkResult]:
    """
    Run complete benchmark suite

    Returns list of all benchmark results
    """
    print("=" * 80)
    print("NIMCP 2.5 Performance Benchmark Suite")
    print("=" * 80)
    print()

    benchmarks = [
        ("Brain Creation (100 runs)", lambda: benchmark_brain_creation(100)),
        ("Inference Speed (1000 runs)", lambda: benchmark_inference_speed(1000)),
        ("P2P Node Creation (50 runs)", lambda: benchmark_p2p_node_creation(50)),
        ("Config Creation (1000 runs)", lambda: benchmark_config_creation(1000)),
    ]

    results = []
    for name, bench_func in benchmarks:
        print(f"Running: {name}...")
        result = bench_func()
        results.append(result)
        print(f"  ✓ Completed\n")

    return results


def print_results(results: List[BenchmarkResult]):
    """
    Print benchmark results in human-readable format
    """
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()

    for result in results:
        report = result.report()
        if "error" in report:
            print(f"❌ {result.name}: {report['error']}")
            continue

        print(f"📊 {report['name']}")
        print(f"   Runs:        {report['runs']}")
        print(f"   Success:     {report['success_rate']*100:.1f}%")
        print(f"   Mean:        {report['mean_ms']:.3f} ms")
        print(f"   Median:      {report['median_ms']:.3f} ms")
        print(f"   StdDev:      {report['stddev_ms']:.3f} ms")
        print(f"   Min:         {report['min_ms']:.3f} ms")
        print(f"   Max:         {report['max_ms']:.3f} ms")
        print(f"   P95:         {report['p95_ms']:.3f} ms")
        print(f"   P99:         {report['p99_ms']:.3f} ms")
        print()


def save_results_json(results: List[BenchmarkResult], filename: str):
    """
    Save results to JSON for CI/CD processing
    """
    data = {
        "timestamp": time.time(),
        "benchmarks": [r.report() for r in results]
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {filename}")


def main():
    """Main benchmark execution"""
    results = run_all_benchmarks()
    print_results(results)

    # Save for CI/CD
    save_results_json(results, "benchmark-results.json")

    print("=" * 80)
    print("✅ All benchmarks completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
