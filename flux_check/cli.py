#!/usr/bin/env python3
"""
flux-check CLI — FLUX constraint engine from the command line.

Commands:
    flux-check check --preset <name> --values v1,v2,...
    flux-check batch --preset <name> --input data.csv --output results.csv
    flux-check fracture --graph graph.json
    flux-check bench --preset <name> [--iterations N]
    flux-check presets
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from typing import List, Optional

import numpy as np

from flux_check.core import FluxExact, Severity, severity, passed
from flux_check.presets import PRESETS, list_presets, get_preset, get_preset_description
from flux_check.fracture import Fracturer, DependencyGraph


def cmd_check(args):
    """Check values against a preset."""
    try:
        constraints = get_preset(args.preset)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    values = [float(v.strip()) for v in args.values.split(",")]
    if len(values) != len(constraints):
        print(f"Error: got {len(values)} values but preset '{args.preset}' has {len(constraints)} constraints",
              file=sys.stderr)
        return 1

    fc = FluxExact(constraints)
    print(f"Preset: {args.preset} ({get_preset_description(args.preset)})")
    print(f"{'Constraint':<25} {'Value':>10} {'Lo':>10} {'Hi':>10} {'Status':<10}")
    print("-" * 70)

    any_fail = False
    for i, (v, c) in enumerate(zip(values, constraints)):
        result = fc.check(v)
        detail = result.details[i]
        status = "PASS" if detail.passed else (
            f"FAIL({('lo' if detail.lo_violated else 'hi')})")
        if not detail.passed:
            any_fail = True
        print(f"{c['name']:<25} {v:>10.4g} {c['lo']:>10.4g} {c['hi']:>10.4g} {status:<10}")

    mask = fc.check_mask(values[0])  # combined from individual checks
    print()
    if any_fail:
        print(f"Result: FAIL — constraint violations detected")
        return 1
    else:
        print(f"Result: PASS — all constraints satisfied")
        return 0


def cmd_batch(args):
    """Batch check from CSV file."""
    try:
        constraints = get_preset(args.preset)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    fc = FluxExact(constraints)
    names = [c["name"] for c in constraints]
    n_constraints = len(constraints)

    # Read input CSV
    rows = []
    with open(args.input, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            print("Error: empty CSV file", file=sys.stderr)
            return 1
        for row in reader:
            rows.append(row)

    if not rows:
        print("Error: no data rows in CSV", file=sys.stderr)
        return 1

    # Validate column count
    if len(header) != n_constraints:
        print(f"Warning: CSV has {len(header)} columns, preset has {n_constraints} constraints",
              file=sys.stderr)

    # Process
    output_rows = []
    total_pass = 0
    total_fail = 0

    for row_idx, row in enumerate(rows):
        try:
            vals = [float(v.strip()) for v in row[:n_constraints]]
        except (ValueError, IndexError) as e:
            print(f"Row {row_idx + 1}: parse error: {e}", file=sys.stderr)
            continue

        # Check each value against its corresponding constraint
        row_results = []
        for j, v in enumerate(vals):
            result = fc.check(v)
            row_results.append(result.details[j].passed)

        all_pass = all(row_results)
        if all_pass:
            total_pass += 1
        else:
            total_fail += 1

        output_rows.append(row + ["PASS" if all_pass else "FAIL"] +
                          ["PASS" if p else "FAIL" for p in row_results])

    # Write output
    out_header = header + ["overall"] + [f"{n}_status" for n in names]
    outfile = args.output or args.input.replace(".csv", "_results.csv")
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        writer.writerows(output_rows)

    print(f"Batch complete: {total_pass} pass, {total_fail} fail, {len(rows)} total rows")
    print(f"Output: {outfile}")
    return 0 if total_fail == 0 else 1


def cmd_fracture(args):
    """Fracture a constraint graph into independent blocks."""
    try:
        with open(args.graph, "r") as f:
            graph_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {args.graph}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        return 1

    # Expected format: {"constraints": [{"dims": [0, 1]}, ...]} or {"adjacency": [[...], ...]}
    if "adjacency" in graph_data:
        adj = np.array(graph_data["adjacency"], dtype=np.uint8)
        cnames = graph_data.get("constraint_names", [])
        dnames = graph_data.get("dimension_names", [])
        graph = DependencyGraph.from_adjacency(adj, cnames, dnames)
    elif "constraints" in graph_data:
        constraints = graph_data["constraints"]
        masks = []
        cnames = []
        for i, c in enumerate(constraints):
            dims = c.get("dims", [i])
            masks.append(np.array(dims, dtype=np.intp))
            cnames.append(c.get("name", f"c{i}"))
        graph = DependencyGraph.from_masks(masks, cnames)
    else:
        print("Error: JSON must contain 'adjacency' or 'constraints' key", file=sys.stderr)
        return 1

    fracturer = Fracturer()
    result = fracturer.fracture(graph)
    summary = result.summary()

    print("Fracture Analysis")
    print("=" * 50)
    print(f"Constraints: {summary['n_constraints']}")
    print(f"Dimensions:  {summary['n_dimensions']}")
    print(f"Blocks:      {summary['n_blocks']}")
    print(f"Largest:     {summary['largest_block_size']} constraints")
    print(f"Speedup:     {summary['speedup_potential']}x")
    print()
    for i, block in enumerate(result.blocks):
        print(f"  Block {i}: {block.size} constraints, {len(block.dimension_indices)} dimensions")
        if block.constraint_indices:
            names = [graph.constraint_names[j] for j in block.constraint_indices
                     if j < len(graph.constraint_names)]
            print(f"    Constraints: {names}")
            print(f"    Dimensions:  {block.dimension_indices}")

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {args.json_output}")

    return 0


def cmd_bench(args):
    """Run performance benchmark."""
    try:
        constraints = get_preset(args.preset)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    iterations = args.iterations or 1_000_000
    fc = FluxExact(constraints)

    print(f"Benchmark: {args.preset} preset, {iterations:,} iterations, {fc.n} constraints")
    print()

    # Hot path benchmark
    t0 = time.perf_counter()
    for i in range(iterations):
        fc.check_mask((i % 1000) - 500)
    elapsed = time.perf_counter() - t0
    rate = iterations / elapsed
    print(f"check_mask (hot path):")
    print(f"  {rate / 1e6:.2f}M checks/sec")
    print(f"  {elapsed * 1000:.1f}ms total")
    print(f"  {elapsed / iterations * 1e9:.1f}ns per check")

    # Batch benchmark
    arr = np.random.randn(100_000)
    t0 = time.perf_counter()
    for _ in range(10):
        fc.check_batch_numpy(arr)
    elapsed = time.perf_counter() - t0
    batch_rate = 1_000_000 / elapsed * 10
    print(f"\ncheck_batch_numpy (100K array, 10 runs):")
    print(f"  {batch_rate / 1e6:.2f}M values/sec")
    print(f"  {elapsed * 1000:.1f}ms total")

    # Detail benchmark
    detail_iters = min(iterations, 100_000)
    t0 = time.perf_counter()
    for i in range(detail_iters):
        fc.check_detail((i % 1000) - 500)
    elapsed = time.perf_counter() - t0
    rate = detail_iters / elapsed
    print(f"\ncheck_detail ({detail_iters:,} iterations):")
    print(f"  {rate / 1e6:.2f}M checks/sec")
    print(f"  {elapsed * 1000:.1f}ms total")

    return 0


def cmd_presets(args):
    """List available presets."""
    print("Available Presets")
    print("=" * 60)
    for name, data in PRESETS.items():
        n = len(data["constraints"])
        desc = data["description"]
        print(f"  {name:<15} ({n} constraints) — {desc}")
    print()
    print("Usage: flux-check check --preset <name> --values v1,v2,...")


def main():
    parser = argparse.ArgumentParser(
        prog="flux-check",
        description="FLUX constraint engine CLI — zero false negatives",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # check
    p_check = sub.add_parser("check", help="Check values against a preset")
    p_check.add_argument("--preset", required=True, help="Preset name")
    p_check.add_argument("--values", required=True, help="Comma-separated values")

    # batch
    p_batch = sub.add_parser("batch", help="Batch check from CSV")
    p_batch.add_argument("--preset", required=True, help="Preset name")
    p_batch.add_argument("--input", required=True, help="Input CSV file")
    p_batch.add_argument("--output", help="Output CSV file (default: input_results.csv)")

    # fracture
    p_fracture = sub.add_parser("fracture", help="Fracture constraint graph into blocks")
    p_fracture.add_argument("--graph", required=True, help="Graph JSON file")
    p_fracture.add_argument("--json-output", help="Save summary as JSON")

    # bench
    p_bench = sub.add_parser("bench", help="Run performance benchmark")
    p_bench.add_argument("--preset", required=True, help="Preset name")
    p_bench.add_argument("--iterations", type=int, default=1_000_000, help="Iterations")

    # presets
    sub.add_parser("presets", help="List available presets")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "check": cmd_check,
        "batch": cmd_batch,
        "fracture": cmd_fracture,
        "bench": cmd_bench,
        "presets": cmd_presets,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
