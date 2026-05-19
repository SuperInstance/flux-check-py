"""Tests for flux_check.cli — command-line interface."""

import csv
import json
import os
import subprocess
import sys
import tempfile

import pytest


def run_cli(*args):
    """Run flux-check CLI and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "flux_check.cli"] + list(args),
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout, result.stderr


class TestPresetsCommand:

    def test_presets_lists_all(self):
        rc, out, err = run_cli("presets")
        assert rc == 0
        assert "automotive" in out
        assert "aviation" in out
        assert "medical" in out
        assert "energy" in out
        assert "iot" in out
        assert "financial" in out


class TestCheckCommand:

    def test_check_pass(self):
        rc, out, err = run_cli("check", "--preset", "automotive",
                               "--values", "3000,50,90,50,100,0,12,75")
        assert rc == 0
        assert "PASS" in out

    def test_check_fail(self):
        rc, out, err = run_cli("check", "--preset", "automotive",
                               "--values", "9000,50,90,50,100,0,12,75")
        assert rc == 1
        assert "FAIL" in out

    def test_check_wrong_count(self):
        rc, out, err = run_cli("check", "--preset", "automotive",
                               "--values", "1,2,3")
        assert rc == 1
        assert "Error" in err

    def test_check_invalid_preset(self):
        rc, out, err = run_cli("check", "--preset", "nonexistent",
                               "--values", "1,2,3")
        assert rc == 1


class TestBatchCommand:

    def test_batch_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["engine_rpm", "vehicle_speed_kmh", "coolant_temp_c",
                           "throttle_pct", "brake_pressure_bar", "steering_angle_deg",
                           "battery_voltage_v", "fuel_level_pct"])
            writer.writerow([3000, 50, 90, 50, 100, 0, 12, 75])
            writer.writerow([9000, 50, 90, 50, 100, 0, 12, 75])
            f.flush()
            infile = f.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as outf:
                outfile = outf.name

            rc, out, err = run_cli("batch", "--preset", "automotive",
                                   "--input", infile, "--output", outfile)
            assert rc == 1  # one row fails
            assert "1 pass" in out
            assert "1 fail" in out

            # Verify output file
            with open(outfile) as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "overall" in header
                rows = list(reader)
                assert len(rows) == 2
        finally:
            os.unlink(infile)
            if os.path.exists(outfile):
                os.unlink(outfile)


class TestFractureCommand:

    def test_fracture_from_constraints(self):
        data = {
            "constraints": [
                {"dims": [0, 1], "name": "c0"},
                {"dims": [0], "name": "c1"},
                {"dims": [2], "name": "c2"},
                {"dims": [2], "name": "c3"},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            infile = f.name

        try:
            rc, out, err = run_cli("fracture", "--graph", infile)
            assert rc == 0
            assert "Blocks:" in out
            assert "2" in out  # 2 blocks
        finally:
            os.unlink(infile)

    def test_fracture_from_adjacency(self):
        data = {
            "adjacency": [[1, 0], [0, 1]],
            "constraint_names": ["c0", "c1"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            infile = f.name

        try:
            rc, out, err = run_cli("fracture", "--graph", infile)
            assert rc == 0
        finally:
            os.unlink(infile)


class TestBenchCommand:

    def test_bench_runs(self):
        rc, out, err = run_cli("bench", "--preset", "automotive", "--iterations", "10000")
        assert rc == 0
        assert "checks/sec" in out
