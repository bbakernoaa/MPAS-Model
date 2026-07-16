#!/usr/bin/env python3
"""
Compare two NetCDF dycore output files for field parity using xarray.

Computes per-field L-infinity relative differences between a reference run
and a C++ dycore run. Reports PASS/FAIL based on a configurable tolerance
(default 1.0e-13).

L-infinity relative norm: max|A(i) - B(i)| / max|A(i)|
where A is the reference field.

Exit codes:
  0 - PASS (all fields within tolerance)
  1 - FAIL (one or more fields exceed tolerance)
  2 - ERROR (file I/O or argument error)
"""

import argparse
import json
import sys
import numpy as np
import xarray as xr

DEFAULT_TOLERANCE = 1.0e-13
DEFAULT_FIELDS = ["u", "w", "theta_m", "rho_zz"]


def compare_fields(ref_path, test_path, fields, tolerance):
    """
    Compare prognostic fields between reference and test NetCDF files using xarray.

    Returns a tuple (results, passed) where:
      - results: list of dicts with field, timestep, difference, and status
      - passed: bool indicating overall PASS/FAIL
    """
    try:
        ref_ds = xr.open_dataset(ref_path)
        test_ds = xr.open_dataset(test_path)
    except Exception as e:
        print(f"ERROR: Failed to open NetCDF files: {e}", file=sys.stderr)
        sys.exit(2)

    results = []
    overall_pass = True

    for field_name in fields:
        if field_name not in ref_ds:
            results.append({
                "field": field_name,
                "timestep": "N/A",
                "difference": "N/A",
                "status": "SKIP",
                "message": f"Field '{field_name}' not found in reference file"
            })
            continue

        if field_name not in test_ds:
            results.append({
                "field": field_name,
                "timestep": "N/A",
                "difference": "N/A",
                "status": "FAIL",
                "message": f"Field '{field_name}' not found in test file"
            })
            overall_pass = False
            continue

        ref_var = ref_ds[field_name]
        test_var = test_ds[field_name]

        if ref_var.shape != test_var.shape:
            results.append({
                "field": field_name,
                "timestep": "N/A",
                "difference": "N/A",
                "status": "FAIL",
                "message": (
                    f"Shape mismatch for '{field_name}': "
                    f"ref={ref_var.shape}, test={test_var.shape}"
                )
            })
            overall_pass = False
            continue

        # Determine if there is a Time dimension
        if "Time" in ref_var.dims:
            ntime = ref_var.sizes["Time"]
            for t in range(ntime):
                r_slice = ref_var.isel(Time=t).values
                t_slice = test_var.isel(Time=t).values

                # Compute L-infinity relative norm
                diff = np.abs(t_slice - r_slice)
                max_diff = np.max(diff)
                ref_max = np.max(np.abs(r_slice))

                rel_diff = max_diff / ref_max if ref_max != 0.0 else max_diff
                status = "PASS" if rel_diff < tolerance else "FAIL"
                if status == "FAIL":
                    overall_pass = False

                results.append({
                    "field": field_name,
                    "timestep": t,
                    "difference": float(rel_diff),
                    "status": status
                })
        else:
            r_val = ref_var.values
            t_val = test_var.values
            diff = np.abs(t_val - r_val)
            max_diff = np.max(diff)
            ref_max = np.max(np.abs(r_val))

            rel_diff = max_diff / ref_max if ref_max != 0.0 else max_diff
            status = "PASS" if rel_diff < tolerance else "FAIL"
            if status == "FAIL":
                overall_pass = False

            results.append({
                "field": field_name,
                "timestep": 0,
                "difference": float(rel_diff),
                "status": status
            })

    try:
        ref_ds.close()
        test_ds.close()
    except Exception:
        pass

    return results, overall_pass


def format_tabular(results, tolerance, overall_pass):
    """Format results as a human-readable table."""
    lines = []
    lines.append(f"{'Field':<15} {'Timestep':<10} {'L-inf Rel Diff':<22} {'Status':<6}")
    lines.append("-" * 55)

    for r in results:
        diff_str = (
            f"{r['difference']:.6e}" if isinstance(r["difference"], float)
            else str(r["difference"])
        )
        ts_str = str(r["timestep"])
        msg = r.get("message", "")
        if msg:
            lines.append(f"{r['field']:<15} {ts_str:<10} {msg}")
        else:
            lines.append(f"{r['field']:<15} {ts_str:<10} {diff_str:<22} {r['status']:<6}")

    lines.append("-" * 55)
    lines.append(f"Tolerance: {tolerance:.2e}")
    lines.append(f"Result: {'PASS' if overall_pass else 'FAIL'}")
    return "\n".join(lines)


def format_json(results, tolerance, overall_pass):
    """Format results as JSON."""
    output = {
        "tolerance": tolerance,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "fields": results
    }
    return json.dumps(output, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two NetCDF dycore output files for field parity."
    )
    parser.add_argument(
        "reference",
        help="Path to the reference NetCDF file (Fortran dycore output)"
    )
    parser.add_argument(
        "test",
        help="Path to the test NetCDF file (C++ dycore output)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Parity tolerance for L-inf relative norm (default: {DEFAULT_TOLERANCE:.1e})"
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help=f"Prognostic fields to compare (default: {' '.join(DEFAULT_FIELDS)})"
    )
    parser.add_argument(
        "--format",
        choices=["json", "table"],
        default="table",
        help="Output format (default: table)"
    )

    args = parser.parse_args()

    results, overall_pass = compare_fields(
        args.reference, args.test, args.fields, args.tolerance
    )

    if args.format == "json":
        print(format_json(results, args.tolerance, overall_pass))
    else:
        print(format_tabular(results, args.tolerance, overall_pass))

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
