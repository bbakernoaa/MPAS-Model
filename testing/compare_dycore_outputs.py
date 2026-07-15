#!/usr/bin/env python3
"""
Compare two NetCDF dycore output files for field parity.

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

try:
    from netCDF4 import Dataset
except ImportError:
    try:
        from scipy.io import netcdf_file
    except ImportError:
        print("ERROR: Neither netCDF4 nor scipy.io.netcdf is available.", file=sys.stderr)
        sys.exit(2)


DEFAULT_TOLERANCE = 1.0e-13
DEFAULT_FIELDS = ["u", "w", "theta_m", "rho_zz", "scalars"]


def open_dataset(filepath):
    """Open a NetCDF file and return the dataset object."""
    try:
        return Dataset(filepath, "r")
    except NameError:
        # netCDF4 not available, fall back to scipy
        from scipy.io import netcdf_file
        return netcdf_file(filepath, "r", mmap=False)


def get_variable_data(dataset, varname):
    """Get variable data as a numpy array, handling both netCDF4 and scipy interfaces."""
    if hasattr(dataset, "variables"):
        if varname not in dataset.variables:
            return None
        var = dataset.variables[varname]
        # netCDF4 Dataset
        if hasattr(var, "__getitem__"):
            return np.asarray(var[:])
        # scipy netcdf_file
        return np.asarray(var.data)
    return None


def get_time_dimension(dataset):
    """Determine the number of time steps in the dataset."""
    # Common time dimension names in MPAS output
    for dimname in ["Time", "time", "nTime"]:
        if hasattr(dataset, "dimensions"):
            dims = dataset.dimensions
            if dimname in dims:
                # netCDF4 style
                if hasattr(dims[dimname], "__len__"):
                    return len(dims[dimname])
                # scipy style - dimensions is a dict of sizes
                return dims[dimname]
    return None


def compute_linf_relative_norm(ref_data, test_data):
    """
    Compute the L-infinity relative norm between reference and test arrays.

    Returns: max|test - ref| / max|ref|

    If max|ref| == 0, returns max|test - ref| (absolute difference).
    """
    diff = np.abs(test_data.astype(np.float64) - ref_data.astype(np.float64))
    max_diff = np.max(diff)

    ref_max = np.max(np.abs(ref_data.astype(np.float64)))

    if ref_max == 0.0:
        return max_diff

    return max_diff / ref_max


def compare_fields(ref_path, test_path, fields, tolerance):
    """
    Compare prognostic fields between reference and test NetCDF files.

    Returns a tuple (results, passed) where:
      - results: list of dicts with field, timestep, difference, and status
      - passed: bool indicating overall PASS/FAIL
    """
    try:
        ref_ds = open_dataset(ref_path)
        test_ds = open_dataset(test_path)
    except Exception as e:
        print(f"ERROR: Failed to open NetCDF files: {e}", file=sys.stderr)
        sys.exit(2)

    results = []
    overall_pass = True

    for field_name in fields:
        ref_data = get_variable_data(ref_ds, field_name)
        test_data = get_variable_data(test_ds, field_name)

        if ref_data is None:
            results.append({
                "field": field_name,
                "timestep": "N/A",
                "difference": "N/A",
                "status": "SKIP",
                "message": f"Field '{field_name}' not found in reference file"
            })
            continue

        if test_data is None:
            results.append({
                "field": field_name,
                "timestep": "N/A",
                "difference": "N/A",
                "status": "FAIL",
                "message": f"Field '{field_name}' not found in test file"
            })
            overall_pass = False
            continue

        if ref_data.shape != test_data.shape:
            results.append({
                "field": field_name,
                "timestep": "N/A",
                "difference": "N/A",
                "status": "FAIL",
                "message": (
                    f"Shape mismatch for '{field_name}': "
                    f"ref={ref_data.shape}, test={test_data.shape}"
                )
            })
            overall_pass = False
            continue

        # Determine if there is a time dimension (first axis)
        ntime = get_time_dimension(ref_ds)

        if ntime is not None and ref_data.ndim >= 1 and ref_data.shape[0] == ntime:
            # Multiple time steps - compare each independently
            for t in range(ntime):
                ref_slice = ref_data[t]
                test_slice = test_data[t]
                diff = compute_linf_relative_norm(ref_slice, test_slice)
                status = "PASS" if diff < tolerance else "FAIL"
                if status == "FAIL":
                    overall_pass = False
                results.append({
                    "field": field_name,
                    "timestep": t,
                    "difference": diff,
                    "status": status
                })
        else:
            # Single time step or no time dimension
            diff = compute_linf_relative_norm(ref_data, test_data)
            status = "PASS" if diff < tolerance else "FAIL"
            if status == "FAIL":
                overall_pass = False
            results.append({
                "field": field_name,
                "timestep": 0,
                "difference": diff,
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
