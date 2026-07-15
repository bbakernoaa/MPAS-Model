#!/usr/bin/env python3
"""
Property 7: Parity Threshold Decision

Test that the comparison utility reports PASS iff all differences < 1.0e-13
and FAIL with correct details otherwise. Generates random difference sets
around the 1e-13 boundary.

Feature: fortran-integration, Property 7: Parity Threshold Decision
Validates: Requirements 6.5, 6.6
"""

import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from netCDF4 import Dataset

# Import the comparison function under test
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_dycore_outputs import compare_fields

TOLERANCE = 1.0e-13


def create_netcdf_pair(ref_data_dict, test_data_dict, ntime=1, tmpdir=None):
    """
    Create a pair of temporary NetCDF files with the given field data.

    ref_data_dict and test_data_dict map field_name -> numpy array of shape (ntime, N).
    Returns (ref_path, test_path).
    """
    ref_path = os.path.join(tmpdir, "ref.nc")
    test_path = os.path.join(tmpdir, "test.nc")

    for path, data_dict in [(ref_path, ref_data_dict), (test_path, test_data_dict)]:
        ds = Dataset(path, "w", format="NETCDF4")
        ds.createDimension("Time", ntime)
        for field_name, data in data_dict.items():
            spatial_size = data.shape[1] if data.ndim > 1 else data.shape[0]
            dim_name = f"n{field_name}"
            if dim_name not in ds.dimensions:
                ds.createDimension(dim_name, spatial_size)
            if ntime > 1 or data.ndim > 1:
                var = ds.createVariable(field_name, "f8", ("Time", dim_name))
                var[:] = data
            else:
                var = ds.createVariable(field_name, "f8", (dim_name,))
                var[:] = data
        ds.close()

    return ref_path, test_path


def make_test_array_with_controlled_diff(ref_array, rel_diff):
    """
    Create a test array that has a known L-infinity relative difference
    from the reference array.

    L-inf relative norm = max|test - ref| / max|ref|

    We set test = ref everywhere except at the location of max|ref|,
    where we perturb by rel_diff * max|ref|.
    """
    test_array = ref_array.copy()
    ref_abs = np.abs(ref_array)
    max_ref = np.max(ref_abs)

    if max_ref == 0.0:
        # If ref is all zeros, add absolute difference at first element
        test_array.flat[0] = rel_diff
        return test_array

    # Find the index of max |ref| and perturb there
    max_idx = np.argmax(ref_abs)
    # Add perturbation: the resulting L-inf relative diff will be exactly rel_diff
    test_array.flat[max_idx] = ref_array.flat[max_idx] + rel_diff * max_ref

    return test_array


# Strategy for generating relative differences around the threshold
st_diff_below = st.floats(min_value=1e-16, max_value=9.99e-14)
st_diff_above = st.floats(min_value=1.001e-13, max_value=1e-10)


class TestParityThresholdPass:
    """Test that PASS is reported when all differences are below tolerance."""

    @settings(max_examples=100, deadline=None)
    @given(
        size=st.integers(min_value=5, max_value=50),
        scale=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        rel_diff=st_diff_below,
    )
    def test_all_below_threshold_passes(self, size, scale, rel_diff):
        """
        Property 7: When all field differences are strictly below 1e-13,
        compare_fields must report PASS.

        Feature: fortran-integration, Property 7: Parity Threshold Decision
        Validates: Requirements 6.5, 6.6
        """
        rng = np.random.default_rng(size + int(scale))
        ref_array = rng.uniform(-scale, scale, size=size)
        # Ensure max|ref| > 0
        assume(np.max(np.abs(ref_array)) > 0)

        test_array = make_test_array_with_controlled_diff(ref_array, rel_diff)

        # Reshape to (1, N) for single timestep
        ref_data = {"field_a": ref_array.reshape(1, -1)}
        test_data = {"field_a": test_array.reshape(1, -1)}

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path, test_path = create_netcdf_pair(
                ref_data, test_data, ntime=1, tmpdir=tmpdir
            )
            results, passed = compare_fields(ref_path, test_path, ["field_a"], TOLERANCE)

        assert passed is True, (
            f"Expected PASS but got FAIL. rel_diff={rel_diff}, "
            f"results={results}"
        )
        for r in results:
            assert r["status"] == "PASS"


class TestParityThresholdFail:
    """Test that FAIL is reported when any difference is at or above tolerance."""

    @settings(max_examples=100, deadline=None)
    @given(
        size=st.integers(min_value=5, max_value=50),
        scale=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        rel_diff=st_diff_above,
    )
    def test_above_threshold_fails(self, size, scale, rel_diff):
        """
        Property 7: When any field difference is above 1e-13,
        compare_fields must report FAIL with correct details.

        Feature: fortran-integration, Property 7: Parity Threshold Decision
        Validates: Requirements 6.5, 6.6
        """
        rng = np.random.default_rng(size + int(scale))
        ref_array = rng.uniform(-scale, scale, size=size)
        assume(np.max(np.abs(ref_array)) > 0)

        test_array = make_test_array_with_controlled_diff(ref_array, rel_diff)

        ref_data = {"field_a": ref_array.reshape(1, -1)}
        test_data = {"field_a": test_array.reshape(1, -1)}

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path, test_path = create_netcdf_pair(
                ref_data, test_data, ntime=1, tmpdir=tmpdir
            )
            results, passed = compare_fields(ref_path, test_path, ["field_a"], TOLERANCE)

        assert passed is False, (
            f"Expected FAIL but got PASS. rel_diff={rel_diff}, "
            f"results={results}"
        )
        # Verify that at least one result has FAIL status
        fail_results = [r for r in results if r["status"] == "FAIL"]
        assert len(fail_results) > 0

        # Verify FAIL result contains correct field name and difference info
        for r in fail_results:
            assert r["field"] == "field_a"
            assert "timestep" in r
            assert "difference" in r
            assert isinstance(r["difference"], float)
            assert r["difference"] >= TOLERANCE

    @settings(max_examples=50, deadline=None)
    @given(
        size=st.integers(min_value=5, max_value=50),
        sign=st.sampled_from([-1.0, 1.0]),
        perturb_idx=st.integers(min_value=1, max_value=49),
    )
    def test_at_boundary_fails(self, size, sign, perturb_idx):
        """
        Property 7: When a field difference is exactly at 1e-13,
        compare_fields must report FAIL (strictly less than is required for PASS).

        Feature: fortran-integration, Property 7: Parity Threshold Decision
        Validates: Requirements 6.5, 6.6
        """
        # To achieve an exact L-inf relative diff of TOLERANCE, we:
        # 1. Set max|ref| = 1.0 (first element = sign * 1.0)
        # 2. Set all other ref elements to 0.0
        # 3. Perturb a ZERO element by adding TOLERANCE
        # This way diff = |TOLERANCE - 0| = TOLERANCE exactly (no cancellation),
        # and ratio = TOLERANCE / 1.0 = TOLERANCE.
        idx = perturb_idx % size
        if idx == 0:
            idx = 1  # Don't perturb the max-magnitude element

        ref_array = np.zeros(size)
        ref_array[0] = sign * 1.0

        test_array = ref_array.copy()
        test_array[idx] = TOLERANCE  # Exact perturbation at a zero element

        ref_data = {"field_a": ref_array.reshape(1, -1)}
        test_data = {"field_a": test_array.reshape(1, -1)}

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path, test_path = create_netcdf_pair(
                ref_data, test_data, ntime=1, tmpdir=tmpdir
            )
            results, passed = compare_fields(ref_path, test_path, ["field_a"], TOLERANCE)

        # Difference exactly at tolerance should FAIL (PASS requires strictly less than)
        assert passed is False, (
            f"Expected FAIL at boundary but got PASS. "
            f"results={results}"
        )


class TestParityThresholdMultiField:
    """Test threshold behavior with multiple fields."""

    @settings(max_examples=50, deadline=None)
    @given(
        size=st.integers(min_value=5, max_value=30),
        scale=st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        diff_below=st_diff_below,
        diff_above=st_diff_above,
    )
    def test_mixed_fields_one_failing(self, size, scale, diff_below, diff_above):
        """
        Property 7: If one field is below threshold and another is above,
        overall result is FAIL, and the failing field is correctly identified.

        Feature: fortran-integration, Property 7: Parity Threshold Decision
        Validates: Requirements 6.5, 6.6
        """
        rng = np.random.default_rng(size + int(scale))
        ref_a = rng.uniform(-scale, scale, size=size)
        ref_b = rng.uniform(-scale, scale, size=size)
        assume(np.max(np.abs(ref_a)) > 0)
        assume(np.max(np.abs(ref_b)) > 0)

        # field_a passes, field_b fails
        test_a = make_test_array_with_controlled_diff(ref_a, diff_below)
        test_b = make_test_array_with_controlled_diff(ref_b, diff_above)

        ref_data = {"field_a": ref_a.reshape(1, -1), "field_b": ref_b.reshape(1, -1)}
        test_data = {"field_a": test_a.reshape(1, -1), "field_b": test_b.reshape(1, -1)}

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path, test_path = create_netcdf_pair(
                ref_data, test_data, ntime=1, tmpdir=tmpdir
            )
            results, passed = compare_fields(
                ref_path, test_path, ["field_a", "field_b"], TOLERANCE
            )

        assert passed is False, (
            f"Expected overall FAIL when one field exceeds threshold. "
            f"results={results}"
        )

        # field_a should PASS
        field_a_results = [r for r in results if r["field"] == "field_a"]
        assert all(r["status"] == "PASS" for r in field_a_results)

        # field_b should FAIL with correct details
        field_b_results = [r for r in results if r["field"] == "field_b"]
        assert any(r["status"] == "FAIL" for r in field_b_results)
        for r in field_b_results:
            if r["status"] == "FAIL":
                assert r["field"] == "field_b"
                assert "timestep" in r
                assert isinstance(r["difference"], float)
                assert r["difference"] >= TOLERANCE


class TestParityThresholdMultiTimestep:
    """Test threshold behavior across multiple timesteps."""

    @settings(max_examples=50, deadline=None)
    @given(
        size=st.integers(min_value=5, max_value=20),
        scale=st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        diff_below=st_diff_below,
        diff_above=st_diff_above,
        fail_timestep=st.integers(min_value=0, max_value=3),
    )
    def test_fail_at_specific_timestep(
        self, size, scale, diff_below, diff_above, fail_timestep
    ):
        """
        Property 7: When a field fails at a specific timestep, the result
        correctly identifies the failing timestep.

        Feature: fortran-integration, Property 7: Parity Threshold Decision
        Validates: Requirements 6.5, 6.6
        """
        ntime = 4
        fail_t = fail_timestep % ntime

        rng = np.random.default_rng(size + int(scale))

        # Build ref array: (ntime, size)
        ref_data_2d = rng.uniform(-scale, scale, size=(ntime, size))
        assume(all(np.max(np.abs(ref_data_2d[t])) > 0 for t in range(ntime)))

        # Build test array: all timesteps pass except fail_t
        test_data_2d = np.zeros_like(ref_data_2d)
        for t in range(ntime):
            if t == fail_t:
                test_data_2d[t] = make_test_array_with_controlled_diff(
                    ref_data_2d[t], diff_above
                )
            else:
                test_data_2d[t] = make_test_array_with_controlled_diff(
                    ref_data_2d[t], diff_below
                )

        ref_data = {"field_a": ref_data_2d}
        test_data = {"field_a": test_data_2d}

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path, test_path = create_netcdf_pair(
                ref_data, test_data, ntime=ntime, tmpdir=tmpdir
            )
            results, passed = compare_fields(ref_path, test_path, ["field_a"], TOLERANCE)

        assert passed is False

        # Verify the failing timestep is reported
        fail_results = [r for r in results if r["status"] == "FAIL"]
        assert len(fail_results) >= 1

        fail_timesteps = [r["timestep"] for r in fail_results]
        assert fail_t in fail_timesteps, (
            f"Expected timestep {fail_t} in failures, got {fail_timesteps}"
        )

        # Verify the reported difference for the failing timestep is >= tolerance
        for r in fail_results:
            if r["timestep"] == fail_t:
                assert r["difference"] >= TOLERANCE
