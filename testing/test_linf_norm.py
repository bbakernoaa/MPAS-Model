#!/usr/bin/env python3
"""
Property-based test for L-Infinity Relative Norm Computation.

Feature: fortran-integration, Property 6: L-Infinity Relative Norm Computation

Validates: Requirements 6.4

For any two floating-point arrays A and B of the same dimensions where max|A| > 0,
the comparison function SHALL compute max|A(i) - B(i)| / max|A(i)| correctly
(i.e., the computed result equals the mathematically expected L-infinity relative
norm to within floating-point precision).
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from compare_dycore_outputs import compute_linf_relative_norm


# --- Strategies ---

# Strategy: generate a shape tuple for arrays (1D to 3D, small sizes for speed)
array_shapes = st.tuples(
    st.integers(min_value=1, max_value=50),
).flatmap(
    lambda s: st.just(s) | st.tuples(
        st.just(s[0]), st.integers(min_value=1, max_value=20)
    ) | st.tuples(
        st.just(s[0]), st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10)
    )
)


def float_arrays(shape):
    """Generate a pair of float64 arrays with the given shape."""
    return st.tuples(
        arrays(dtype=np.float64, shape=shape,
               elements=st.floats(min_value=-1e10, max_value=1e10,
                                  allow_nan=False, allow_infinity=False)),
        arrays(dtype=np.float64, shape=shape,
               elements=st.floats(min_value=-1e10, max_value=1e10,
                                  allow_nan=False, allow_infinity=False)),
    )


# --- Naive reference implementation ---

def naive_linf_relative_norm(ref_data, test_data):
    """
    Naive implementation: max|A(i) - B(i)| / max|A(i)|.
    If max|A| == 0, returns max|A(i) - B(i)| (absolute difference).
    """
    ref = ref_data.astype(np.float64)
    test = test_data.astype(np.float64)
    max_diff = 0.0
    ref_max = 0.0
    for idx in np.ndindex(ref.shape):
        diff = abs(test[idx] - ref[idx])
        if diff > max_diff:
            max_diff = diff
        abs_ref = abs(ref[idx])
        if abs_ref > ref_max:
            ref_max = abs_ref
    if ref_max == 0.0:
        return max_diff
    return max_diff / ref_max


# --- Property Tests ---

class TestLinfNormProperty:
    """Property 6: L-Infinity Relative Norm Computation."""

    @given(
        shape=st.sampled_from([(10,), (5, 4), (3, 4, 5), (50,), (7, 8)]),
        data=st.data(),
    )
    @settings(max_examples=150)
    def test_matches_naive_implementation(self, shape, data):
        """
        For any two arrays of the same shape, the computed L-inf relative norm
        matches the naive element-by-element implementation.

        **Validates: Requirements 6.4**
        """
        ref_data = data.draw(
            arrays(dtype=np.float64, shape=shape,
                   elements=st.floats(min_value=-1e10, max_value=1e10,
                                      allow_nan=False, allow_infinity=False))
        )
        test_data = data.draw(
            arrays(dtype=np.float64, shape=shape,
                   elements=st.floats(min_value=-1e10, max_value=1e10,
                                      allow_nan=False, allow_infinity=False))
        )

        result = compute_linf_relative_norm(ref_data, test_data)
        expected = naive_linf_relative_norm(ref_data, test_data)

        # Allow for floating-point precision differences
        if expected == 0.0:
            assert result == 0.0
        else:
            assert abs(result - expected) / max(abs(expected), 1e-300) < 1e-12

    @given(
        shape=st.sampled_from([(10,), (5, 4), (3, 4, 5), (20,)]),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_identical_arrays_yield_zero(self, shape, data):
        """
        When ref and test are identical, the L-inf relative norm must be 0.

        **Validates: Requirements 6.4**
        """
        ref_data = data.draw(
            arrays(dtype=np.float64, shape=shape,
                   elements=st.floats(min_value=-1e10, max_value=1e10,
                                      allow_nan=False, allow_infinity=False))
        )
        test_data = ref_data.copy()

        result = compute_linf_relative_norm(ref_data, test_data)
        assert result == 0.0

    @given(
        shape=st.sampled_from([(10,), (5, 4), (3, 4, 5)]),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_zero_reference_returns_absolute_diff(self, shape, data):
        """
        When all elements of ref are zero, the function returns the absolute
        max difference (max|test|).

        **Validates: Requirements 6.4**
        """
        ref_data = np.zeros(shape, dtype=np.float64)
        test_data = data.draw(
            arrays(dtype=np.float64, shape=shape,
                   elements=st.floats(min_value=-1e10, max_value=1e10,
                                      allow_nan=False, allow_infinity=False))
        )

        result = compute_linf_relative_norm(ref_data, test_data)
        expected = np.max(np.abs(test_data))

        assert result == expected

    @given(
        shape=st.sampled_from([(10,), (5, 4), (3, 4, 5)]),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_result_is_non_negative(self, shape, data):
        """
        The L-inf relative norm is always >= 0 for any input pair.

        **Validates: Requirements 6.4**
        """
        ref_data = data.draw(
            arrays(dtype=np.float64, shape=shape,
                   elements=st.floats(min_value=-1e10, max_value=1e10,
                                      allow_nan=False, allow_infinity=False))
        )
        test_data = data.draw(
            arrays(dtype=np.float64, shape=shape,
                   elements=st.floats(min_value=-1e10, max_value=1e10,
                                      allow_nan=False, allow_infinity=False))
        )

        result = compute_linf_relative_norm(ref_data, test_data)
        assert result >= 0.0

    @given(
        shape=st.sampled_from([(10,), (5, 4), (3, 4, 5)]),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_norm_bounded_by_ratio_of_maxima(self, shape, data):
        """
        When max|ref| > 0, the L-inf relative norm is bounded above by
        (max|test| + max|ref|) / max|ref|.

        **Validates: Requirements 6.4**
        """
        ref_data = data.draw(
            arrays(dtype=np.float64, shape=shape,
                   elements=st.floats(min_value=-1e10, max_value=1e10,
                                      allow_nan=False, allow_infinity=False))
        )
        assume(np.max(np.abs(ref_data)) > 0.0)

        test_data = data.draw(
            arrays(dtype=np.float64, shape=shape,
                   elements=st.floats(min_value=-1e10, max_value=1e10,
                                      allow_nan=False, allow_infinity=False))
        )

        result = compute_linf_relative_norm(ref_data, test_data)
        ref_max = np.max(np.abs(ref_data))
        test_max = np.max(np.abs(test_data))

        # max|test - ref| <= max|test| + max|ref| by triangle inequality
        upper_bound = (test_max + ref_max) / ref_max

        assert result <= upper_bound + 1e-10  # small tolerance for float ops
