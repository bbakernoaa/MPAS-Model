# MPAS Dycore C++ Tests

## Running Tests

```bash
cd build
cmake ..
cmake --build .
ctest --output-on-failure
```

## Test Categories

- **Unit tests** (`test_*`): GoogleTest-based tests for individual components
- **Property tests** (`prop_*`): RapidCheck-based property testing for kernel correctness

## Test Targets

All tests are automatically discovered by CTest. Run `ctest -N` to list all available tests.
