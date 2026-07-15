cmake_minimum_required(VERSION 3.21)

# Try locating Google Test globally
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found via find_package. Fetching via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )
    # Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    if(NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
endif()

# Locate RapidCheck (property-based testing)
find_package(rapidcheck QUIET)
if(NOT rapidcheck_FOUND)
    # Fall back to FetchContent for environments without a global install.
    message(STATUS "rapidcheck not found via find_package. Fetching via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        GIT_SUBMODULES ""
    )
    set(RC_ENABLE_GTEST ON CACHE BOOL "Build RapidCheck GTest integration" FORCE)
    FetchContent_MakeAvailable(rapidcheck)
endif()

# RapidCheck GTest extras if not found globally
if(NOT TARGET rapidcheck_gtest)
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest")
        add_subdirectory(
            ${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest
            ${CMAKE_CURRENT_BINARY_DIR}/rapidcheck_gtest
        )
    endif()
endif()

# Build Component Test Runner
add_executable(component_test_suite
    main.cpp
    test_placeholder.cpp
    test_field_store.cpp
    test_field_store_property_dualview.cpp
    test_field_store_property_extents.cpp
    test_les_options.cpp
    test_halo_group_registry.cpp
    test_halo_manager.cpp
    test_accumulation.cpp
    test_accumulation_property.cpp
    test_iau_module.cpp
    test_diagnostics_module.cpp
    test_dissipation_module.cpp
    test_dissipation_eddy_visc_property.cpp
    test_boundary_module.cpp
    test_halo_exchange_integration.cpp
    test_tridiagonal_solve.cpp
    test_tridiagonal_solve_property.cpp
    test_scalar_transport.cpp
    test_scalar_transport_mono.cpp
    test_scalar_transport_mono_property.cpp
    test_scalar_transport_mono_property_water.cpp
    test_dyn_tend_module.cpp
    test_acoustic_solver.cpp
    test_time_integrator.cpp
    test_time_integrator_advance.cpp
    test_post_timestep_diagnostics.cpp
    test_dry_mass_conservation_property.cpp
    test_execution_space_parity_property.cpp
    test_end_to_end_parity.cpp
)

target_link_libraries(component_test_suite
    PRIVATE
        dycore_cpp
        GTest::gtest
        rapidcheck
)

# Kokkos requires OpenMP link flags — ensure they propagate to the test binary
target_link_options(component_test_suite PRIVATE -fopenmp)

# Add rapidcheck_gtest link dependency
if(TARGET rapidcheck_gtest)
    target_link_libraries(component_test_suite PRIVATE rapidcheck_gtest)
elseif(TARGET rapidcheck-gtest)
    target_link_libraries(component_test_suite PRIVATE rapidcheck-gtest)
endif()

# Register the CTest test with custom environment for default iteration count (>=100)
add_test(NAME component_test_suite COMMAND $<TARGET_FILE:component_test_suite>)
set_tests_properties(component_test_suite PROPERTIES
    TIMEOUT 120
    ENVIRONMENT "RC_PARAMS=num_tests=100"
)

# ─── Interop Boundary Test Suite ─────────────────────────────────────────────
# Separate executable: does NOT pre-initialize Kokkos because the interop tests
# exercise the full Kokkos lifecycle through dycore_init / dycore_finalize.
add_executable(interop_test_suite
    main_interop.cpp
    test_interop_boundary.cpp
)

target_link_libraries(interop_test_suite
    PRIVATE
        dycore_cpp
        GTest::gtest
)

# Kokkos requires OpenMP link flags
target_link_options(interop_test_suite PRIVATE -fopenmp)

add_test(NAME interop_test_suite COMMAND $<TARGET_FILE:interop_test_suite>)
set_tests_properties(interop_test_suite PROPERTIES
    TIMEOUT 60
)

# ─── Index Remapping Property Test Suite ─────────────────────────────────────
# Separate executable: does NOT link against the full dycore.
# Tests the pure index remapping logic (1-based → 0-based) independently.
add_executable(index_remapping_test_suite
    test_index_remapping.cpp
)

target_link_libraries(index_remapping_test_suite
    PRIVATE
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME index_remapping_test_suite COMMAND $<TARGET_FILE:index_remapping_test_suite>)
set_tests_properties(index_remapping_test_suite PROPERTIES
    TIMEOUT 60
)

# ─── FFI Value Passthrough Property Test Suite ───────────────────────────────
# Separate executable: exercises the C API FFI boundary with random valid scalars.
# Does NOT pre-initialize Kokkos (the full lifecycle test calls dycore_init itself).
add_executable(ffi_passthrough_test_suite
    test_ffi_passthrough.cpp
)

target_link_libraries(ffi_passthrough_test_suite
    PRIVATE
        dycore_cpp
        GTest::gtest
        GTest::gtest_main
)

# Kokkos requires OpenMP link flags
target_link_options(ffi_passthrough_test_suite PRIVATE -fopenmp)

add_test(NAME ffi_passthrough_test_suite COMMAND $<TARGET_FILE:ffi_passthrough_test_suite>)
set_tests_properties(ffi_passthrough_test_suite PROPERTIES
    TIMEOUT 60
)

# ─── Dimension Validation Property Test Suite ────────────────────────────────
# Separate executable: does NOT pre-initialize Kokkos because the test verifies
# that invalid dimensions cause dycore_init to return error code 2 WITHOUT
# initializing Kokkos. Links against the full dycore to call dycore_init.
add_executable(dimension_validation_test_suite
    main_interop.cpp
    test_dimension_validation.cpp
)

target_link_libraries(dimension_validation_test_suite
    PRIVATE
        dycore_cpp
        GTest::gtest
)

# Kokkos requires OpenMP link flags
target_link_options(dimension_validation_test_suite PRIVATE -fopenmp)

add_test(NAME dimension_validation_test_suite COMMAND $<TARGET_FILE:dimension_validation_test_suite>)
set_tests_properties(dimension_validation_test_suite PROPERTIES
    TIMEOUT 60
)

# ─── Dynamic Bindings Unit Test Suite ────────────────────────────────────────
add_executable(test_dynamic_bindings
    main_interop.cpp
    test_dynamic_bindings.cpp
)

target_link_libraries(test_dynamic_bindings
    PRIVATE
        dycore_cpp
        GTest::gtest
)

# Kokkos requires OpenMP link flags
target_link_options(test_dynamic_bindings PRIVATE -fopenmp)

add_test(NAME test_dynamic_bindings COMMAND $<TARGET_FILE:test_dynamic_bindings>)
set_tests_properties(test_dynamic_bindings PROPERTIES
    TIMEOUT 60
)
