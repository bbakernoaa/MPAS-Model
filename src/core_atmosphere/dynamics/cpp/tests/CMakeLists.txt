# Test targets for mpas_dycore_cpp
# Unit tests and property-based tests are added here as they are implemented.

# Common test utilities (placeholder until actual test sources are added)
# add_library(test_utils STATIC test_utils.cpp)
# target_link_libraries(test_utils PUBLIC mpas_dycore_cpp GTest::gtest)

# Helper function to register a unit test
function(mpas_add_unit_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE
        mpas_dycore_cpp
        GTest::gtest
        GTest::gtest_main
        MPI::MPI_CXX
    )
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Helper function to register a property-based test (RapidCheck)
function(mpas_add_property_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE
        mpas_dycore_cpp
        GTest::gtest
        GTest::gtest_main
        rapidcheck
        rapidcheck_gtest
        MPI::MPI_CXX
    )
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Unit tests
mpas_add_unit_test(test_accessors test_accessors.cpp)
mpas_add_unit_test(test_mesh test_mesh.cpp)
mpas_add_unit_test(test_state test_state.cpp)

# Property tests
mpas_add_property_test(prop_field_views property/prop_field_views.cpp)
mpas_add_property_test(prop_constants property/prop_constants.cpp)
mpas_add_property_test(prop_connectivity property/prop_connectivity.cpp)
target_compile_definitions(prop_connectivity PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_execution_policy property/prop_execution_policy.cpp)
mpas_add_property_test(prop_halo property/prop_halo.cpp)
mpas_add_property_test(prop_api_lifecycle property/prop_api_lifecycle.cpp)
mpas_add_property_test(prop_moist_coefficients property/prop_moist_coefficients.cpp)
target_compile_definitions(prop_moist_coefficients PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_vert_implicit property/prop_vert_implicit.cpp)
mpas_add_property_test(prop_diagnostics property/prop_diagnostics.cpp)
target_compile_definitions(prop_diagnostics PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_tendencies property/prop_tendencies.cpp)
target_compile_definitions(prop_tendencies PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_rayleigh_damping property/prop_rayleigh_damping.cpp)
mpas_add_property_test(prop_acoustic property/prop_acoustic.cpp)
target_compile_definitions(prop_acoustic PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_scalars_disabled property/prop_scalars_disabled.cpp)
target_compile_definitions(prop_scalars_disabled PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_scalars property/prop_scalars.cpp)
target_compile_definitions(prop_scalars PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_dissipation property/prop_dissipation.cpp)
target_compile_definitions(prop_dissipation PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_boundaries property/prop_boundaries.cpp)
target_compile_definitions(prop_boundaries PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_iau property/prop_iau.cpp)
target_compile_definitions(prop_iau PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_velocity_reconstruct property/prop_velocity_reconstruct.cpp)
target_compile_definitions(prop_velocity_reconstruct PRIVATE MPAS_HAVE_RAPIDCHECK)
