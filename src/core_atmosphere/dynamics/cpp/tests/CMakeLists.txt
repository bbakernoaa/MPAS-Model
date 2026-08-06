# Test targets for mpas_dycore_cpp
# Unit tests and property-based tests are added here as they are implemented.

# Common test utilities
add_library(test_utils STATIC planar_hex_mesh.cpp)
target_link_libraries(test_utils PUBLIC mpas_dycore_cpp)
target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

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
mpas_add_property_test(prop_srk3_config property/prop_srk3_config.cpp)
target_compile_definitions(prop_srk3_config PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_geometry property/prop_geometry.cpp)
target_compile_definitions(prop_geometry PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_workspace property/prop_workspace.cpp)
target_compile_definitions(prop_workspace PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_recover_state property/prop_recover_state.cpp)
target_compile_definitions(prop_recover_state PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_recover_state_validation property/prop_recover_state_validation.cpp)
target_compile_definitions(prop_recover_state_validation PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_recover_state_bugcondition property/prop_recover_state_bugcondition.cpp)
target_compile_definitions(prop_recover_state_bugcondition PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_recover_state_preservation property/prop_recover_state_preservation.cpp)
target_compile_definitions(prop_recover_state_preservation PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_api_geometry property/prop_api_geometry.cpp)
target_compile_definitions(prop_api_geometry PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_stability property/prop_stability.cpp)
target_link_libraries(prop_stability PRIVATE test_utils)
target_compile_definitions(prop_stability PRIVATE MPAS_HAVE_RAPIDCHECK)
mpas_add_property_test(prop_mass_conservation property/prop_mass_conservation.cpp)
target_link_libraries(prop_mass_conservation PRIVATE test_utils)
target_compile_definitions(prop_mass_conservation PRIVATE MPAS_HAVE_RAPIDCHECK)

# Idealized test driver
add_executable(test_idealized_driver test_idealized_driver.cpp)
target_link_libraries(test_idealized_driver PRIVATE
    test_utils
    mpas_dycore_cpp
    MPI::MPI_CXX
)
target_include_directories(test_idealized_driver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_idealized_driver COMMAND test_idealized_driver)
set_tests_properties(test_idealized_driver PROPERTIES TIMEOUT 60)

# Mass error investigation (diagnostic, not a pass/fail test)
add_executable(test_mass_error_investigation test_mass_error_investigation.cpp)
target_link_libraries(test_mass_error_investigation PRIVATE
    test_utils
    mpas_dycore_cpp
    MPI::MPI_CXX
)
target_include_directories(test_mass_error_investigation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_mass_error_investigation COMMAND test_mass_error_investigation)
set_tests_properties(test_mass_error_investigation PROPERTIES TIMEOUT 120)
