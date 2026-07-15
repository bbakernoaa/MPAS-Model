#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# validate_cpp_dycore.sh
#
# Jablonowski-Williamson Baroclinic Wave Comparison Test
#
# Runs 120 timesteps (1 simulated day) at 240km/55 levels with both the
# Fortran and C++ dycores, then compares prognostic field outputs for parity.
#
# Expects to run inside the mpas-fortran-integration Docker image where
# `atmosphere_model` is already built with USE_CPP_DYCORE=true.
#
# Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPAS_ROOT="${MPAS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ATMOSPHERE_MODEL="${MPAS_ROOT}/atmosphere_model"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_dycore_outputs.py"

# Test case parameters (240km quasi-uniform, 55 levels, 720s dt, 120 steps)
MESH_RES="240km"
N_VERT_LEVELS=55
DT=720.0
N_TIMESTEPS=120
PARITY_TOLERANCE="1.0e-13"

# Working directories
WORKDIR="${WORKDIR:-/tmp/mpas_jw_validation}"
REF_DIR="${WORKDIR}/reference"
CPP_DIR="${WORKDIR}/cpp_run"

# Input data (mesh and initial conditions for JW test)
# These should be available in the Docker image or mounted as a volume.
JW_MESH="${JW_MESH:-/work/test_data/jw_baroclinic_240km/x1.40962.init.nc}"
JW_GRAPH="${JW_GRAPH:-/work/test_data/jw_baroclinic_240km/x1.40962.graph.info.part.4}"

# Number of MPI ranks
NPROCS="${NPROCS:-4}"

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

die() {
    log_error "$@"
    exit 1
}

# Generate the namelist.atmosphere for the JW test case
generate_namelist() {
    local use_cpp_dycore="$1"
    local output_dir="$2"

    cat > "${output_dir}/namelist.atmosphere" << EOF
&nhyd_model
    config_init_case = 7
    config_start_time = '0000-01-01_00:00:00'
    config_run_duration = '1_00:00:00'
    config_dt = ${DT}
    config_time_integration_order = 2
    config_dynamics_split_steps = 3
    config_number_of_sub_steps = 2
    config_h_ScaleWithMesh = .true.
    config_use_cpp_dycore = ${use_cpp_dycore}
/
&damping
    config_zd = 22000.0
    config_xnutr = 0.2
/
&io
    config_pio_num_iotasks = 0
    config_pio_stride = 1
/
&decomposition
    config_block_decomp_file_prefix = '${output_dir}/x1.40962.graph.info.part.'
/
&restart
    config_do_restart = .false.
/
&physics
    config_physics_suite = 'none'
/
&soundings
    config_sounding_interval = 'none'
/
EOF
}

# Generate the streams.atmosphere file
generate_streams() {
    local output_dir="$1"

    cat > "${output_dir}/streams.atmosphere" << EOF
<streams>
<immutable_stream name="input"
                  type="input"
                  filename_template="${output_dir}/init.nc"
                  input_interval="initial_only" />

<stream name="output"
        type="output"
        filename_template="${output_dir}/output.nc"
        output_interval="1_00:00:00"
        clobber_mode="overwrite">
    <var name="u"/>
    <var name="w"/>
    <var name="theta_m"/>
    <var name="rho_zz"/>
    <var name="scalars"/>
    <var name="xtime"/>
</stream>
</streams>
EOF
}

# Run the MPAS atmosphere model and return wall-clock time
run_mpas() {
    local run_dir="$1"
    local label="$2"

    log_info "Running MPAS (${label}) in ${run_dir}..."

    local start_time end_time elapsed

    start_time=$(date +%s%N)
    (
        cd "${run_dir}"
        mpirun --allow-run-as-root -np "${NPROCS}" "${ATMOSPHERE_MODEL}" 2>&1 | \
            tee "${run_dir}/mpas.log"
    )
    local exit_code=${PIPESTATUS[0]}
    end_time=$(date +%s%N)

    if [ "$exit_code" -ne 0 ]; then
        die "MPAS run (${label}) failed with exit code ${exit_code}. See ${run_dir}/mpas.log"
    fi

    # Compute elapsed time in seconds (nanosecond precision)
    elapsed=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc)
    echo "${elapsed}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

main() {
    log_info "=========================================="
    log_info "MPAS C++ Dycore Validation Test"
    log_info "  Mesh:       ${MESH_RES} quasi-uniform"
    log_info "  Levels:     ${N_VERT_LEVELS}"
    log_info "  Timestep:   ${DT}s"
    log_info "  Steps:      ${N_TIMESTEPS} (1 simulated day)"
    log_info "  Tolerance:  ${PARITY_TOLERANCE}"
    log_info "  MPI ranks:  ${NPROCS}"
    log_info "=========================================="

    # Verify prerequisites
    [ -x "${ATMOSPHERE_MODEL}" ] || die "atmosphere_model not found at ${ATMOSPHERE_MODEL}"
    [ -f "${JW_MESH}" ] || die "JW mesh/init file not found at ${JW_MESH}"
    [ -f "${COMPARE_SCRIPT}" ] || die "Comparison script not found at ${COMPARE_SCRIPT}"

    # Set up working directories
    log_info "Setting up working directories..."
    rm -rf "${WORKDIR}"
    mkdir -p "${REF_DIR}" "${CPP_DIR}"

    # Link/copy mesh and initial condition files
    cp "${JW_MESH}" "${REF_DIR}/init.nc"
    cp "${JW_MESH}" "${CPP_DIR}/init.nc"

    # Copy graph partition files
    for part_file in "${JW_GRAPH}"*; do
        if [ -f "$part_file" ]; then
            cp "$part_file" "${REF_DIR}/"
            cp "$part_file" "${CPP_DIR}/"
        fi
    done

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Run with Fortran dycore (reference)
    # ─────────────────────────────────────────────────────────────────────────
    log_info "Step 1: Generating namelist for Fortran reference run..."
    generate_namelist ".false." "${REF_DIR}"
    generate_streams "${REF_DIR}"

    ref_time=$(run_mpas "${REF_DIR}" "Fortran reference")
    log_info "Fortran reference run completed in ${ref_time}s"

    # Verify output was produced
    [ -f "${REF_DIR}/output.nc" ] || die "Reference output not produced: ${REF_DIR}/output.nc"

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Run with C++ dycore
    # ─────────────────────────────────────────────────────────────────────────
    log_info "Step 2: Generating namelist for C++ dycore run..."
    generate_namelist ".true." "${CPP_DIR}"
    generate_streams "${CPP_DIR}"

    cpp_time=$(run_mpas "${CPP_DIR}" "C++ dycore")
    log_info "C++ dycore run completed in ${cpp_time}s"

    # Verify output was produced
    [ -f "${CPP_DIR}/output.nc" ] || die "C++ dycore output not produced: ${CPP_DIR}/output.nc"

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Compare outputs
    # ─────────────────────────────────────────────────────────────────────────
    log_info "Step 3: Comparing outputs..."

    local compare_result
    compare_result=$(python3 "${COMPARE_SCRIPT}" \
        --reference "${REF_DIR}/output.nc" \
        --test "${CPP_DIR}/output.nc" \
        --tolerance "${PARITY_TOLERANCE}" \
        --fields u w theta_m rho_zz scalars)
    local compare_exit=$?

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Report results
    # ─────────────────────────────────────────────────────────────────────────
    echo ""
    log_info "=========================================="
    log_info "RESULTS"
    log_info "=========================================="
    echo ""

    # Timing report
    local ref_per_step cpp_per_step
    ref_per_step=$(echo "scale=4; ${ref_time} / ${N_TIMESTEPS}" | bc)
    cpp_per_step=$(echo "scale=4; ${cpp_time} / ${N_TIMESTEPS}" | bc)

    echo "  Wall-clock timing:"
    echo "    Fortran dycore:  ${ref_time}s total, ${ref_per_step}s/timestep"
    echo "    C++ dycore:      ${cpp_time}s total, ${cpp_per_step}s/timestep"
    echo ""

    # Comparison results
    echo "${compare_result}"
    echo ""

    if [ "${compare_exit}" -eq 0 ]; then
        log_info "=========================================="
        log_info "PASS - All prognostic fields within parity tolerance (${PARITY_TOLERANCE})"
        log_info "=========================================="
        exit 0
    else
        log_error "=========================================="
        log_error "FAIL - Field parity check failed"
        log_error "=========================================="
        exit 1
    fi
}

main "$@"
