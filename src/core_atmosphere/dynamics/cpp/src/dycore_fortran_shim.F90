!> @file dycore_fortran_shim.F90
!> @brief Fortran iso_c_binding interface to the C++ dycore C API.
!>
!> This module provides Fortran-callable interface blocks that match the C API
!> signatures in dycore_c_api.h.  The atmosphere core driver uses this module
!> to call dycore_init, dycore_timestep, and dycore_finalize without any
!> knowledge of C++ or Kokkos.
!>
!> Requirements: 13.1, 13.2, 13.5, 13.7, 13.8, 13.9

module mpas_dycore_interface
  use iso_c_binding, only : c_int, c_double, c_ptr, c_loc
  implicit none
  private

  public :: dycore_init
  public :: dycore_timestep
  public :: dycore_finalize

  interface

    !> Initialize the C++ dycore (Kokkos, field wrapping, mesh sync).
    !> Returns 0 on success, non-zero error code on failure.
    !>
    !> Error codes:
    !>   0 = Success
    !>   1 = Kokkos initialization failed
    !>   2 = Invalid dimensions (nVertLevels <= 0, nCells <= 0, etc.)
    !>   3 = Null pointer for a required array
    !>
    !> @param nCells           Number of cells (owned + halo)
    !> @param nEdges           Number of edges (owned + halo)
    !> @param nVertices        Number of vertices (owned + halo)
    !> @param nVertLevels      Number of vertical layers
    !> @param maxEdges         Maximum edges per cell
    !> @param num_scalars      Number of scalar tracers
    !> @param cellsOnEdge      Mesh connectivity (2, nEdges)
    !> @param edgesOnCell      Mesh connectivity (maxEdges, nCells)
    !> @param verticesOnEdge   Mesh connectivity (2, nEdges)
    !> @param nEdgesOnCell_ptr Per-cell edge count (nCells)
    !> @param dvEdge           Edge length dual (nEdges)
    !> @param dcEdge           Distance between cell centers (nEdges)
    !> @param areaCell         Cell area (nCells)
    !> @param zgrid            Height of layer interfaces (nVertLevels+1, nCells)
    !> @param zz               d(zeta)/dz metric (nVertLevels, nCells)
    !> @param fzm              Vertical interpolation weight (nVertLevels, nCells)
    !> @param fzp              Vertical interpolation weight (nVertLevels, nCells)
    !> @param u_tl1            Horizontal momentum TL1 (nVertLevels, nEdges)
    !> @param u_tl2            Horizontal momentum TL2 (nVertLevels, nEdges)
    !> @param w_tl1            Vertical velocity TL1 (nVertLevels+1, nCells)
    !> @param w_tl2            Vertical velocity TL2 (nVertLevels+1, nCells)
    !> @param theta_m_tl1      Coupled theta TL1 (nVertLevels, nCells)
    !> @param theta_m_tl2      Coupled theta TL2 (nVertLevels, nCells)
    !> @param rho_zz_tl1       Dry density TL1 (nVertLevels, nCells)
    !> @param rho_zz_tl2       Dry density TL2 (nVertLevels, nCells)
    !> @param scalars_tl1      Scalars TL1 (num_scalars*nVertLevels, nCells)
    !> @param scalars_tl2      Scalars TL2 (num_scalars*nVertLevels, nCells)
    !> @param time_integration_order RK order
    !> @param number_of_sub_steps    Acoustic substeps per stage
    !> @param dynamics_split_steps   Dynamics-transport splitting
    !> @param config_monotonic       Monotonic flag (0/1)
    !> @param config_scalar_advection Scalar advection enable (0/1)
    !> @param config_apply_lbcs      Regional mode (0/1)
    !> @param config_mix_full        Full-state mixing (0/1)
    !> @param config_iau             IAU enable (0/1)
    !> @param gpu_aware_comm         GPU-aware MPI (0/1)
    !> @param mpi_comm_fortran  Fortran MPI communicator handle
    function dycore_init( &
        nCells, nEdges, nVertices, nVertLevels, maxEdges, num_scalars, &
        nCellsSolve, nEdgesSolve, &
        cellsOnEdge, edgesOnCell, verticesOnEdge, nEdgesOnCell_ptr, &
        dvEdge, dcEdge, areaCell, zgrid, zz, fzm, fzp, &
        u_tl1, u_tl2, w_tl1, w_tl2, &
        theta_m_tl1, theta_m_tl2, rho_zz_tl1, rho_zz_tl2, &
        scalars_tl1, scalars_tl2, &
        time_integration_order, number_of_sub_steps, dynamics_split_steps, &
        config_monotonic, config_scalar_advection, config_apply_lbcs, &
        config_mix_full, config_iau, gpu_aware_comm, &
        config_smdiv, config_len_disp, config_apvm_upwinding, config_hollingsworth, &
        mpi_comm_fortran) &
        result(ierr) bind(C, name='dycore_init')
      import :: c_int, c_double
      integer(c_int) :: ierr
      integer(c_int), value, intent(in) :: nCells
      integer(c_int), value, intent(in) :: nEdges
      integer(c_int), value, intent(in) :: nVertices
      integer(c_int), value, intent(in) :: nVertLevels
      integer(c_int), value, intent(in) :: maxEdges
      integer(c_int), value, intent(in) :: num_scalars
      integer(c_int), value, intent(in) :: nCellsSolve
      integer(c_int), value, intent(in) :: nEdgesSolve
      integer(c_int), intent(in) :: cellsOnEdge(*)
      integer(c_int), intent(in) :: edgesOnCell(*)
      integer(c_int), intent(in) :: verticesOnEdge(*)
      integer(c_int), intent(in) :: nEdgesOnCell_ptr(*)
      real(c_double), intent(in) :: dvEdge(*)
      real(c_double), intent(in) :: dcEdge(*)
      real(c_double), intent(in) :: areaCell(*)
      real(c_double), intent(in) :: zgrid(*)
      real(c_double), intent(in) :: zz(*)
      real(c_double), intent(in) :: fzm(*)
      real(c_double), intent(in) :: fzp(*)
      real(c_double), intent(inout) :: u_tl1(*)
      real(c_double), intent(inout) :: u_tl2(*)
      real(c_double), intent(inout) :: w_tl1(*)
      real(c_double), intent(inout) :: w_tl2(*)
      real(c_double), intent(inout) :: theta_m_tl1(*)
      real(c_double), intent(inout) :: theta_m_tl2(*)
      real(c_double), intent(inout) :: rho_zz_tl1(*)
      real(c_double), intent(inout) :: rho_zz_tl2(*)
      real(c_double), intent(inout) :: scalars_tl1(*)
      real(c_double), intent(inout) :: scalars_tl2(*)
      integer(c_int), value, intent(in) :: time_integration_order
      integer(c_int), value, intent(in) :: number_of_sub_steps
      integer(c_int), value, intent(in) :: dynamics_split_steps
      integer(c_int), value, intent(in) :: config_monotonic
      integer(c_int), value, intent(in) :: config_scalar_advection
      integer(c_int), value, intent(in) :: config_apply_lbcs
      integer(c_int), value, intent(in) :: config_mix_full
      integer(c_int), value, intent(in) :: config_iau
      integer(c_int), value, intent(in) :: gpu_aware_comm
      real(c_double), value, intent(in) :: config_smdiv
      real(c_double), value, intent(in) :: config_len_disp
      real(c_double), value, intent(in) :: config_apvm_upwinding
      integer(c_int), value, intent(in) :: config_hollingsworth
      integer(c_int), value, intent(in) :: mpi_comm_fortran
    end function dycore_init

    !> Advance the dycore by one timestep.
    !>
    !> On entry: sync_to_device on all prognostic state DualView fields.
    !> On exit:  sync_to_host on all modified prognostic state fields.
    !>
    !> @param dt         Timestep size in seconds.
    !> @param itimestep  Integer timestep index.
    subroutine dycore_timestep(dt, itimestep) bind(C, name='dycore_timestep')
      import :: c_int, c_double
      real(c_double), value, intent(in) :: dt
      integer(c_int), value, intent(in) :: itimestep
    end subroutine dycore_timestep

    !> Finalize the C++ dycore and shut down Kokkos.
    subroutine dycore_finalize() bind(C, name='dycore_finalize')
    end subroutine dycore_finalize

  end interface

end module mpas_dycore_interface
