! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!

#ifdef MPAS_CPP_DYCORE

!> @file mpas_dycore_marshalling.F
!> @brief Marshalling layer between MPAS pool structures and the C++ dycore C API.
!>
!> This module extracts fields from MPAS pools, performs 1-based to 0-based
!> index remapping for connectivity arrays, and invokes the C++ dycore via
!> the iso_c_binding interface in mpas_dycore_interface.
!>
!> Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 1.2, 1.3, 1.4, 1.5, 1.6

module mpas_dycore_marshalling

   use mpas_derived_types, only : domain_type, mpas_pool_type, MPAS_LOG_CRIT
   use mpas_pool_routines, only : mpas_pool_get_subpool, mpas_pool_get_array, &
                                  mpas_pool_get_dimension, mpas_pool_get_config
   use mpas_kind_types, only : RKIND, StrKIND
   use mpas_log, only : mpas_log_write
   use mpas_dycore_interface, only : dycore_init, dycore_timestep, dycore_finalize

   implicit none
   private

   public :: mpas_dycore_call_init
   public :: mpas_dycore_call_timestep
   public :: mpas_dycore_call_finalize
   public :: mpas_cpp_get_pointer
   public :: mpas_cpp_get_int_pointer
   public :: global_domain

   type (domain_type), pointer, save :: global_domain => null()

   !-----------------------------------------------------------------------
   ! Module-level buffers for 0-based connectivity arrays.
   ! These persist from dycore_init until dycore_finalize.
   !-----------------------------------------------------------------------
   integer, dimension(:), allocatable :: cellsOnEdge_0based
   integer, dimension(:), allocatable :: edgesOnCell_0based
   integer, dimension(:), allocatable :: verticesOnEdge_0based
   integer, dimension(:), allocatable :: nEdgesOnCell_0based

   contains

   !***********************************************************************
   !> @brief Extract fields from MPAS pools and initialize the C++ dycore.
   !>
   !> Extracts mesh dimensions, connectivity, geometry, prognostic state,
   !> and configuration from the domain's pool structures. Connectivity
   !> arrays are copied into module-level buffers with 1-to-0 index
   !> remapping. Geometry and prognostic arrays are passed directly
   !> (zero-copy).
   !>
   !> @param[inout] domain  MPAS domain structure
   !***********************************************************************
   subroutine mpas_dycore_call_init(domain)

      implicit none

      type (domain_type), intent(inout), target :: domain

      ! Pool pointers
      type (mpas_pool_type), pointer :: mesh
      type (mpas_pool_type), pointer :: state
      type (mpas_pool_type), pointer :: block_configs

      ! Dimension scalars
      integer, pointer :: nCells_ptr, nEdges_ptr, nVertices_ptr
      integer, pointer :: nVertLevels_ptr, maxEdges_ptr, num_scalars_ptr
      integer, pointer :: nCellsSolve_ptr, nEdgesSolve_ptr
      integer :: nCells, nEdges, nVertices, nVertLevels, maxEdges_val, num_scalars
      integer :: nCellsSolve, nEdgesSolve

      ! Connectivity arrays (1-based from MPAS)
      integer, dimension(:,:), pointer :: cellsOnEdge_mpas
      integer, dimension(:,:), pointer :: edgesOnCell_mpas
      integer, dimension(:,:), pointer :: verticesOnEdge_mpas
      integer, dimension(:),   pointer :: nEdgesOnCell_mpas

      ! Geometry arrays (passed directly, zero-copy)
      real (kind=RKIND), dimension(:),   pointer :: dvEdge, dcEdge, areaCell
      real (kind=RKIND), dimension(:,:), pointer :: zgrid, zz
      real (kind=RKIND), dimension(:),   pointer :: fzm, fzp

      ! Prognostic state arrays (two time levels)
      real (kind=RKIND), dimension(:,:),  pointer :: u_tl1, u_tl2
      real (kind=RKIND), dimension(:,:),  pointer :: w_tl1, w_tl2
      real (kind=RKIND), dimension(:,:),  pointer :: theta_m_tl1, theta_m_tl2
      real (kind=RKIND), dimension(:,:),  pointer :: rho_zz_tl1, rho_zz_tl2
      real (kind=RKIND), dimension(:,:,:), pointer :: scalars_tl1, scalars_tl2

      ! Config variables
      integer, pointer :: config_time_integration_order_ptr
      integer, pointer :: config_number_of_sub_steps_ptr
      integer, pointer :: config_dynamics_split_steps_ptr
      logical, pointer :: config_monotonic_ptr
      logical, pointer :: config_scalar_advection_ptr
      logical, pointer :: config_apply_lbcs_ptr
      real (kind=RKIND), pointer :: config_smdiv_ptr
      real (kind=RKIND), pointer :: config_len_disp_ptr
      real (kind=RKIND), pointer :: config_apvm_upwinding_ptr
      logical, pointer :: config_mix_full_ptr
      logical, pointer :: config_gpu_aware_mpi_ptr
      character(len=StrKIND), pointer :: config_IAU_option_ptr

      ! Real config values for C API
      real (kind=RKIND) :: cfg_smdiv
      real (kind=RKIND) :: cfg_len_disp
      real (kind=RKIND) :: cfg_apvm_upwinding
      integer :: cfg_hollingsworth

      ! Integer config values for C API
      integer :: cfg_time_integration_order
      integer :: cfg_number_of_sub_steps
      integer :: cfg_dynamics_split_steps
      integer :: cfg_monotonic
      integer :: cfg_scalar_advection
      integer :: cfg_apply_lbcs
      integer :: cfg_mix_full
      integer :: cfg_iau
      integer :: cfg_gpu_aware_comm

      ! Return code from dycore_init
      integer :: ierr

      ! MPI communicator handle
      integer :: mpi_comm_handle

      ! Loop counter
      integer :: i

      global_domain => domain

      !--------------------------------------------------------------------
      ! Get sub-pools from domain
      !--------------------------------------------------------------------
      nullify(mesh)
      nullify(state)
      call mpas_pool_get_subpool(domain % blocklist % structs, 'mesh', mesh)
      call mpas_pool_get_subpool(domain % blocklist % structs, 'state', state)
      block_configs => domain % blocklist % configs

      ! Get the MPI communicator from the domain's dminfo structure
      mpi_comm_handle = domain % dminfo % comm

      !--------------------------------------------------------------------
      ! Extract dimensions
      !--------------------------------------------------------------------
      call mpas_pool_get_dimension(mesh, 'nCells', nCells_ptr)
      call mpas_pool_get_dimension(mesh, 'nEdges', nEdges_ptr)
      call mpas_pool_get_dimension(mesh, 'nVertices', nVertices_ptr)
      call mpas_pool_get_dimension(mesh, 'nVertLevels', nVertLevels_ptr)
      call mpas_pool_get_dimension(mesh, 'maxEdges', maxEdges_ptr)
      call mpas_pool_get_dimension(state, 'num_scalars', num_scalars_ptr)
      call mpas_pool_get_dimension(mesh, 'nCellsSolve', nCellsSolve_ptr)
      call mpas_pool_get_dimension(mesh, 'nEdgesSolve', nEdgesSolve_ptr)

      nCells = nCells_ptr
      nEdges = nEdges_ptr
      nVertices = nVertices_ptr
      nVertLevels = nVertLevels_ptr
      maxEdges_val = maxEdges_ptr
      num_scalars = num_scalars_ptr
      nCellsSolve = nCellsSolve_ptr
      nEdgesSolve = nEdgesSolve_ptr

      !--------------------------------------------------------------------
      ! Extract connectivity arrays (1-based) from mesh pool
      !--------------------------------------------------------------------
      call mpas_pool_get_array(mesh, 'cellsOnEdge', cellsOnEdge_mpas)
      call mpas_pool_get_array(mesh, 'edgesOnCell', edgesOnCell_mpas)
      call mpas_pool_get_array(mesh, 'verticesOnEdge', verticesOnEdge_mpas)
      call mpas_pool_get_array(mesh, 'nEdgesOnCell', nEdgesOnCell_mpas)

      !--------------------------------------------------------------------
      ! Allocate module-level buffers and perform 1-to-0 index remapping.
      ! Connectivity arrays store entity indices which are 1-based in
      ! Fortran and need to be 0-based for the C++ kernels.
      ! nEdgesOnCell stores counts (not indices) and is copied as-is.
      !--------------------------------------------------------------------

      ! IMPORTANT: connectivity indices are passed to the C++ dycore as the
      ! Fortran 1-based values VERBATIM. The C++ MeshData contract states the
      ! raw pointers are not renumbered here; every C++ kernel performs the
      ! 1-based -> 0-based conversion at the point of use (idx - 1). Subtracting
      ! 1 here as well would double-decrement and read the wrong cells/edges.
      ! The buffers are copied (not aliased) only to guarantee contiguous,
      ! persistent storage for the lifetime of the C++ context.

      ! cellsOnEdge: shape (2, nEdges) — contiguous column-major in memory
      allocate(cellsOnEdge_0based(2 * nEdges))
      do i = 1, 2 * nEdges
         cellsOnEdge_0based(i) = cellsOnEdge_mpas(1 + mod(i-1, 2), 1 + (i-1)/2)
      end do

      ! edgesOnCell: shape (maxEdges, nCells) — contiguous column-major
      allocate(edgesOnCell_0based(maxEdges_val * nCells))
      do i = 1, maxEdges_val * nCells
         edgesOnCell_0based(i) = edgesOnCell_mpas(1 + mod(i-1, maxEdges_val), 1 + (i-1)/maxEdges_val)
      end do

      ! verticesOnEdge: shape (2, nEdges) — contiguous column-major
      allocate(verticesOnEdge_0based(2 * nEdges))
      do i = 1, 2 * nEdges
         verticesOnEdge_0based(i) = verticesOnEdge_mpas(1 + mod(i-1, 2), 1 + (i-1)/2)
      end do

      ! nEdgesOnCell: shape (nCells) — per-cell edge count (not an index),
      ! copied into a contiguous buffer without remapping
      allocate(nEdgesOnCell_0based(nCells))
      do i = 1, nCells
         nEdgesOnCell_0based(i) = nEdgesOnCell_mpas(i)
      end do

      !--------------------------------------------------------------------
      ! Extract geometry arrays (passed directly, zero-copy)
      !--------------------------------------------------------------------
      call mpas_pool_get_array(mesh, 'dvEdge', dvEdge)
      call mpas_pool_get_array(mesh, 'dcEdge', dcEdge)
      call mpas_pool_get_array(mesh, 'areaCell', areaCell)
      call mpas_pool_get_array(mesh, 'zgrid', zgrid)
      call mpas_pool_get_array(mesh, 'zz', zz)
      call mpas_pool_get_array(mesh, 'fzm', fzm)
      call mpas_pool_get_array(mesh, 'fzp', fzp)

      !--------------------------------------------------------------------
      ! Extract prognostic state arrays (two time levels, zero-copy)
      !--------------------------------------------------------------------
      call mpas_pool_get_array(state, 'u', u_tl1, 1)
      call mpas_pool_get_array(state, 'u', u_tl2, 2)
      call mpas_pool_get_array(state, 'w', w_tl1, 1)
      call mpas_pool_get_array(state, 'w', w_tl2, 2)
      call mpas_pool_get_array(state, 'theta_m', theta_m_tl1, 1)
      call mpas_pool_get_array(state, 'theta_m', theta_m_tl2, 2)
      call mpas_pool_get_array(state, 'rho_zz', rho_zz_tl1, 1)
      call mpas_pool_get_array(state, 'rho_zz', rho_zz_tl2, 2)
      call mpas_pool_get_array(state, 'scalars', scalars_tl1, 1)
      call mpas_pool_get_array(state, 'scalars', scalars_tl2, 2)

      !--------------------------------------------------------------------
      ! Extract configuration scalars from block configs
      !--------------------------------------------------------------------
      call mpas_pool_get_config(block_configs, 'config_time_integration_order', config_time_integration_order_ptr)
      call mpas_pool_get_config(block_configs, 'config_number_of_sub_steps', config_number_of_sub_steps_ptr)
      call mpas_pool_get_config(block_configs, 'config_dynamics_split_steps', config_dynamics_split_steps_ptr)
      call mpas_pool_get_config(block_configs, 'config_monotonic', config_monotonic_ptr)
      call mpas_pool_get_config(block_configs, 'config_scalar_advection', config_scalar_advection_ptr)
      call mpas_pool_get_config(block_configs, 'config_apply_lbcs', config_apply_lbcs_ptr)
      call mpas_pool_get_config(block_configs, 'config_smdiv', config_smdiv_ptr)
      call mpas_pool_get_config(block_configs, 'config_len_disp', config_len_disp_ptr)
      call mpas_pool_get_config(block_configs, 'config_apvm_upwinding', config_apvm_upwinding_ptr)
      call mpas_pool_get_config(block_configs, 'config_mix_full', config_mix_full_ptr)
      call mpas_pool_get_config(block_configs, 'config_gpu_aware_mpi', config_gpu_aware_mpi_ptr)
      call mpas_pool_get_config(block_configs, 'config_IAU_option', config_IAU_option_ptr)

      ! Map config values to c_int for the C API
      cfg_time_integration_order = config_time_integration_order_ptr
      cfg_number_of_sub_steps = config_number_of_sub_steps_ptr
      cfg_dynamics_split_steps = config_dynamics_split_steps_ptr
      cfg_smdiv = config_smdiv_ptr
      cfg_len_disp = config_len_disp_ptr
      cfg_apvm_upwinding = config_apvm_upwinding_ptr
      cfg_hollingsworth = 1  ! parameter in reference model is always .true.

      ! Logical to integer mapping: .true. → 1, .false. → 0
      cfg_monotonic = 0
      if (config_monotonic_ptr) cfg_monotonic = 1

      cfg_scalar_advection = 0
      if (config_scalar_advection_ptr) cfg_scalar_advection = 1

      cfg_apply_lbcs = 0
      if (config_apply_lbcs_ptr) cfg_apply_lbcs = 1

      cfg_mix_full = 0
      if (config_mix_full_ptr) cfg_mix_full = 1

      cfg_gpu_aware_comm = 0
      if (config_gpu_aware_mpi_ptr) cfg_gpu_aware_comm = 1

      ! Map IAU option string to integer: 'off' → 0, anything else → 1
      cfg_iau = 0
      if (trim(config_IAU_option_ptr) /= 'off') cfg_iau = 1

      !--------------------------------------------------------------------
      ! Call the C++ dycore initialization and check return code (Req 8.2, 8.3)
      !--------------------------------------------------------------------
      ierr = dycore_init( &
         nCells, nEdges, nVertices, nVertLevels, maxEdges_val, num_scalars, &
         nCellsSolve, nEdgesSolve, &
         cellsOnEdge_0based, edgesOnCell_0based, verticesOnEdge_0based, nEdgesOnCell_0based, &
         dvEdge, dcEdge, areaCell, zgrid, zz, fzm, fzp, &
         u_tl1, u_tl2, w_tl1, w_tl2, &
         theta_m_tl1, theta_m_tl2, rho_zz_tl1, rho_zz_tl2, &
         scalars_tl1, scalars_tl2, &
         cfg_time_integration_order, cfg_number_of_sub_steps, cfg_dynamics_split_steps, &
         cfg_monotonic, cfg_scalar_advection, cfg_apply_lbcs, &
         cfg_mix_full, cfg_iau, cfg_gpu_aware_comm, &
         cfg_smdiv, cfg_len_disp, cfg_apvm_upwinding, cfg_hollingsworth, &
         mpi_comm_handle)

      if (ierr /= 0) then
         if (ierr == 1) then
            call mpas_log_write('C++ dycore initialization failed: Kokkos initialization error (code 1)', &
                                messageType=MPAS_LOG_CRIT)
         else if (ierr == 2) then
            call mpas_log_write('C++ dycore initialization failed: invalid dimensions (code 2)', &
                                messageType=MPAS_LOG_CRIT)
         else if (ierr == 3) then
            call mpas_log_write('C++ dycore initialization failed: null pointer for required array (code 3)', &
                                messageType=MPAS_LOG_CRIT)
         else
            call mpas_log_write('C++ dycore initialization failed: unknown error code $i', &
                                messageType=MPAS_LOG_CRIT, intArgs=(/ierr/))
         end if
      end if

   end subroutine mpas_dycore_call_init


   !***********************************************************************
   !> @brief Advance the C++ dycore by one timestep.
   !>
   !> Passes dt and itimestep directly through to dycore_timestep.
   !>
   !> @param[in] dt         Timestep size in seconds.
   !> @param[in] itimestep  Integer timestep index.
   !***********************************************************************
   subroutine mpas_dycore_call_timestep(dt, itimestep)

      use mpas_kind_types, only : RKIND

      implicit none

      real (kind=RKIND), intent(in) :: dt
      integer, intent(in) :: itimestep

      call dycore_timestep(dt, itimestep)

   end subroutine mpas_dycore_call_timestep


   !***********************************************************************
   !> @brief Finalize the C++ dycore and release connectivity buffers.
   !>
   !> Calls dycore_finalize to shut down Kokkos, then deallocates the
   !> module-level connectivity buffers that were allocated during init.
   !***********************************************************************
   subroutine mpas_dycore_call_finalize()

      implicit none

      call dycore_finalize()

      ! Deallocate connectivity buffers
      if (allocated(cellsOnEdge_0based))    deallocate(cellsOnEdge_0based)
      if (allocated(edgesOnCell_0based))     deallocate(edgesOnCell_0based)
      if (allocated(verticesOnEdge_0based))  deallocate(verticesOnEdge_0based)
      if (allocated(nEdgesOnCell_0based))    deallocate(nEdgesOnCell_0based)

      global_domain => null()

   end subroutine mpas_dycore_call_finalize


   !***********************************************************************
   !> @brief Dynamically look up and return a C pointer to variables in MPAS pools.
   !>
   !> Resolves 1D, 2D, and 3D variables from subpools dynamically using
   !> the global_domain cached during initialization.
   !***********************************************************************
   function mpas_cpp_get_pointer(pool_name_c, var_name_c, dim_num, time_level) &
       result(cptr) bind(C, name='mpas_cpp_get_pointer')
     use iso_c_binding, only : c_char, c_int, c_ptr, c_null_ptr, c_loc
     use mpas_pool_routines, only : mpas_pool_get_subpool, mpas_pool_get_array
     use mpas_c_interfacing, only : mpas_c_to_f_string
     implicit none

     character(kind=c_char), intent(in) :: pool_name_c(*)
     character(kind=c_char), intent(in) :: var_name_c(*)
     integer(c_int), value :: dim_num
     integer(c_int), value :: time_level
     type(c_ptr) :: cptr

     type(mpas_pool_type), pointer :: pool
     character(len=StrKIND) :: pool_name, var_name
     real(kind=RKIND), dimension(:), pointer :: arr1d
     real(kind=RKIND), dimension(:,:), pointer :: arr2d
     real(kind=RKIND), dimension(:,:,:), pointer :: arr3d

     cptr = c_null_ptr
     if (.not. associated(global_domain)) return

     call mpas_c_to_f_string(pool_name_c, pool_name)
     call mpas_c_to_f_string(var_name_c, var_name)

     nullify(pool)
     call mpas_pool_get_subpool(global_domain % blocklist % structs, trim(pool_name), pool)
     if (.not. associated(pool)) return

     if (dim_num == 1) then
        if (time_level > 0) then
           call mpas_pool_get_array(pool, trim(var_name), arr1d, time_level)
        else
           call mpas_pool_get_array(pool, trim(var_name), arr1d)
        end if
        if (associated(arr1d)) cptr = c_loc(arr1d(1))
     else if (dim_num == 2) then
        if (time_level > 0) then
           call mpas_pool_get_array(pool, trim(var_name), arr2d, time_level)
        else
           call mpas_pool_get_array(pool, trim(var_name), arr2d)
        end if
        if (associated(arr2d)) cptr = c_loc(arr2d(1,1))
     else if (dim_num == 3) then
        if (time_level > 0) then
           call mpas_pool_get_array(pool, trim(var_name), arr3d, time_level)
        else
           call mpas_pool_get_array(pool, trim(var_name), arr3d)
        end if
        if (associated(arr3d)) cptr = c_loc(arr3d(1,1,1))
     end if
   end function mpas_cpp_get_pointer


   !***********************************************************************
   !> @brief FFI lookup bridge to dynamically query integer array pointers.
   !***********************************************************************
   function mpas_cpp_get_int_pointer(pool_name_c, var_name_c, dim_num, time_level) result(cptr) bind(C, name='mpas_cpp_get_int_pointer')
      use iso_c_binding
      use mpas_c_interfacing, only : mpas_c_to_f_string
      use mpas_pool_routines, only : mpas_pool_get_subpool, mpas_pool_get_array
      implicit none
      character(kind=c_char), intent(in) :: pool_name_c(*)
      character(kind=c_char), intent(in) :: var_name_c(*)
      integer(c_int), value :: dim_num
      integer(c_int), value :: time_level
      type(c_ptr) :: cptr

      type(mpas_pool_type), pointer :: pool
      integer, dimension(:), pointer :: arr1d
      integer, dimension(:,:), pointer :: arr2d
      integer, dimension(:,:,:), pointer :: arr3d
      character(len=StrKIND) :: pool_name, var_name

      cptr = c_null_ptr
      if (.not. associated(global_domain)) return

      call mpas_c_to_f_string(pool_name_c, pool_name)
      call mpas_c_to_f_string(var_name_c, var_name)

      call mpas_pool_get_subpool(global_domain % blocklist % structs, trim(pool_name), pool)
      if (.not. associated(pool)) return

      if (dim_num == 1) then
         nullify(arr1d)
         if (time_level > 0) then
            call mpas_pool_get_array(pool, trim(var_name), arr1d, time_level)
         else
            call mpas_pool_get_array(pool, trim(var_name), arr1d)
         end if
         if (associated(arr1d)) cptr = c_loc(arr1d(1))
      else if (dim_num == 2) then
         nullify(arr2d)
         if (time_level > 0) then
            call mpas_pool_get_array(pool, trim(var_name), arr2d, time_level)
         else
            call mpas_pool_get_array(pool, trim(var_name), arr2d)
         end if
         if (associated(arr2d)) cptr = c_loc(arr2d(1,1))
      else if (dim_num == 3) then
         nullify(arr3d)
         if (time_level > 0) then
            call mpas_pool_get_array(pool, trim(var_name), arr3d, time_level)
         else
            call mpas_pool_get_array(pool, trim(var_name), arr3d)
         end if
         if (associated(arr3d)) cptr = c_loc(arr3d(1,1,1))
      end if
   end function mpas_cpp_get_int_pointer

   end module mpas_dycore_marshalling

#endif /* MPAS_CPP_DYCORE */
