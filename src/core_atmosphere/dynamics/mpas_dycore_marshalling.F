! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  mpas_dycore_marshalling
!
!> \brief Fortran-to-C++ marshalling layer for the C++ dynamical core
!> \details
!>  Extracts arrays from MPAS pool data structures, converts 1-based
!>  connectivity indices to 0-based for the C API, and invokes the C++
!>  dynamical core through the ISO_C_BINDING interface module.
!>
!>  Requirements: 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
!
!-----------------------------------------------------------------------

module mpas_dycore_marshalling

   use mpas_kind_types, only : RKIND, StrKIND
   use mpas_derived_types, only : domain_type, block_type, mpas_pool_type, &
                                  MPAS_LOG_CRIT, MPAS_LOG_ERR
   use mpas_pool_routines, only : mpas_pool_get_array, mpas_pool_get_dimension, &
                                  mpas_pool_get_subpool, mpas_pool_get_config
   use mpas_log, only : mpas_log_write
   use mpas_dmpar
   use mpas_dycore_cpp_interface
   use iso_c_binding, only : c_int, c_double, c_char, c_null_char

   implicit none

   private
   public :: mpas_cpp_dycore_init, &
             mpas_cpp_dycore_set_geometry, &
             mpas_cpp_dycore_timestep, &
             mpas_cpp_dycore_finalize

   logical, save :: cpp_dycore_initialized = .false.

contains


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine mpas_cpp_dycore_init
!
!> \brief Initialize the C++ dynamical core
!> \details
!>  Extracts mesh connectivity arrays from the MPAS mesh pool,
!>  converts from 1-based to 0-based indexing, extracts halo
!>  communication descriptors, reads dynamics configuration
!>  parameters, and calls mpas_dycore_cpp_init.
!>
!>  Requirements: 1.4, 2.1, 2.6, 2.7
!
!-----------------------------------------------------------------------
   subroutine mpas_cpp_dycore_init(domain, ierr)

      implicit none

      type (domain_type), intent(inout) :: domain
      integer, intent(out) :: ierr

      ! Local variables
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: meshPool
      type (mpas_pool_type), pointer :: configs

      ! Mesh dimensions
      integer, pointer :: nCells_ptr, nEdges_ptr, nVertices_ptr
      integer, pointer :: nVertLevels_ptr, maxEdges_ptr
      integer :: nCells, nEdges, nVertices, nVertLevels, maxEdges

      ! Connectivity arrays from mesh pool (Fortran 1-based)
      integer, dimension(:,:), pointer :: cellsOnEdge       ! (2, nEdges)
      integer, dimension(:,:), pointer :: verticesOnEdge    ! (2, nEdges)
      integer, dimension(:,:), pointer :: edgesOnCell       ! (maxEdges, nCells)
      integer, dimension(:,:), pointer :: cellsOnCell       ! (maxEdges, nCells)
      integer, dimension(:), pointer   :: nEdgesOnCell      ! (nCells)

      ! 0-based connectivity copies for C API
      integer(c_int), dimension(:), allocatable :: cellsOnEdge_0
      integer(c_int), dimension(:), allocatable :: verticesOnEdge_0
      integer(c_int), dimension(:), allocatable :: edgesOnCell_0
      integer(c_int), dimension(:), allocatable :: cellsOnCell_0
      integer(c_int), dimension(:), allocatable :: nEdgesOnCell_c

      ! Halo exchange descriptors (placeholder)
      integer(c_int), dimension(1) :: halo_ranks_empty
      integer(c_int), dimension(1) :: halo_counts_empty
      integer(c_int), dimension(1) :: halo_send_indices_empty
      integer(c_int), dimension(1) :: halo_recv_indices_empty

      ! Configuration parameters
      real (kind=RKIND), pointer :: config_dt
      integer, pointer :: config_number_of_sub_steps
      integer, pointer :: config_num_halos
      logical, pointer :: config_apply_lbcs_ptr
      character (len=StrKIND), pointer :: config_horiz_mixing
      real (kind=RKIND), pointer :: config_h_mom_eddy_visc2
      real (kind=RKIND), pointer :: config_h_mom_eddy_visc4
      real (kind=RKIND), pointer :: config_h_theta_eddy_visc2

      ! C API arguments
      integer(c_int) :: apply_lbcs_int
      character(c_char) :: horiz_mixing_c(256)
      character(c_char) :: errmsg(256)
      integer(c_int) :: rc
      integer :: i, j, idx, slen

      !---------------------------------------------------------------------

      ierr = 0

      ! Guard against double initialization
      if (cpp_dycore_initialized) then
         call mpas_log_write('C++ dycore already initialized', messageType=MPAS_LOG_ERR)
         ierr = 1
         return
      end if

      block => domain % blocklist

      ! Get pool handles
      nullify(meshPool)
      nullify(configs)
      call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
      configs => block % configs

      !---------------------------------------------------------------------
      ! Retrieve mesh dimensions
      !---------------------------------------------------------------------
      call mpas_pool_get_dimension(meshPool, 'nCells', nCells_ptr)
      call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges_ptr)
      call mpas_pool_get_dimension(meshPool, 'nVertices', nVertices_ptr)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels_ptr)
      call mpas_pool_get_dimension(meshPool, 'maxEdges', maxEdges_ptr)

      nCells = nCells_ptr
      nEdges = nEdges_ptr
      nVertices = nVertices_ptr
      nVertLevels = nVertLevels_ptr
      maxEdges = maxEdges_ptr

      !---------------------------------------------------------------------
      ! Retrieve connectivity arrays (Fortran 1-based)
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'verticesOnEdge', verticesOnEdge)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'cellsOnCell', cellsOnCell)
      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)

      !---------------------------------------------------------------------
      ! Allocate 0-based connectivity copies and convert indices
      ! Connectivity arrays in MPAS are stored as:
      !   cellsOnEdge(2, nEdges), verticesOnEdge(2, nEdges)
      !   edgesOnCell(maxEdges, nCells), cellsOnCell(maxEdges, nCells)
      ! The C API expects flat arrays with explicit dimensions.
      !---------------------------------------------------------------------
      allocate(cellsOnEdge_0(2 * nEdges))
      allocate(verticesOnEdge_0(2 * nEdges))
      allocate(edgesOnCell_0(maxEdges * nCells))
      allocate(cellsOnCell_0(maxEdges * nCells))
      allocate(nEdgesOnCell_c(nCells))

      ! Convert cellsOnEdge: (2, nEdges) -> flat, subtract 1
      idx = 1
      do j = 1, nEdges
         do i = 1, 2
            cellsOnEdge_0(idx) = int(cellsOnEdge(i, j) - 1, c_int)
            idx = idx + 1
         end do
      end do

      ! Convert verticesOnEdge: (2, nEdges) -> flat, subtract 1
      idx = 1
      do j = 1, nEdges
         do i = 1, 2
            verticesOnEdge_0(idx) = int(verticesOnEdge(i, j) - 1, c_int)
            idx = idx + 1
         end do
      end do

      ! Convert edgesOnCell: (maxEdges, nCells) -> flat, subtract 1
      idx = 1
      do j = 1, nCells
         do i = 1, maxEdges
            edgesOnCell_0(idx) = int(edgesOnCell(i, j) - 1, c_int)
            idx = idx + 1
         end do
      end do

      ! Convert cellsOnCell: (maxEdges, nCells) -> flat, subtract 1
      idx = 1
      do j = 1, nCells
         do i = 1, maxEdges
            cellsOnCell_0(idx) = int(cellsOnCell(i, j) - 1, c_int)
            idx = idx + 1
         end do
      end do

      ! Copy nEdgesOnCell (no index conversion needed, just integer kind)
      do i = 1, nCells
         nEdgesOnCell_c(i) = int(nEdgesOnCell(i), c_int)
      end do

      !---------------------------------------------------------------------
      ! Halo exchange descriptors
      !
      ! The CSR halo descriptors are NOT populated here because the Fortran
      ! callback path handles all halo exchanges (halo_exchange_callback in
      ! mpas_dycore_cpp_callbacks.F). The C++ native halo exchange path
      ! (which would use these descriptors) is disabled by registering the
      ! Fortran callback via mpas_dycore_cpp_set_callbacks.
      !
      ! To enable C++ native halo in the future:
      !   1. Extract exchange lists from a cell field (e.g., indexToCellID)
      !      using mpas_dmpar_build_comm_lists
      !   2. Flatten into CSR arrays: ranks, counts, send_indices, recv_indices
      !   3. Pass c_null_funptr for the halo callback in mpas_cpp_callbacks_init
      !   See core_landice/mode_forward/mpas_li_velocity_external.F for the
      !   reference implementation of exchange list -> flat array conversion.
      !---------------------------------------------------------------------
      halo_ranks_empty(1) = 0
      halo_counts_empty(1) = 0
      halo_send_indices_empty(1) = 0
      halo_recv_indices_empty(1) = 0

      !---------------------------------------------------------------------
      ! Read dynamics configuration parameters
      !---------------------------------------------------------------------
      call mpas_pool_get_config(configs, 'config_dt', config_dt)
      call mpas_pool_get_config(configs, 'config_number_of_sub_steps', config_number_of_sub_steps)
      call mpas_pool_get_config(configs, 'config_num_halos', config_num_halos)
      call mpas_pool_get_config(configs, 'config_apply_lbcs', config_apply_lbcs_ptr)
      call mpas_pool_get_config(configs, 'config_horiz_mixing', config_horiz_mixing)
      call mpas_pool_get_config(configs, 'config_h_mom_eddy_visc2', config_h_mom_eddy_visc2)
      call mpas_pool_get_config(configs, 'config_h_mom_eddy_visc4', config_h_mom_eddy_visc4)
      call mpas_pool_get_config(configs, 'config_h_theta_eddy_visc2', config_h_theta_eddy_visc2)

      ! Convert logical to integer for C API
      if (config_apply_lbcs_ptr) then
         apply_lbcs_int = 1_c_int
      else
         apply_lbcs_int = 0_c_int
      end if

      ! Convert config_horiz_mixing string to C null-terminated char array
      horiz_mixing_c = c_null_char
      slen = len_trim(config_horiz_mixing)
      if (slen > 255) slen = 255
      do i = 1, slen
         horiz_mixing_c(i) = config_horiz_mixing(i:i)
      end do
      horiz_mixing_c(slen + 1) = c_null_char

      ! Clear error message buffer
      errmsg = c_null_char

      !---------------------------------------------------------------------
      ! Call C++ dycore initialization
      !---------------------------------------------------------------------
      rc = mpas_dycore_cpp_init( &
         int(nCells, c_int), &
         int(nEdges, c_int), &
         int(nVertices, c_int), &
         int(maxEdges, c_int), &
         int(nVertLevels, c_int), &
         cellsOnEdge_0, int(2, c_int), int(nEdges, c_int), &
         verticesOnEdge_0, int(2, c_int), int(nEdges, c_int), &
         edgesOnCell_0, int(maxEdges, c_int), int(nCells, c_int), &
         cellsOnCell_0, int(maxEdges, c_int), int(nCells, c_int), &
         nEdgesOnCell_c, int(nCells, c_int), &
         halo_ranks_empty, int(0, c_int), &
         halo_counts_empty, int(0, c_int), &
         halo_send_indices_empty, int(0, c_int), &
         halo_recv_indices_empty, int(0, c_int), &
         int(config_num_halos, c_int), &
         real(config_dt, c_double), &
         int(config_number_of_sub_steps, c_int), &
         apply_lbcs_int, &
         horiz_mixing_c, &
         real(config_h_mom_eddy_visc2, c_double), &
         real(config_h_mom_eddy_visc4, c_double), &
         real(config_h_theta_eddy_visc2, c_double), &
#ifdef MPAS_USE_MPI_F08
         domain % dminfo % comm % MPI_VAL, &
#else
         int(domain % dminfo % comm, c_int), &
#endif
         errmsg, int(256, c_int))

      ! Check return code
      if (rc /= 0) then
         call mpas_log_write('C++ dycore initialization failed: ' // &
                             trim(c_errmsg_to_fortran(errmsg)), &
                             messageType=MPAS_LOG_CRIT)
      end if

      ! Deallocate temporary 0-based copies
      deallocate(cellsOnEdge_0)
      deallocate(verticesOnEdge_0)
      deallocate(edgesOnCell_0)
      deallocate(cellsOnCell_0)
      deallocate(nEdgesOnCell_c)

      cpp_dycore_initialized = .true.
      ierr = 0

   end subroutine mpas_cpp_dycore_init


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine mpas_cpp_dycore_set_geometry
!
!> \brief Pass geometry/metric arrays to the C++ dycore
!> \details
!>  Extracts geometry arrays from the mesh pool and base-state profiles
!>  from the diag pool, then passes them to mpas_dycore_cpp_set_geometry.
!>  Real arrays are passed directly (no copy needed — contiguous Fortran
!>  memory matches the C API expectation). Integer connectivity arrays
!>  (edgesOnEdge, advCellsForEdge) require 0-based copies.
!>
!>  Requirements: 2.2
!
!-----------------------------------------------------------------------
   subroutine mpas_cpp_dycore_set_geometry(domain, ierr)

      implicit none

      type (domain_type), intent(inout) :: domain
      integer, intent(out) :: ierr

      ! Local variables
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: meshPool
      type (mpas_pool_type), pointer :: diagPool

      ! Mesh dimensions
      integer, pointer :: nCells_ptr, nEdges_ptr, nVertices_ptr
      integer, pointer :: nVertLevels_ptr, maxEdges_ptr, maxEdges2_ptr
      integer :: nCells, nEdges, nVertices, nVertLevels, maxEdges_dim, maxEdges2
      integer :: maxAdvCells

      ! Cell geometry
      real (kind=RKIND), dimension(:), pointer :: areaCell
      real (kind=RKIND), dimension(:), pointer :: invAreaCell

      ! Edge geometry
      real (kind=RKIND), dimension(:), pointer :: dvEdge
      real (kind=RKIND), dimension(:), pointer :: dcEdge
      real (kind=RKIND), dimension(:), pointer :: invDcEdge

      ! Vertical metrics
      real (kind=RKIND), dimension(:), pointer :: rdzw
      real (kind=RKIND), dimension(:), pointer :: rdzu
      real (kind=RKIND), dimension(:), pointer :: fzm
      real (kind=RKIND), dimension(:), pointer :: fzp
      real (kind=RKIND), dimension(:), pointer :: etp
      real (kind=RKIND), dimension(:), pointer :: etm
      real (kind=RKIND), dimension(:), pointer :: ewp
      real (kind=RKIND), dimension(:), pointer :: ewm

      ! Terrain metric
      real (kind=RKIND), dimension(:,:), pointer :: zz

      ! Base-state profiles (from mesh pool: rb, rtb, pb are really
      ! rho_zz_base * zz, rtheta_base, pressure_base in the mesh pool)
      ! Actually, from diag pool: rho_base, rtheta_base, exner_base
      real (kind=RKIND), dimension(:,:), pointer :: rho_base
      real (kind=RKIND), dimension(:,:), pointer :: rtheta_base
      real (kind=RKIND), dimension(:,:), pointer :: exner_base

      ! Edge orientation signs
      real (kind=RKIND), dimension(:,:), pointer :: edgesOnCell_sign

      ! Specified zone masks
      real (kind=RKIND), dimension(:), pointer :: specZoneMaskEdge
      real (kind=RKIND), dimension(:), pointer :: specZoneMaskCell

      ! TRiSK reconstruction weights
      real (kind=RKIND), dimension(:,:), pointer :: weightsOnEdge
      integer, dimension(:), pointer :: nEdgesOnEdge
      integer, dimension(:,:), pointer :: edgesOnEdge

      ! Advection stencils
      integer, dimension(:,:), pointer :: advCellsForEdge
      integer, dimension(:), pointer :: nAdvCellsForEdge
      real (kind=RKIND), dimension(:,:), pointer :: adv_coefs
      real (kind=RKIND), dimension(:,:), pointer :: adv_coefs_3rd

      ! Vertex geometry
      real (kind=RKIND), dimension(:), pointer :: fVertex
      real (kind=RKIND), dimension(:), pointer :: areaTriangle

      ! Terrain correction arrays
      real (kind=RKIND), dimension(:,:,:), pointer :: zb_cell
      real (kind=RKIND), dimension(:,:,:), pointer :: zb3_cell

      ! Vertical extrapolation coefficients (scalars in mesh pool)
      real (kind=RKIND), pointer :: cf1_ptr, cf2_ptr, cf3_ptr

      ! Per-cell edge count
      integer, dimension(:), pointer :: nEdgesOnCell

      ! Temporary 0-based copies for integer connectivity arrays
      integer(c_int), dimension(:), allocatable :: edgesOnEdge_0
      integer(c_int), dimension(:), allocatable :: advCellsForEdge_0
      integer(c_int), dimension(:), allocatable :: nEdgesOnEdge_c
      integer(c_int), dimension(:), allocatable :: nAdvCellsForEdge_c
      integer(c_int), dimension(:), allocatable :: nEdgesOnCell_c

      ! Error message buffer
      character(c_char) :: errmsg(256)
      integer(c_int) :: rc
      integer :: i, j, idx

      !---------------------------------------------------------------------

      ierr = 0

      block => domain % blocklist

      ! Get pool handles
      nullify(meshPool)
      nullify(diagPool)
      call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
      call mpas_pool_get_subpool(block % structs, 'diag', diagPool)

      !---------------------------------------------------------------------
      ! Retrieve mesh dimensions
      !---------------------------------------------------------------------
      call mpas_pool_get_dimension(meshPool, 'nCells', nCells_ptr)
      call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges_ptr)
      call mpas_pool_get_dimension(meshPool, 'nVertices', nVertices_ptr)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels_ptr)
      call mpas_pool_get_dimension(meshPool, 'maxEdges', maxEdges_ptr)
      call mpas_pool_get_dimension(meshPool, 'maxEdges2', maxEdges2_ptr)

      nCells = nCells_ptr
      nEdges = nEdges_ptr
      nVertices = nVertices_ptr
      nVertLevels = nVertLevels_ptr
      maxEdges_dim = maxEdges_ptr
      maxEdges2 = maxEdges2_ptr

      !---------------------------------------------------------------------
      ! Retrieve cell geometry arrays from mesh pool
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'invAreaCell', invAreaCell)

      !---------------------------------------------------------------------
      ! Retrieve edge geometry arrays from mesh pool
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
      call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)
      call mpas_pool_get_array(meshPool, 'invDcEdge', invDcEdge)

      !---------------------------------------------------------------------
      ! Retrieve vertical metric arrays from mesh pool
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'rdzw', rdzw)
      call mpas_pool_get_array(meshPool, 'rdzu', rdzu)
      call mpas_pool_get_array(meshPool, 'fzm', fzm)
      call mpas_pool_get_array(meshPool, 'fzp', fzp)
      call mpas_pool_get_array(meshPool, 'etp', etp)
      call mpas_pool_get_array(meshPool, 'etm', etm)
      call mpas_pool_get_array(meshPool, 'ewp', ewp)
      call mpas_pool_get_array(meshPool, 'ewm', ewm)

      !---------------------------------------------------------------------
      ! Terrain metric
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'zz', zz)

      !---------------------------------------------------------------------
      ! Edge orientation signs and specified zone masks from mesh pool
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'edgesOnCell_sign', edgesOnCell_sign)
      call mpas_pool_get_array(meshPool, 'specZoneMaskEdge', specZoneMaskEdge)
      call mpas_pool_get_array(meshPool, 'specZoneMaskCell', specZoneMaskCell)

      !---------------------------------------------------------------------
      ! TRiSK reconstruction data from mesh pool
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'weightsOnEdge', weightsOnEdge)
      call mpas_pool_get_array(meshPool, 'nEdgesOnEdge', nEdgesOnEdge)
      call mpas_pool_get_array(meshPool, 'edgesOnEdge', edgesOnEdge)

      !---------------------------------------------------------------------
      ! Advection stencil data from mesh pool
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'advCellsForEdge', advCellsForEdge)
      call mpas_pool_get_array(meshPool, 'nAdvCellsForEdge', nAdvCellsForEdge)
      call mpas_pool_get_array(meshPool, 'adv_coefs', adv_coefs)
      call mpas_pool_get_array(meshPool, 'adv_coefs_3rd', adv_coefs_3rd)
      maxAdvCells = size(advCellsForEdge, 1)  ! First dimension of advCellsForEdge array

      !---------------------------------------------------------------------
      ! Vertex geometry from mesh pool
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'fVertex', fVertex)
      call mpas_pool_get_array(meshPool, 'areaTriangle', areaTriangle)

      !---------------------------------------------------------------------
      ! Terrain correction arrays from mesh pool
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'zb_cell', zb_cell)
      call mpas_pool_get_array(meshPool, 'zb3_cell', zb3_cell)

      !---------------------------------------------------------------------
      ! Base-state profiles from diag pool
      !---------------------------------------------------------------------
      call mpas_pool_get_array(diagPool, 'rho_base', rho_base)
      call mpas_pool_get_array(diagPool, 'rtheta_base', rtheta_base)
      call mpas_pool_get_array(diagPool, 'exner_base', exner_base)

      !---------------------------------------------------------------------
      ! Vertical extrapolation coefficients (scalars) from mesh pool
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'cf1', cf1_ptr)
      call mpas_pool_get_array(meshPool, 'cf2', cf2_ptr)
      call mpas_pool_get_array(meshPool, 'cf3', cf3_ptr)

      !---------------------------------------------------------------------
      ! Per-cell edge count
      !---------------------------------------------------------------------
      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)

      !---------------------------------------------------------------------
      ! Convert integer connectivity arrays to 0-based for C API
      ! edgesOnEdge: (maxEdges2, nEdges) — subtract 1
      ! advCellsForEdge: (FIFTEEN, nEdges) — subtract 1
      ! nEdgesOnEdge, nAdvCellsForEdge, nEdgesOnCell: no index conversion,
      !   just integer kind conversion
      !---------------------------------------------------------------------
      allocate(edgesOnEdge_0(maxEdges2 * nEdges))
      allocate(advCellsForEdge_0(maxAdvCells * nEdges))
      allocate(nEdgesOnEdge_c(nEdges))
      allocate(nAdvCellsForEdge_c(nEdges))
      allocate(nEdgesOnCell_c(nCells))

      ! Convert edgesOnEdge: (maxEdges2, nEdges) -> flat, subtract 1
      idx = 1
      do j = 1, nEdges
         do i = 1, maxEdges2
            edgesOnEdge_0(idx) = int(edgesOnEdge(i, j) - 1, c_int)
            idx = idx + 1
         end do
      end do

      ! Convert advCellsForEdge: (FIFTEEN, nEdges) -> flat, subtract 1
      idx = 1
      do j = 1, nEdges
         do i = 1, maxAdvCells
            advCellsForEdge_0(idx) = int(advCellsForEdge(i, j) - 1, c_int)
            idx = idx + 1
         end do
      end do

      ! Copy nEdgesOnEdge (integer kind conversion only)
      do i = 1, nEdges
         nEdgesOnEdge_c(i) = int(nEdgesOnEdge(i), c_int)
      end do

      ! Copy nAdvCellsForEdge (integer kind conversion only)
      do i = 1, nEdges
         nAdvCellsForEdge_c(i) = int(nAdvCellsForEdge(i), c_int)
      end do

      ! Copy nEdgesOnCell (integer kind conversion only)
      do i = 1, nCells
         nEdgesOnCell_c(i) = int(nEdgesOnCell(i), c_int)
      end do

      ! Clear error message buffer
      errmsg = c_null_char

      !---------------------------------------------------------------------
      ! Call C++ dycore set_geometry
      ! Real arrays are passed directly (contiguous column-major memory).
      ! Integer connectivity arrays pass the 0-based copies.
      !---------------------------------------------------------------------
      rc = mpas_dycore_cpp_set_geometry( &
         areaCell, invAreaCell, &
         dvEdge, dcEdge, invDcEdge, &
         rdzw, rdzu, &
         fzm, fzp, &
         etp, etm, &
         ewp, ewm, &
         zz, &
         rho_base, rtheta_base, exner_base, &
         edgesOnCell_sign, &
         specZoneMaskEdge, specZoneMaskCell, &
         weightsOnEdge, &
         nEdgesOnEdge_c, edgesOnEdge_0, &
         advCellsForEdge_0, nAdvCellsForEdge_c, &
         adv_coefs, adv_coefs_3rd, &
         fVertex, areaTriangle, &
         zb_cell, zb3_cell, &
         rho_base, rtheta_base, exner_base, &
         real(cf1_ptr, c_double), real(cf2_ptr, c_double), real(cf3_ptr, c_double), &
         nEdgesOnCell_c, &
         int(nCells, c_int), int(nEdges, c_int), int(nVertices, c_int), &
         int(nVertLevels, c_int), int(maxEdges_dim, c_int), &
         int(maxEdges2, c_int), int(maxAdvCells, c_int), &
         errmsg, int(256, c_int))

      ! Check return code
      if (rc /= 0) then
         call mpas_log_write('C++ dycore set_geometry failed: ' // &
                             trim(c_errmsg_to_fortran(errmsg)), &
                             messageType=MPAS_LOG_CRIT)
      end if

      ! Deallocate temporary 0-based copies
      deallocate(edgesOnEdge_0)
      deallocate(advCellsForEdge_0)
      deallocate(nEdgesOnEdge_c)
      deallocate(nAdvCellsForEdge_c)
      deallocate(nEdgesOnCell_c)

      ierr = rc

   end subroutine mpas_cpp_dycore_set_geometry


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine mpas_cpp_dycore_timestep
!
!> \brief Delegate one dynamics timestep to the C++ dycore
!> \details
!>  Extracts prognostic field pointers from the state pool and
!>  passes them to mpas_dycore_cpp_timestep. Updated fields are
!>  written in place by the C++ dycore.
!>  Implemented in task 4.3.
!>
!>  Requirements: 2.3, 2.4, 2.5, 2.6
!
!-----------------------------------------------------------------------
   subroutine mpas_cpp_dycore_timestep(domain, ierr)

      implicit none

      type (domain_type), intent(inout) :: domain
      integer, intent(out) :: ierr

      ! Local variables
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: statePool, meshPool

      ! Prognostic fields at time level 2 (target for advancement)
      real (kind=RKIND), dimension(:,:), pointer :: u
      real (kind=RKIND), dimension(:,:), pointer :: theta_m
      real (kind=RKIND), dimension(:,:), pointer :: rho_zz
      real (kind=RKIND), dimension(:,:), pointer :: w
      real (kind=RKIND), dimension(:,:,:), pointer :: scalars

      ! Mesh dimensions
      integer, pointer :: nCells_ptr, nEdges_ptr, nVertLevels_ptr
      integer :: nCells, nEdges, nVertLevels, nScalars

      ! Error handling
      character(c_char) :: errmsg(256)
      integer(c_int) :: rc

      !---------------------------------------------------------------------

      ierr = 0

      ! Guard: ensure dycore has been initialized
      if (.not. cpp_dycore_initialized) then
         call mpas_log_write('C++ dycore timestep called before initialization', &
                             messageType=MPAS_LOG_CRIT)
      end if

      block => domain % blocklist

      ! Get pool handles
      call mpas_pool_get_subpool(block % structs, 'state', statePool)
      call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)

      !---------------------------------------------------------------------
      ! Retrieve mesh dimensions
      !---------------------------------------------------------------------
      call mpas_pool_get_dimension(meshPool, 'nCells', nCells_ptr)
      call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges_ptr)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels_ptr)
      nCells = nCells_ptr
      nEdges = nEdges_ptr
      nVertLevels = nVertLevels_ptr

      !---------------------------------------------------------------------
      ! Extract prognostic fields at time level 2 (the advanced state).
      ! The C++ dycore reads input from these arrays and writes updated
      ! values in place. The caller (atm_srk3 or framework) is responsible
      ! for copying time level 1 to time level 2 before calling this routine
      ! if separate memory is used. On return, time level 2 contains the
      ! advanced state.
      !---------------------------------------------------------------------
      call mpas_pool_get_array(statePool, 'u', u, 2)
      call mpas_pool_get_array(statePool, 'theta_m', theta_m, 2)
      call mpas_pool_get_array(statePool, 'rho_zz', rho_zz, 2)
      call mpas_pool_get_array(statePool, 'w', w, 2)
      call mpas_pool_get_array(statePool, 'scalars', scalars, 2)

      nScalars = size(scalars, 1)

      ! Clear error message buffer
      errmsg = c_null_char

      !---------------------------------------------------------------------
      ! Call C++ dycore timestep with direct pointer passing.
      ! Fields are updated in place — no copy-back needed.
      ! Dimension arguments describe the array extents:
      !   u:       (nVertLevels, nEdges)
      !   theta_m: (nVertLevels, nCells)
      !   rho_zz:  (nVertLevels, nCells)
      !   w:       (nVertLevels+1, nCells)
      !   scalars: (nScalars, nVertLevels, nCells)
      !---------------------------------------------------------------------
      rc = mpas_dycore_cpp_timestep( &
         u, int(nVertLevels, c_int), int(nEdges, c_int), &
         theta_m, int(nVertLevels, c_int), int(nCells, c_int), &
         rho_zz, int(nVertLevels, c_int), int(nCells, c_int), &
         w, int(nVertLevels + 1, c_int), int(nCells, c_int), &
         scalars, int(nScalars, c_int), int(nVertLevels, c_int), int(nCells, c_int), &
         errmsg, int(256, c_int))

      ! Check return code — halt on error (Requirement 2.6)
      if (rc /= 0) then
         call mpas_log_write('C++ dycore timestep failed: ' // &
                             trim(c_errmsg_to_fortran(errmsg)), &
                             messageType=MPAS_LOG_CRIT)
      end if

      ierr = rc

   end subroutine mpas_cpp_dycore_timestep


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine mpas_cpp_dycore_finalize
!
!> \brief Finalize and deallocate the C++ dycore
!> \details
!>  Calls mpas_dycore_cpp_finalize to release all C++ resources
!>  and resets the initialization flag.
!>  Implemented in task 4.4.
!>
!>  Requirements: 2.6
!
!-----------------------------------------------------------------------
   subroutine mpas_cpp_dycore_finalize(ierr)

      implicit none

      integer, intent(out) :: ierr

      character(c_char) :: errmsg(256)
      integer(c_int) :: rc

      ierr = 0

      if (.not. cpp_dycore_initialized) then
         ! Nothing to finalize
         return
      end if

      ! Clear error message buffer
      errmsg = c_null_char

      ! Call C++ dycore finalization
      rc = mpas_dycore_cpp_finalize(errmsg, int(256, c_int))

      if (rc /= 0) then
         call mpas_log_write('C++ dycore finalization failed: ' // &
                             trim(c_errmsg_to_fortran(errmsg)), &
                             messageType=MPAS_LOG_ERR)
         ierr = 1
      end if

      ! Reset initialization flag
      cpp_dycore_initialized = .false.

   end subroutine mpas_cpp_dycore_finalize


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  function c_errmsg_to_fortran
!
!> \brief Convert a C null-terminated char array to a Fortran string
!
!-----------------------------------------------------------------------
   function c_errmsg_to_fortran(cmsg) result(fstr)

      implicit none

      character(c_char), intent(in) :: cmsg(256)
      character(len=255) :: fstr
      integer :: i

      fstr = ' '
      do i = 1, 255
         if (cmsg(i) == c_null_char) exit
         fstr(i:i) = cmsg(i)
      end do

   end function c_errmsg_to_fortran


end module mpas_dycore_marshalling
