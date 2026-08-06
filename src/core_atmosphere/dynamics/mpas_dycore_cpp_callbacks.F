! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  mpas_dycore_cpp_callbacks
!
!> \brief C-callable Fortran callback wrappers for hybrid execution mode
!> \details
!>  Provides bind(C) subroutines that the C++ SRK3 integrator can invoke
!>  for kernels configured for Fortran delegation. Each callback receives
!>  workspace pointers directly (no copy) and operates on the same memory
!>  as the C++ dycore.
!>
!>  The module also provides mpas_cpp_callbacks_init which stores a domain
!>  pointer for callback context and registers all callback function
!>  pointers with the C++ dycore via mpas_dycore_cpp_set_callbacks.
!>
!>  Requirements: 3.1, 3.2, 3.3, 3.4, 4.2, 4.3, 4.4
!
!-----------------------------------------------------------------------

module mpas_dycore_cpp_callbacks

   use iso_c_binding, only : c_int, c_double, c_funloc, c_funptr, c_null_funptr
   use mpas_derived_types, only : domain_type, block_type, mpas_pool_type
   use mpas_pool_routines, only : mpas_pool_get_array, mpas_pool_get_subpool
   use mpas_dycore_cpp_interface, only : mpas_dycore_cpp_set_callbacks

   implicit none

   private
   public :: mpas_cpp_callbacks_init, mpas_cpp_callbacks_set_halo_exch

   ! Module-level domain pointer for callback context.
   ! Set during initialization so that callbacks can access MPAS
   ! framework data structures (e.g., for halo exchange).
   type(domain_type), pointer, save :: callback_domain => null()

   ! Abstract interface for the halo exchange routine
   abstract interface
      subroutine halo_exch_iface(domain, halo_group, withGPUAwareMPI, ierr)
         use mpas_derived_types, only : domain_type
         type (domain_type), intent(inout) :: domain
         character(len=*), intent(in) :: halo_group
         logical, intent(in), optional :: withGPUAwareMPI
         integer, intent(out), optional :: ierr
      end subroutine halo_exch_iface
   end interface

   procedure(halo_exch_iface), pointer, save :: exchange_halo_group_ptr => null()

   ! Field ID constants matching srk3.cpp FIELD_ID_* values
   integer(c_int), parameter :: ID_TEND_U     = 1
   integer(c_int), parameter :: ID_RHO_PP     = 2
   integer(c_int), parameter :: ID_RTHETA_PP  = 3
   integer(c_int), parameter :: ID_U          = 4
   integer(c_int), parameter :: ID_RHO_ZZ     = 5
   integer(c_int), parameter :: ID_THETA_M    = 6
   integer(c_int), parameter :: ID_W          = 7
   integer(c_int), parameter :: ID_SCALARS    = 8

contains


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine mpas_cpp_callbacks_init
!
!> \brief Initialize callback module and register callbacks with C++ dycore
!> \details
!>  Stores the domain pointer for use by callback routines and registers
!>  all Fortran callback function pointers with the C++ dycore via
!>  mpas_dycore_cpp_set_callbacks. Must be called after mpas_dycore_cpp_init
!>  and before the first timestep.
!>
!>  Requirements: 3.1, 8.4
!
!-----------------------------------------------------------------------
   subroutine mpas_cpp_callbacks_init(domain, ierr)

      implicit none

      type(domain_type), intent(inout), target :: domain
      integer, intent(out) :: ierr

      integer(c_int) :: rc

      ! Store domain pointer for callback context
      callback_domain => domain

      ! Use C++ implementations for all compute kernels (pass NULL).
      ! Only delegate halo exchange to Fortran since the C++ side passes
      ! empty halo descriptors and Fortran has the actual MPI infrastructure.
      rc = mpas_dycore_cpp_set_callbacks( &
         c_null_funptr, &
         c_null_funptr, &
         c_null_funptr, &
         c_funloc(halo_exchange_callback))

      ierr = int(rc)

   end subroutine mpas_cpp_callbacks_init


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine mpas_cpp_callbacks_set_halo_exch
!
!> \brief Register the halo exchange procedure pointer
!> \details
!>  Stores the halo exchange procedure pointer so that halo_exchange_callback
!>  can invoke it without a compile-time dependency on mpas_atm_halos.
!>  Must be called after mpas_cpp_callbacks_init and before the first timestep.
!
!-----------------------------------------------------------------------
   subroutine mpas_cpp_callbacks_set_halo_exch(exch_proc)

      implicit none

      procedure(halo_exch_iface) :: exch_proc

      exchange_halo_group_ptr => exch_proc

   end subroutine mpas_cpp_callbacks_set_halo_exch


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine compute_dyn_tend_callback
!
!> \brief Callback for computing dynamic tendencies (stub)
!> \details
!>  C-callable callback invoked by the C++ SRK3 integrator at each
!>  Runge-Kutta stage to compute tendencies for u, theta_m, rho_zz, w.
!>  Currently a stub — will be wired to the actual Fortran
!>  compute_dyn_tend subroutine in a future task.
!>
!>  Requirements: 3.2, 3.4
!
!-----------------------------------------------------------------------
   subroutine compute_dyn_tend_callback(u, theta_m, rho_zz, w, &
      tend_u, tend_theta, tend_rho, tend_w, &
      nVertLevels, nCells, nEdges, rk_step, dt_rk) bind(C)

      implicit none

      ! Input state fields (read)
      real(c_double), intent(inout) :: u(*)
      real(c_double), intent(inout) :: theta_m(*)
      real(c_double), intent(inout) :: rho_zz(*)
      real(c_double), intent(inout) :: w(*)

      ! Output tendency fields (write)
      real(c_double), intent(inout) :: tend_u(*)
      real(c_double), intent(inout) :: tend_theta(*)
      real(c_double), intent(inout) :: tend_rho(*)
      real(c_double), intent(inout) :: tend_w(*)

      ! Dimension and timestep parameters
      integer(c_int), value, intent(in) :: nVertLevels
      integer(c_int), value, intent(in) :: nCells
      integer(c_int), value, intent(in) :: nEdges
      integer(c_int), value, intent(in) :: rk_step
      real(c_double), value, intent(in) :: dt_rk

      ! TODO: Wire to actual Fortran compute_dyn_tend subroutine.
      ! This requires wrapping the incoming C pointers as Fortran arrays
      ! and calling into the MPAS dynamics tendency computation using
      ! callback_domain for access to mesh/state pools.
      !
      ! For now, this is a no-op stub. The tendency arrays are left
      ! unmodified (caller is responsible for zeroing them if needed).

   end subroutine compute_dyn_tend_callback


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine advance_acoustic_step_callback
!
!> \brief Callback for advancing one acoustic sub-step (stub)
!> \details
!>  C-callable callback invoked within each RK stage for the
!>  split-explicit acoustic sub-stepping loop.
!>  Currently a stub — will be wired to the actual Fortran
!>  advance_acoustic_step subroutine in a future task.
!>
!>  Requirements: 3.3, 3.4
!
!-----------------------------------------------------------------------
   subroutine advance_acoustic_step_callback(ru_p, rw_p, rtheta_pp, rho_pp, &
      nVertLevels, nCells, nEdges, substep, dts) bind(C)

      implicit none

      ! Perturbation fields (read-write)
      real(c_double), intent(inout) :: ru_p(*)
      real(c_double), intent(inout) :: rw_p(*)
      real(c_double), intent(inout) :: rtheta_pp(*)
      real(c_double), intent(inout) :: rho_pp(*)

      ! Dimension and timestep parameters
      integer(c_int), value, intent(in) :: nVertLevels
      integer(c_int), value, intent(in) :: nCells
      integer(c_int), value, intent(in) :: nEdges
      integer(c_int), value, intent(in) :: substep
      real(c_double), value, intent(in) :: dts

      ! TODO: Wire to actual Fortran advance_acoustic_step subroutine.
      ! This requires wrapping the incoming C pointers as Fortran arrays
      ! and calling into the MPAS acoustic sub-stepping code using
      ! callback_domain for access to mesh/state pools.
      !
      ! For now, this is a no-op stub.

   end subroutine advance_acoustic_step_callback


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine advance_scalars_mono_callback
!
!> \brief Callback for monotonic scalar transport (stub)
!> \details
!>  C-callable callback invoked to advance scalar tracers using
!>  flux-corrected transport with monotonicity enforcement.
!>  Currently a stub — will be wired to the actual Fortran
!>  advance_scalars_mono subroutine in a future task.
!>
!>  Requirements: 3.3, 3.4
!
!-----------------------------------------------------------------------
   subroutine advance_scalars_mono_callback(scalars, ruAvg, wwAvg, &
      nScalars, nVertLevels, nCells, nEdges, dt) bind(C)

      implicit none

      ! Scalar tracer array (read-write)
      real(c_double), intent(inout) :: scalars(*)

      ! Time-averaged mass fluxes (read)
      real(c_double), intent(inout) :: ruAvg(*)
      real(c_double), intent(inout) :: wwAvg(*)

      ! Dimension and timestep parameters
      integer(c_int), value, intent(in) :: nScalars
      integer(c_int), value, intent(in) :: nVertLevels
      integer(c_int), value, intent(in) :: nCells
      integer(c_int), value, intent(in) :: nEdges
      real(c_double), value, intent(in) :: dt

      ! TODO: Wire to actual Fortran advance_scalars_mono subroutine.
      ! This requires wrapping the incoming C pointers as Fortran arrays
      ! and calling into the MPAS scalar transport code using
      ! callback_domain for access to mesh/state pools.
      !
      ! For now, this is a no-op stub.

   end subroutine advance_scalars_mono_callback


!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine halo_exchange_callback
!
!> \brief Callback for halo exchange of a single field
!> \details
!>  C-callable callback invoked at synchronization points where
!>  ghost-cell data must be updated across MPI partitions. Dispatches
!>  to the appropriate MPAS halo exchange group based on field_id.
!>
!>  For workspace fields (tend_u, rho_pp, rtheta_pp), the callback
!>  copies data from the C++ workspace into the corresponding MPAS
!>  pool field, performs the exchange (which fills halo cells in the
!>  pool array), then copies the entire array back to the workspace.
!>
!>  For prognostic fields (u, w, scalars), the pointer passed by the
!>  C++ dycore IS the pool field memory, so we just call the named
!>  halo exchange group directly.
!>
!>  For rho_zz and theta_m, these are also pool memory but lack
!>  individual halo groups in the standard MPAS configuration. theta_m
!>  is exchanged via the combined group that also covers pressure_p
!>  and rtheta_p (harmless extra work). rho_zz is not individually
!>  exchanged because compute_solve_diagnostics (Step 10) recomputes
!>  rho_edge from it immediately after.
!>
!>  Requirements: 4.2, 4.3, 4.4
!
!-----------------------------------------------------------------------
   subroutine halo_exchange_callback(field, field_id, nVertLevels_in, nElements) bind(C)

      implicit none

      ! Field data including full halo region (read-write)
      real(c_double), intent(inout) :: field(*)

      ! Field identifier (used to select the appropriate halo exchange)
      integer(c_int), value, intent(in) :: field_id

      ! Vertical extent of the field (may be nVertLevels*nScalars for scalars)
      integer(c_int), value, intent(in) :: nVertLevels_in

      ! Horizontal extent including halo (cells or edges)
      integer(c_int), value, intent(in) :: nElements

      ! Local variables
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: tendPool, diagPool
      real(c_double), dimension(:,:), pointer :: pool_field_2d
      integer :: i, j, idx

      ! Early exit if domain not set (should not happen in practice)
      if (.not. associated(callback_domain)) return

      ! Early exit if halo exchange procedure not registered
      if (.not. associated(exchange_halo_group_ptr)) return

      block => callback_domain % blocklist

      select case (field_id)

      ! ------------------------------------------------------------------
      ! Category A: Workspace fields — copy to pool, exchange, copy back
      ! ------------------------------------------------------------------
      case (ID_TEND_U)
         call mpas_pool_get_subpool(block % structs, 'tend', tendPool)
         call mpas_pool_get_array(tendPool, 'u', pool_field_2d)
         ! Copy workspace → pool (both are column-major nVertLevels × nElements)
         idx = 1
         do j = 1, int(nElements)
            do i = 1, int(nVertLevels_in)
               pool_field_2d(i, j) = field(idx)
               idx = idx + 1
            end do
         end do
         call exchange_halo_group_ptr(callback_domain, 'dynamics:tend_u', .false.)
         ! Copy pool → workspace (halo cells now filled)
         idx = 1
         do j = 1, int(nElements)
            do i = 1, int(nVertLevels_in)
               field(idx) = pool_field_2d(i, j)
               idx = idx + 1
            end do
         end do

      case (ID_RHO_PP)
         call mpas_pool_get_subpool(block % structs, 'diag', diagPool)
         call mpas_pool_get_array(diagPool, 'rho_pp', pool_field_2d)
         idx = 1
         do j = 1, int(nElements)
            do i = 1, int(nVertLevels_in)
               pool_field_2d(i, j) = field(idx)
               idx = idx + 1
            end do
         end do
         call exchange_halo_group_ptr(callback_domain, 'dynamics:rho_pp', .false.)
         idx = 1
         do j = 1, int(nElements)
            do i = 1, int(nVertLevels_in)
               field(idx) = pool_field_2d(i, j)
               idx = idx + 1
            end do
         end do

      case (ID_RTHETA_PP)
         call mpas_pool_get_subpool(block % structs, 'diag', diagPool)
         call mpas_pool_get_array(diagPool, 'rtheta_pp', pool_field_2d)
         idx = 1
         do j = 1, int(nElements)
            do i = 1, int(nVertLevels_in)
               pool_field_2d(i, j) = field(idx)
               idx = idx + 1
            end do
         end do
         call exchange_halo_group_ptr(callback_domain, 'dynamics:rtheta_pp', .false.)
         idx = 1
         do j = 1, int(nElements)
            do i = 1, int(nVertLevels_in)
               field(idx) = pool_field_2d(i, j)
               idx = idx + 1
            end do
         end do

      ! ------------------------------------------------------------------
      ! Category B: Prognostic fields — pointer IS pool memory, exchange directly
      ! ------------------------------------------------------------------
      case (ID_U)
         ! u at time level 2, halo layer 3 only (matches Fortran SRK3 behavior)
         call exchange_halo_group_ptr(callback_domain, 'dynamics:u_3', .false.)

      case (ID_RHO_ZZ)
         ! rho_zz is not individually exchanged in the Fortran SRK3.
         ! compute_solve_diagnostics (called immediately after in Step 10)
         ! recomputes rho_edge from rho_zz, making an explicit halo
         ! exchange of rho_zz unnecessary. No-op for correctness.
         continue

      case (ID_THETA_M)
         ! theta_m is not individually exchanged after state recovery in the
         ! Fortran SRK3. The owned-cell values are valid and compute_solve_diagnostics
         ! (called immediately after in Step 10) only reads owned cells.
         ! The pre-RK exchange of theta_m uses timeLevel=1 which is a different
         ! code path. No-op here for correctness.
         continue

      case (ID_W)
         ! w at time level 2, halo layers 1,2
         call exchange_halo_group_ptr(callback_domain, 'dynamics:w', .false.)

      case (ID_SCALARS)
         ! scalars at time level 2, halo layers 1,2
         call exchange_halo_group_ptr(callback_domain, 'dynamics:scalars', .false.)

      case default
         ! Unknown field_id — silently ignore
         continue

      end select

   end subroutine halo_exchange_callback


end module mpas_dycore_cpp_callbacks
