!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  mpas_dycore_cpp_interface
!
!> \brief ISO_C_BINDING interface to the C++ dynamical core API
!> \details
!>  Provides Fortran interface blocks for all C API entry points defined
!>  in api.h, enabling type-safe calls from Fortran to the C++ dycore.
!>  All connectivity/geometry arrays are passed as assumed-size arrays;
!>  scalar parameters use value semantics.
!>
!>  Requirements: 2.1, 2.2, 2.3, 8.1, 8.3
!
!-----------------------------------------------------------------------

module mpas_dycore_cpp_interface

   use iso_c_binding, only : c_int, c_double, c_char, c_funptr

   implicit none

   private
   public :: mpas_dycore_cpp_init, &
             mpas_dycore_cpp_set_geometry, &
             mpas_dycore_cpp_timestep, &
             mpas_dycore_cpp_finalize, &
             mpas_dycore_cpp_set_callbacks

   interface

      !--------------------------------------------------------------------
      ! mpas_dycore_cpp_init
      !
      ! Initialize the C++ dynamical core with mesh connectivity, halo
      ! descriptors, and configuration parameters.
      !--------------------------------------------------------------------
      integer(c_int) function mpas_dycore_cpp_init( &
         nCells, nEdges, nVertices, maxEdges, nVertLevels, &
         cellsOnEdge, cellsOnEdge_d0, cellsOnEdge_d1, &
         verticesOnEdge, verticesOnEdge_d0, verticesOnEdge_d1, &
         edgesOnCell, edgesOnCell_d0, edgesOnCell_d1, &
         cellsOnCell, cellsOnCell_d0, cellsOnCell_d1, &
         nEdgesOnCell_arr, nEdgesOnCell_len, &
         halo_ranks, halo_n_neighbors, &
         halo_counts, halo_counts_len, &
         halo_send_indices, halo_send_len, &
         halo_recv_indices, halo_recv_len, &
         halo_n_layers, &
         dt, number_of_sub_steps, &
         config_apply_lbcs, &
         config_horiz_mixing, &
         config_h_mom_eddy_visc2, &
         config_h_mom_eddy_visc4, &
         config_h_theta_eddy_visc2, &
         mpi_comm_fortran, &
         errmsg, errmsg_len) bind(C, name='mpas_dycore_cpp_init')

         import :: c_int, c_double, c_char

         ! Mesh dimensions
         integer(c_int), value, intent(in) :: nCells
         integer(c_int), value, intent(in) :: nEdges
         integer(c_int), value, intent(in) :: nVertices
         integer(c_int), value, intent(in) :: maxEdges
         integer(c_int), value, intent(in) :: nVertLevels

         ! Connectivity arrays (zero-based from marshaller)
         integer(c_int), intent(in) :: cellsOnEdge(*)
         integer(c_int), value, intent(in) :: cellsOnEdge_d0
         integer(c_int), value, intent(in) :: cellsOnEdge_d1

         integer(c_int), intent(in) :: verticesOnEdge(*)
         integer(c_int), value, intent(in) :: verticesOnEdge_d0
         integer(c_int), value, intent(in) :: verticesOnEdge_d1

         integer(c_int), intent(in) :: edgesOnCell(*)
         integer(c_int), value, intent(in) :: edgesOnCell_d0
         integer(c_int), value, intent(in) :: edgesOnCell_d1

         integer(c_int), intent(in) :: cellsOnCell(*)
         integer(c_int), value, intent(in) :: cellsOnCell_d0
         integer(c_int), value, intent(in) :: cellsOnCell_d1

         integer(c_int), intent(in) :: nEdgesOnCell_arr(*)
         integer(c_int), value, intent(in) :: nEdgesOnCell_len

         ! Halo CSR descriptors
         integer(c_int), intent(in) :: halo_ranks(*)
         integer(c_int), value, intent(in) :: halo_n_neighbors

         integer(c_int), intent(in) :: halo_counts(*)
         integer(c_int), value, intent(in) :: halo_counts_len

         integer(c_int), intent(in) :: halo_send_indices(*)
         integer(c_int), value, intent(in) :: halo_send_len

         integer(c_int), intent(in) :: halo_recv_indices(*)
         integer(c_int), value, intent(in) :: halo_recv_len

         integer(c_int), value, intent(in) :: halo_n_layers

         ! Configuration
         real(c_double), value, intent(in) :: dt
         integer(c_int), value, intent(in) :: number_of_sub_steps
         integer(c_int), value, intent(in) :: config_apply_lbcs
         character(c_char), intent(in) :: config_horiz_mixing(*)
         real(c_double), value, intent(in) :: config_h_mom_eddy_visc2
         real(c_double), value, intent(in) :: config_h_mom_eddy_visc4
         real(c_double), value, intent(in) :: config_h_theta_eddy_visc2

         ! MPI communicator (Fortran handle)
         integer(c_int), value, intent(in) :: mpi_comm_fortran

         ! Error output
         character(c_char), intent(inout) :: errmsg(*)
         integer(c_int), value, intent(in) :: errmsg_len

      end function mpas_dycore_cpp_init

      !--------------------------------------------------------------------
      ! mpas_dycore_cpp_set_geometry
      !
      ! Set mesh geometry, metric terms, base-state profiles, and stencil
      ! data. Must be called after init and before timestep.
      !--------------------------------------------------------------------
      integer(c_int) function mpas_dycore_cpp_set_geometry( &
         areaCell, invAreaCell, &
         dvEdge, dcEdge, invDcEdge, &
         rdzw, rdzu, &
         fzm, fzp, &
         etp, etm, &
         ewp, ewm, &
         zz, &
         rb, rtb, pb, &
         edgesOnCell_sign, &
         specZoneMaskEdge, specZoneMaskCell, &
         weightsOnEdge, &
         nEdgesOnEdge, edgesOnEdge, &
         advCellsForEdge, nAdvCellsForEdge, &
         adv_coefs, adv_coefs_3rd, &
         fVertex, areaTriangle, &
         zb_cell_ptr, zb3_cell_ptr, &
         rho_base_ptr, rtheta_base_ptr, exner_base_ptr, &
         cf1, cf2, cf3, &
         nEdgesOnCell_ptr, &
         nCells, nEdges, nVertices, nVertLevels, &
         maxEdges, maxEdges2, maxAdvCells, &
         errmsg, errmsg_len) bind(C, name='mpas_dycore_cpp_set_geometry')

         import :: c_int, c_double, c_char

         ! Cell geometry
         real(c_double), intent(in) :: areaCell(*)
         real(c_double), intent(in) :: invAreaCell(*)

         ! Edge geometry
         real(c_double), intent(in) :: dvEdge(*)
         real(c_double), intent(in) :: dcEdge(*)
         real(c_double), intent(in) :: invDcEdge(*)

         ! Vertical metrics
         real(c_double), intent(in) :: rdzw(*)
         real(c_double), intent(in) :: rdzu(*)
         real(c_double), intent(in) :: fzm(*)
         real(c_double), intent(in) :: fzp(*)
         real(c_double), intent(in) :: etp(*)
         real(c_double), intent(in) :: etm(*)
         real(c_double), intent(in) :: ewp(*)
         real(c_double), intent(in) :: ewm(*)

         ! Terrain metric
         real(c_double), intent(in) :: zz(*)

         ! Base-state profiles
         real(c_double), intent(in) :: rb(*)
         real(c_double), intent(in) :: rtb(*)
         real(c_double), intent(in) :: pb(*)

         ! Edge orientation signs
         real(c_double), intent(in) :: edgesOnCell_sign(*)

         ! Specified zone masks
         real(c_double), intent(in) :: specZoneMaskEdge(*)
         real(c_double), intent(in) :: specZoneMaskCell(*)

         ! Edge reconstruction data (TRiSK)
         real(c_double), intent(in) :: weightsOnEdge(*)
         integer(c_int), intent(in) :: nEdgesOnEdge(*)
         integer(c_int), intent(in) :: edgesOnEdge(*)

         ! Advection stencils
         integer(c_int), intent(in) :: advCellsForEdge(*)
         integer(c_int), intent(in) :: nAdvCellsForEdge(*)
         real(c_double), intent(in) :: adv_coefs(*)
         real(c_double), intent(in) :: adv_coefs_3rd(*)

         ! Vertex geometry
         real(c_double), intent(in) :: fVertex(*)
         real(c_double), intent(in) :: areaTriangle(*)

         ! Terrain correction arrays (may be NULL/C_NULL_PTR from caller)
         real(c_double), intent(in) :: zb_cell_ptr(*)
         real(c_double), intent(in) :: zb3_cell_ptr(*)

         ! Base-state for perturbation formulation (may be NULL)
         real(c_double), intent(in) :: rho_base_ptr(*)
         real(c_double), intent(in) :: rtheta_base_ptr(*)
         real(c_double), intent(in) :: exner_base_ptr(*)

         ! Vertical extrapolation coefficients
         real(c_double), value, intent(in) :: cf1
         real(c_double), value, intent(in) :: cf2
         real(c_double), value, intent(in) :: cf3

         ! Per-cell edge count for terrain correction (may be NULL)
         integer(c_int), intent(in) :: nEdgesOnCell_ptr(*)

         ! Dimension metadata
         integer(c_int), value, intent(in) :: nCells
         integer(c_int), value, intent(in) :: nEdges
         integer(c_int), value, intent(in) :: nVertices
         integer(c_int), value, intent(in) :: nVertLevels
         integer(c_int), value, intent(in) :: maxEdges
         integer(c_int), value, intent(in) :: maxEdges2
         integer(c_int), value, intent(in) :: maxAdvCells

         ! Error output
         character(c_char), intent(inout) :: errmsg(*)
         integer(c_int), value, intent(in) :: errmsg_len

      end function mpas_dycore_cpp_set_geometry

      !--------------------------------------------------------------------
      ! mpas_dycore_cpp_timestep
      !
      ! Execute one large timestep of the dynamical core.
      ! Prognostic fields are read-write (updated in place).
      !--------------------------------------------------------------------
      integer(c_int) function mpas_dycore_cpp_timestep( &
         u, u_d0, u_d1, &
         theta_m, theta_m_d0, theta_m_d1, &
         rho_zz, rho_zz_d0, rho_zz_d1, &
         w, w_d0, w_d1, &
         scalars, scalars_d0, scalars_d1, scalars_d2, &
         errmsg, errmsg_len) bind(C, name='mpas_dycore_cpp_timestep')

         import :: c_int, c_double, c_char

         ! Horizontal velocity (nVertLevels x nEdges), read-write
         real(c_double), intent(inout) :: u(*)
         integer(c_int), value, intent(in) :: u_d0
         integer(c_int), value, intent(in) :: u_d1

         ! Potential temperature (nVertLevels x nCells), read-write
         real(c_double), intent(inout) :: theta_m(*)
         integer(c_int), value, intent(in) :: theta_m_d0
         integer(c_int), value, intent(in) :: theta_m_d1

         ! Dry air density (nVertLevels x nCells), read-write
         real(c_double), intent(inout) :: rho_zz(*)
         integer(c_int), value, intent(in) :: rho_zz_d0
         integer(c_int), value, intent(in) :: rho_zz_d1

         ! Vertical velocity (nVertLevels+1 x nCells), read-write
         real(c_double), intent(inout) :: w(*)
         integer(c_int), value, intent(in) :: w_d0
         integer(c_int), value, intent(in) :: w_d1

         ! Scalar tracers (nScalars x nVertLevels x nCells), read-write
         real(c_double), intent(inout) :: scalars(*)
         integer(c_int), value, intent(in) :: scalars_d0
         integer(c_int), value, intent(in) :: scalars_d1
         integer(c_int), value, intent(in) :: scalars_d2

         ! Error output
         character(c_char), intent(inout) :: errmsg(*)
         integer(c_int), value, intent(in) :: errmsg_len

      end function mpas_dycore_cpp_timestep

      !--------------------------------------------------------------------
      ! mpas_dycore_cpp_finalize
      !
      ! Release all internally-allocated resources and reset to
      ! uninitialized state.
      !--------------------------------------------------------------------
      integer(c_int) function mpas_dycore_cpp_finalize( &
         errmsg, errmsg_len) bind(C, name='mpas_dycore_cpp_finalize')

         import :: c_int, c_char

         character(c_char), intent(inout) :: errmsg(*)
         integer(c_int), value, intent(in) :: errmsg_len

      end function mpas_dycore_cpp_finalize

      !--------------------------------------------------------------------
      ! mpas_dycore_cpp_set_callbacks
      !
      ! Register Fortran callback function pointers for hybrid execution.
      ! Must be called after init and before the first timestep.
      ! Pass C_NULL_FUNPTR for any kernel that should use the C++
      ! implementation instead of a Fortran callback.
      !--------------------------------------------------------------------
      integer(c_int) function mpas_dycore_cpp_set_callbacks( &
         compute_dyn_tend_cb, advance_acoustic_step_cb, &
         advance_scalars_mono_cb, halo_exchange_cb) &
         bind(C, name='mpas_dycore_cpp_set_callbacks')

         import :: c_int, c_funptr

         type(c_funptr), value, intent(in) :: compute_dyn_tend_cb
         type(c_funptr), value, intent(in) :: advance_acoustic_step_cb
         type(c_funptr), value, intent(in) :: advance_scalars_mono_cb
         type(c_funptr), value, intent(in) :: halo_exchange_cb

      end function mpas_dycore_cpp_set_callbacks

   end interface

end module mpas_dycore_cpp_interface
