#include <Kokkos_Core.hpp>
#include "atm_dynamics_kernels.hpp"

extern "C" {

void mpas_kokkos_init() {
    int argc = 0;
    char** argv = nullptr;
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize(argc, argv);
    }
}

void mpas_kokkos_finalize() {
    if (Kokkos::is_initialized()) {
        Kokkos::finalize();
    }
}

void atm_compute_vert_imp_coefs_kokkos(
    int nCells, int nVertLevels, double dts, double epssm,
    double* zz_ptr, double* cqw_ptr, double* p_ptr, double* t_ptr, double* rb_ptr, double* rtb_ptr,
    double* pb_ptr, double* rt_ptr, double* qtot_ptr,
    double* cofwr_ptr, double* cofwz_ptr, double* coftz_ptr, double* cofwt_ptr,
    double* a_tri_ptr, double* alpha_tri_ptr, double* gamma_tri_ptr,
    double* cofrz_ptr, double* rdzw_ptr, double* fzm_ptr, double* fzp_ptr, double* rdzu_ptr,
    int cellSolveStart, int cellSolveEnd,
    double gravity, double rgas, double cp
) {
    typedef Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>> View2D;
    typedef Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>> View1D;

    View2D zz(zz_ptr, nVertLevels, nCells + 1);
    View2D cqw(cqw_ptr, nVertLevels, nCells + 1);
    View2D p(p_ptr, nVertLevels, nCells + 1);
    View2D t(t_ptr, nVertLevels, nCells + 1);
    View2D rb(rb_ptr, nVertLevels, nCells + 1);
    View2D rtb(rtb_ptr, nVertLevels, nCells + 1);
    View2D pb(pb_ptr, nVertLevels, nCells + 1);
    View2D rt(rt_ptr, nVertLevels, nCells + 1);
    View2D qtot(qtot_ptr, nVertLevels, nCells + 1);

    View2D cofwr(cofwr_ptr, nVertLevels, nCells + 1);
    View2D cofwz(cofwz_ptr, nVertLevels, nCells + 1);
    View2D coftz(coftz_ptr, nVertLevels + 1, nCells + 1);
    View2D cofwt(cofwt_ptr, nVertLevels, nCells + 1);
    View2D a_tri(a_tri_ptr, nVertLevels, nCells + 1);
    View2D alpha_tri(alpha_tri_ptr, nVertLevels, nCells + 1);
    View2D gamma_tri(gamma_tri_ptr, nVertLevels, nCells + 1);

    View1D cofrz(cofrz_ptr, nVertLevels);
    View1D rdzw(rdzw_ptr, nVertLevels);
    View1D fzm(fzm_ptr, nVertLevels);
    View1D fzp(fzp_ptr, nVertLevels);
    View1D rdzu(rdzu_ptr, nVertLevels);

    double dtseps = 0.5 * dts * (1.0 + epssm);
    double rcv = rgas / (cp - rgas);
    double c2 = cp * rcv;

    Kokkos::parallel_for("set_cofrz", nVertLevels, KOKKOS_LAMBDA(int k) {
        cofrz(k) = dtseps * rdzw(k);
    });

    int start = cellSolveStart - 1;
    int end = cellSolveEnd;

    // Use a scratch pad for b_tri and c_tri if running on device, or allocate if on host
    // For simplicity, we use internal 2D views for b_tri and c_tri.
    // In a real GPU scenario, this should be carefully managed (e.g., TeamPolicy with scratch memory).
    Kokkos::View<double**, Kokkos::LayoutLeft> b_tri("b_tri", nVertLevels, nCells + 1);
    Kokkos::View<double**, Kokkos::LayoutLeft> c_tri("c_tri", nVertLevels, nCells + 1);

    Kokkos::parallel_for("atm_compute_vert_imp_coefs", Kokkos::RangePolicy<int>(start, end), KOKKOS_LAMBDA(int i) {
        for (int k = 1; k < nVertLevels; ++k) {
            cofwr(k, i) = 0.5 * dtseps * gravity * (fzm(k) * zz(k, i) + fzp(k) * zz(k-1, i));
        }

        coftz(0, i) = 0.0;
        for (int k = 1; k < nVertLevels; ++k) {
            cofwz(k, i) = dtseps * c2 * (fzm(k) * zz(k, i) + fzp(k) * zz(k-1, i))
                        * rdzu(k) * cqw(k, i) * (fzm(k) * p(k, i) + fzp(k) * p(k-1, i));
            coftz(k, i) = dtseps * (fzm(k) * t(k, i) + fzp(k) * t(k-1, i));
        }
        coftz(nVertLevels, i) = 0.0;

        for (int k = 0; k < nVertLevels; ++k) {
            double qtotal = qtot(k, i);
            cofwt(k, i) = 0.5 * dtseps * rcv * zz(k, i) * gravity * rb(k, i) / (1.0 + qtotal)
                        * p(k, i) / ((rtb(k, i) + rt(k, i)) * pb(k, i));
        }

        a_tri(0, i) = 0.0;
        gamma_tri(0, i) = 0.0;
        alpha_tri(0, i) = 0.0;

        for (int k = 1; k < nVertLevels; ++k) {
            a_tri(k, i) = -cofwz(k, i) * coftz(k-1, i) * rdzw(k-1) * zz(k-1, i)
                        + cofwr(k, i) * cofrz(k-1)
                        - cofwt(k-1, i) * coftz(k-1, i) * rdzw(k-1);

            b_tri(k, i) = 1.0
                           + cofwz(k, i) * (coftz(k, i) * rdzw(k) * zz(k, i) + coftz(k, i) * rdzw(k-1) * zz(k-1, i))
                           - coftz(k, i) * (cofwt(k, i) * rdzw(k) - cofwt(k-1, i) * rdzw(k-1))
                           + cofwr(k, i) * (cofrz(k) - cofrz(k-1));

            c_tri(k, i) = -cofwz(k, i) * coftz(k + 1, i) * rdzw(k) * zz(k, i)
                           - cofwr(k, i) * cofrz(k)
                           + cofwt(k, i) * coftz(k + 1, i) * rdzw(k);
        }

        for (int k = 1; k < nVertLevels; ++k) {
            alpha_tri(k, i) = 1.0 / (b_tri(k, i) - a_tri(k, i) * gamma_tri(k-1, i));
            gamma_tri(k, i) = c_tri(k, i) * alpha_tri(k, i);
        }
    });
}

}
