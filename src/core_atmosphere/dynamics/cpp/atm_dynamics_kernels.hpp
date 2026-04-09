#ifndef ATM_DYNAMICS_KERNELS_HPP
#define ATM_DYNAMICS_KERNELS_HPP

extern "C" {
void mpas_kokkos_init();
void mpas_kokkos_finalize();

void atm_compute_vert_imp_coefs_kokkos(
    int nCells, int nVertLevels, double dts, double epssm,
    double* zz, double* cqw, double* p, double* t, double* rb, double* rtb,
    double* pb, double* rt, double* qtot,
    double* cofwr, double* cofwz, double* coftz, double* cofwt,
    double* a_tri, double* alpha_tri, double* gamma_tri,
    double* cofrz, double* rdzw, double* fzm, double* fzp, double* rdzu,
    int cellSolveStart, int cellSolveEnd,
    double gravity, double rgas, double cp
);
}

#endif
