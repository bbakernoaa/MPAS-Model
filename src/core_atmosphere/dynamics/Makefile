.SUFFIXES: .F .o

OBJS = mpas_atm_time_integration.o \
       mpas_atm_boundaries.o \
       mpas_atm_dissipation_models.o

# ---------------------------------------------------------------------------
# C++/Kokkos dynamical-core integration (fortran-integration spec).
# When USE_CPP_DYCORE=true is set at the top-level, CPPFLAGS contains
# -DMPAS_CPP_DYCORE. We detect that here to add the Fortran shim and
# marshalling objects. No C++ toolchain is required when the flag is absent.
# Requirements: 5.4, 5.6, 7.2
ifneq (,$(findstring MPAS_CPP_DYCORE,$(CPPFLAGS)))
OBJS += mpas_dycore_interface.o mpas_dycore_marshalling.o
endif

# ---------------------------------------------------------------------------
# C++/Kokkos dynamical-core port (Dycore_Port).
# Opt-in from the classic MPAS make build with `make ... DYCORE_CPP=true`.
# Delegates to the self-contained CMake project under ./cpp so the Kokkos,
# KokkosKernels, and Halo_Library dependencies are resolved from the
# Dev_Container toolchain (Requirements 13.3, 13.6, 1.6). RKIND is forwarded to
# the CMake option MPAS_DOUBLE_PRECISION so the C++ `Scalar` alias matches the
# Fortran working precision.
DYCORE_CPP ?= false
DYCORE_CPP_BUILD_DIR ?= $(CURDIR)/cpp/build
ifeq "$(PRECISION)" "single"
    DYCORE_CPP_DOUBLE = OFF
else
    DYCORE_CPP_DOUBLE = ON
endif

all: $(OBJS) dycore_cpp

mpas_atm_time_integration.o: mpas_atm_boundaries.o mpas_atm_iau.o mpas_atm_dissipation_models.o

mpas_atm_boundaries.o:

# Explicit rule to compile the Fortran shim (dycore_fortran_shim.F90) into
# mpas_dycore_interface.o. The module name is mpas_dycore_interface but the source
# file lives in cpp/src/ with a different filename.
mpas_dycore_interface.o: cpp/src/dycore_fortran_shim.F90
	$(RM) $@ mpas_dycore_interface.mod
	$(FC) $(CPPFLAGS) $(FFLAGS) -c $< -o $@ $(CPPINCLUDES) $(FCINCLUDES) -I.. -I../../framework

# Marshalling module depends on the interface module (.mod file produced by the shim)
mpas_dycore_marshalling.o: mpas_dycore_interface.o

.PHONY: dycore_cpp
dycore_cpp:
ifeq "$(DYCORE_CPP)" "true"
	cmake -S $(CURDIR)/cpp -B $(DYCORE_CPP_BUILD_DIR) \
	      -DMPAS_DOUBLE_PRECISION=$(DYCORE_CPP_DOUBLE)
	cmake --build $(DYCORE_CPP_BUILD_DIR)
else
	@echo "Skipping C++ dycore (set DYCORE_CPP=true to build it)"
endif

clean:
	$(RM) *.o *.mod *.f90
	@# Certain systems with intel compilers generate *.i files
	@# This removes them during the clean process
	$(RM) *.i
	$(RM) -r $(DYCORE_CPP_BUILD_DIR)

.F.o:
	$(RM) $@ $*.mod
ifeq "$(GEN_F90)" "true"
	$(CPP) $(CPPFLAGS) $(PHYSICS) $(CPPINCLUDES) -I../../framework $< > $*.f90
	$(FC) $(FFLAGS) -c $*.f90 $(FCINCLUDES) -I.. -I../../framework -I../../operators -I../physics -I../physics/physics_wrf -I../physics/physics_mmm $(MPAS_ESMF_INC)
else
	$(FC) $(CPPFLAGS) $(PHYSICS) $(FFLAGS) -c $*.F $(CPPINCLUDES) $(FCINCLUDES) -I.. -I../../framework -I../../operators -I../physics -I../physics/physics_wrf -I../physics/physics_mmm $(MPAS_ESMF_INC)
endif
