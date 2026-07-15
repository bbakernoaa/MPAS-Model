# cpp_dycore.mk — Optional build integration for the C++/Kokkos dynamical core.
#
# Include this file from the MPAS build system. When USE_CPP_DYCORE=true is set,
# the C++ dycore library is built via a CMake sub-build and linked into the
# MPAS executable. When USE_CPP_DYCORE is unset or false, this file is a no-op
# and no C++ compiler or Kokkos installation is required.
#
# Requirements: 5.1, 5.2, 5.3, 5.4, 5.5

ifdef USE_CPP_DYCORE

CPP_DYCORE_DIR = $(PWD)/src/core_atmosphere/dynamics/cpp
CPP_DYCORE_LIB = $(CPP_DYCORE_DIR)/build/libmpas_dycore_cpp.a
CPP_DYCORE_INC = -I$(CPP_DYCORE_DIR)/include

# Build the C++ library via CMake if not already built
$(CPP_DYCORE_LIB):
	cmake -S $(CPP_DYCORE_DIR) -B $(CPP_DYCORE_DIR)/build \
	      -DCMAKE_BUILD_TYPE=Release \
	      -DMPAS_BUILD_TESTING=OFF
	$(MAKE) -C $(CPP_DYCORE_DIR)/build

override CPPFLAGS += -DMPAS_CPP_DYCORE
FCINCLUDES += $(CPP_DYCORE_INC)

# The C++ dycore link flags are placed in CPP_DYCORE_LINK which is passed to
# the final link step via LDFLAGS. This avoids polluting LIBS which is used by
# prerequisite checks (e.g., PIO link test) that should not depend on the dycore.
CPP_DYCORE_LINK = $(CPP_DYCORE_LIB) -L$(CPP_DYCORE_DIR)/build -L$(CPP_DYCORE_DIR)/build/halo -lhalo -lhalo_c_interop -lstdc++ -lkokkoscore -lkokkoskernels

endif
