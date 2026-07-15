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
CPP_DYCORE_BUILD_DIR = $(CPP_DYCORE_DIR)/build
CPP_DYCORE_LIB = $(CPP_DYCORE_BUILD_DIR)/libmpas_dycore_cpp.a
CPP_DYCORE_INC = -I$(CPP_DYCORE_DIR)/include

$(info [cpp_dycore.mk] USE_CPP_DYCORE is set)
$(info [cpp_dycore.mk] CPP_DYCORE_DIR = $(CPP_DYCORE_DIR))
$(info [cpp_dycore.mk] CPP_DYCORE_LIB = $(CPP_DYCORE_LIB))

# Build the C++ library via CMake if not already built.
$(CPP_DYCORE_LIB):
	@echo "=== [cpp_dycore.mk] Building C++ dycore library ==="
	@echo "=== Source: $(CPP_DYCORE_DIR) ==="
	@echo "=== Build:  $(CPP_DYCORE_BUILD_DIR) ==="
	@echo "=== Target: $@ ==="
	cmake -S $(CPP_DYCORE_DIR) -B $(CPP_DYCORE_BUILD_DIR) \
	      -DCMAKE_BUILD_TYPE=Release \
	      -DMPAS_BUILD_TESTING=OFF
	cmake --build $(CPP_DYCORE_BUILD_DIR) -- -j4
	@test -f $@ || { echo "ERROR: $@ was not produced"; exit 1; }
	@echo "=== [cpp_dycore.mk] Library built successfully ==="

override CPPFLAGS += -DMPAS_CPP_DYCORE
FCINCLUDES += $(CPP_DYCORE_INC)

# The C++ dycore link flags are placed in CPP_DYCORE_LINK which is passed to
# the final link step. This avoids polluting LIBS which is used by prerequisite
# checks (e.g., PIO link test) that should not depend on the dycore.
CPP_DYCORE_LINK = $(CPP_DYCORE_LIB) -L$(CPP_DYCORE_BUILD_DIR) -L$(CPP_DYCORE_BUILD_DIR)/halo -lhalo -lhalo_c_interop -lstdc++ -lkokkoscore -lkokkoskernels

endif
