#include "mpas_dycore/field_store.hpp"
#include "mpas_dycore/mesh_data.hpp"

namespace mpas {
namespace dycore {

// Explicit instantiation for the default execution space to verify the template
// compiles cleanly. Since Field_Store is header-only, this is primarily for
// early compile-error detection rather than link-time symbol generation.
template class Field_Store<Scalar, Kokkos::DefaultExecutionSpace>;

}  // namespace dycore
}  // namespace mpas
