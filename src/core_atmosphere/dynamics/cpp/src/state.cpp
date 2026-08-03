/// @file state.cpp
/// @brief Implementation of the global DycoreState singleton accessor.

#include <mpas_dycore/state.hpp>

namespace mpas::dycore {

DycoreState& get_dycore_state() {
    static DycoreState instance;
    return instance;
}

} // namespace mpas::dycore
