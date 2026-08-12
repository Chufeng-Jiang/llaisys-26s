#pragma once

#include <mcr/mc_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace llaisys::device::metax {

inline constexpr std::size_t METAX_BLOCK_SIZE = 256;
inline constexpr std::size_t METAX_DEFAULT_MAX_GRID_SIZE = 4096;

inline void checkMc(mcError_t status, const char *expression) {
    if (status == mcSuccess) { return; }

    throw std::runtime_error(
        std::string("MetaX MACA runtime call failed: ") + expression + " (error code "
        + std::to_string(static_cast<int>(status)) + ")");
}

} // namespace llaisys::device::metax

#define MC_CHECK(expression) ::llaisys::device::metax::checkMc((expression), #expression)