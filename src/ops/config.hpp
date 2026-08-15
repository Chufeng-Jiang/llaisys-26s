#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llaisys::ops::config {

/**
 * @brief Reads an unsigned integer environment variable.
 *
 * The parser is intentionally backend-independent so operation-specific
 * configuration can be shared by every device backend.
 */
inline unsigned int get_unsigned(
    const char *name, unsigned int default_value, unsigned int minimum, unsigned int maximum) {
    const char *value = std::getenv(name);

    if (value == nullptr) { return default_value; }

    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);

    if (end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
        throw std::invalid_argument(
            std::string(name) + " must be between " + std::to_string(minimum) + " and "
            + std::to_string(maximum) + ".");
    }

    return static_cast<unsigned int>(parsed);
}

/**
 * @brief Reads a boolean environment variable.
 */
inline bool get_bool(const char *name, bool default_value = false) {
    const char *value = std::getenv(name);

    if (value == nullptr) { return default_value; }

    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0
        || std::strcmp(value, "on") == 0 || std::strcmp(value, "yes") == 0) {
        return true;
    }

    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0
        || std::strcmp(value, "off") == 0 || std::strcmp(value, "no") == 0) {
        return false;
    }

    throw std::invalid_argument(
        std::string(name) + " must be one of: 0, 1, false, true, off, on, no, yes.");
}

inline constexpr unsigned int DEFAULT_BLOCK_SIZE = 256;

inline unsigned int block_size() {
    static const unsigned int value
        = get_unsigned("LLAISYS_BLOCK_SIZE", DEFAULT_BLOCK_SIZE, 1, 1024);

    return value;
}

inline bool debug_enabled() { return get_bool("LLAISYS_DEBUG", false); }

} // namespace llaisys::ops::config