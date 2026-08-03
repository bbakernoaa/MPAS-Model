#pragma once

/// @file error.hpp
/// @brief Error handling utilities for the MPAS dynamical core C API.
///
/// Provides ErrorInfo for structured error reporting and api_wrap() for
/// converting C++ exceptions into C-compatible error codes at API boundaries.

#include <cstring>
#include <exception>

namespace mpas::dycore {

// ============================================================================
// ErrorInfo: structured error report
// ============================================================================

/// @brief Structured error information for the C API layer.
///
/// Contains a numeric error code and a null-terminated diagnostic message.
/// Code 0 indicates success; codes 1-255 indicate various error conditions.
struct ErrorInfo {
    int code;           ///< 0 = success, 1 = std::exception, 2 = unknown exception
    char message[512];  ///< Null-terminated diagnostic message
};

// ============================================================================
// api_wrap: exception-to-error-code converter
// ============================================================================

/// @brief Convert exceptions to C-compatible error codes at API boundaries.
///
/// Invokes the provided callable within a try/catch block. On success, returns 0.
/// On std::exception, writes the what() string to the error message buffer and
/// returns 1. On unknown exceptions, writes a generic message and returns 2.
///
/// @tparam Func  Callable type (typically a lambda).
/// @param errmsg     Buffer to write the error message into (may be nullptr).
/// @param errmsg_len Maximum number of characters to write (including null terminator).
/// @param func       Callable to execute; should return void.
/// @return 0 on success, 1 for std::exception, 2 for unknown exceptions.
///
/// Usage:
/// @code
///   return api_wrap(errmsg, errmsg_len, [&]() {
///       // C++ code that may throw
///   });
/// @endcode
template <typename Func>
int api_wrap(char* errmsg, int errmsg_len, Func&& func) noexcept {
    try {
        func();
        return 0;
    } catch (const std::exception& e) {
        if (errmsg != nullptr && errmsg_len > 0) {
            std::strncpy(errmsg, e.what(), static_cast<std::size_t>(errmsg_len - 1));
            errmsg[errmsg_len - 1] = '\0';
        }
        return 1;
    } catch (...) {
        if (errmsg != nullptr && errmsg_len > 0) {
            const char* msg = "unknown exception";
            std::strncpy(errmsg, msg, static_cast<std::size_t>(errmsg_len - 1));
            errmsg[errmsg_len - 1] = '\0';
        }
        return 2;
    }
}

} // namespace mpas::dycore
