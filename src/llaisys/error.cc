#include "llaisys.h"

#include "error.hpp"

#include <string>

namespace {

thread_local std::string last_error;

} // namespace

namespace llaisys::c_api {

void clear_last_error() noexcept {
    try {
        last_error.clear();
    } catch (...) {}
}

void set_last_error(const char *message) noexcept {
    try {
        last_error = message != nullptr ? message : "Unknown error";
    } catch (...) {}
}

void set_last_error(const std::string &message) noexcept {
    try {
        last_error = message;
    } catch (...) {}
}

const char *get_last_error() noexcept { return last_error.c_str(); }

} // namespace llaisys::c_api

__C __export const char *llaisysGetLastError() { return llaisys::c_api::get_last_error(); }