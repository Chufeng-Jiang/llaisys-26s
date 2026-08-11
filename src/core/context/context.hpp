#pragma once

#include "llaisys.h"

#include "../runtime/runtime.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace llaisys::core {

class Context {
private:
    std::unordered_map<llaisysDeviceType_t, std::vector<std::unique_ptr<Runtime>>> _runtime_map;

    Runtime *_current_runtime{nullptr};

    Runtime *getOrCreateRuntime(llaisysDeviceType_t device_type, int device_id);

    Context();

public:
    ~Context() = default;

    Context(const Context &) = delete;

    Context &operator=(const Context &) = delete;

    Context(Context &&) = delete;

    Context &operator=(Context &&) = delete;

    void setDevice(llaisysDeviceType_t device_type, int device_id);

    Runtime &runtime();

    friend Context &context();
};

} // namespace llaisys::core