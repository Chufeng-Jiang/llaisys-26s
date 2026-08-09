#pragma once

#include "llaisys/runtime.h"

#include <cstddef>

namespace llaisys::core {

class MemoryAllocator {
protected:
    const LlaisysRuntimeAPI *_api;

    explicit MemoryAllocator(const LlaisysRuntimeAPI *runtime_api) : _api(runtime_api) {}

public:
    virtual ~MemoryAllocator() = default;

    MemoryAllocator(const MemoryAllocator &) = delete;

    MemoryAllocator &operator=(const MemoryAllocator &) = delete;

    MemoryAllocator(MemoryAllocator &&) = delete;

    MemoryAllocator &operator=(MemoryAllocator &&) = delete;

    virtual std::byte *allocate(std::size_t size) = 0;

    virtual void release(std::byte *memory) = 0;
};

} // namespace llaisys::core