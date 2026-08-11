#pragma once

#include "allocator.hpp"

namespace llaisys::core::allocators {

class NaiveAllocator final : public MemoryAllocator {
public:
    explicit NaiveAllocator(const LlaisysRuntimeAPI *runtime_api);

    ~NaiveAllocator() override = default;

    std::byte *allocate(std::size_t size) override;

    void release(std::byte *memory) override;
};

} // namespace llaisys::core::allocators