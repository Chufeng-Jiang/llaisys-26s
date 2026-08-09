#include "naive_allocator.hpp"

namespace llaisys::core::allocators {

NaiveAllocator::NaiveAllocator(const LlaisysRuntimeAPI *runtime_api)
    : MemoryAllocator(runtime_api) {}

std::byte *NaiveAllocator::allocate(std::size_t size) {
    return static_cast<std::byte *>(_api->malloc_device(size));
}

void NaiveAllocator::release(std::byte *memory) { _api->free_device(memory); }

} // namespace llaisys::core::allocators