#include "qwen2_workspace.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

Qwen2Workspace::Qwen2Workspace(llaisysDeviceType_t device, int device_id)
    : device_(device), device_id_(device_id) {}

std::size_t Qwen2Workspace::checked_numel(const std::vector<std::size_t> &shape) {
    std::size_t numel = 1;

    for (const std::size_t dimension : shape) {
        if (dimension == 0) { return 0; }

        if (numel > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("Qwen2 workspace tensor size overflow.");
        }

        numel *= dimension;
    }

    return numel;
}

std::size_t Qwen2Workspace::grow_capacity(std::size_t current, std::size_t required) {
    if (required == 0) { return 0; }

    if (current >= required) { return current; }

    std::size_t capacity = current == 0 ? required : current;

    while (capacity < required) {
        if (capacity > std::numeric_limits<std::size_t>::max() / 2) { return required; }

        capacity *= 2;
    }

    return capacity;
}

llaisys::tensor_t Qwen2Workspace::get(
    Qwen2WorkspaceSlot slot, const std::vector<std::size_t> &shape, llaisysDataType_t dtype) {
    const std::size_t index = static_cast<std::size_t>(slot);

    if (index >= entries_.size()) {
        throw std::out_of_range("Qwen2 workspace slot is out of range.");
    }

    const std::size_t required_elements = checked_numel(shape);

    if (required_elements == 0) {
        throw std::invalid_argument(
            "Qwen2 workspace does not allocate zero-sized scratch tensors.");
    }

    auto &entry = entries_[index];

    const bool dtype_changed = entry.initialized && entry.dtype != dtype;

    if (!entry.initialized || dtype_changed || entry.capacity_elements < required_elements) {
        const std::size_t new_capacity
            = dtype_changed ? required_elements
                            : grow_capacity(entry.capacity_elements, required_elements);

        entry.backing = llaisys::Tensor::create({new_capacity}, dtype, device_, device_id_);

        entry.capacity_elements = new_capacity;

        entry.dtype = dtype;
        entry.initialized = true;
    }

    // The backing tensor is deliberately 1-D and sized by capacity.
    // Create a prefix view containing exactly the requested number of
    // elements, then reinterpret that contiguous prefix with the requested
    // shape. slice()/view() share storage and therefore do not allocate
    // another CPU/GPU data buffer.
    auto prefix = entry.backing->slice(0, 0, required_elements);

    return prefix->view(shape);
}

std::vector<std::int64_t> &Qwen2Workspace::position_values(std::size_t size) {
    position_values_.resize(size);

    return position_values_;
}
