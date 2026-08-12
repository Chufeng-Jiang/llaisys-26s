#pragma once
#include "llaisys.h"

#include "../core.hpp"

#include <memory>

namespace llaisys::core {

/**
 * @brief Manages a block of memory used to store tensor data.
 *
 * Storage represents the actual physical memory behind a Tensor.
 * A Tensor itself mainly describes the logical view of the data
 * (shape, strides, dtype, offset), while Storage owns and manages
 * the underlying memory block.
 *
 * For example:
 *
 *     Tensor
 *       |
 *       |-- _meta     -> shape / strides / dtype
 *       |
 *       |-- _storage  -> Storage
 *                         |
 *                         |-- _memory -> actual CPU/GPU memory
 *                         |-- _size   -> memory size in bytes
 *                         |-- _runtime-> runtime used to manage the memory
 *                         |-- _is_host-> whether this is host memory
 *
 * Multiple Tensor objects may share the same Storage. This is useful for
 * tensor views such as slice, reshape, or other operations that do not need
 * to copy the underlying data.
 *
 * The Storage constructor is private because memory objects should normally
 * be created by Runtime rather than directly by users. Runtime is declared
 * as a friend so that it can call the private constructor.
 */
class Storage {
private:
    /**
     * @brief Pointer to the beginning of the allocated memory block.
     *
     * The pointer may refer to CPU memory, pinned host memory, or device
     * memory depending on how the Storage was allocated.
     */
    std::byte *_memory;

    /**
     * @brief Size of the allocated memory block in bytes.
     */
    size_t _size;

    /**
     * @brief Runtime responsible for this memory allocation.
     *
     * The Runtime knows which backend/device is being used and is also
     * responsible for releasing the memory correctly.
     */
    Runtime &_runtime;

    /**
     * @brief Indicates whether the memory is host memory.
     *
     * true  -> host-side memory
     * false -> device-side memory
     */
    bool _is_host;

    /**
     * @brief Constructs a Storage object for an already allocated memory block.
     *
     * This constructor is private so Storage objects are created through
     * Runtime, which ensures that allocation and deallocation use the
     * correct device backend.
     *
     * @param memory Pointer to the allocated memory.
     * @param size Size of the memory block in bytes.
     * @param runtime Runtime that manages this memory.
     * @param is_host Whether the memory is host memory.
     */
    Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host);

public:
    /**
     * Runtime is allowed to access the private constructor of Storage.
     */
    friend class Runtime;

    /**
     * @brief Releases the memory owned by this Storage.
     *
     * The exact deallocation method depends on the Runtime and whether the
     * memory is host or device memory.
     */
    ~Storage();

    /**
     * @brief Returns the starting address of the underlying memory block.
     */
    std::byte *memory() const;

    /**
     * @brief Returns the size of the storage in bytes.
     */
    size_t size() const;

    /**
     * @brief Returns the device type associated with this storage.
     *
     * Examples may include CPU, NVIDIA GPU, or MetaX GPU.
     */
    llaisysDeviceType_t deviceType() const;

    /**
     * @brief Returns the ID of the device that owns this storage.
     *
     * For example, GPU 0 or GPU 1.
     */
    int deviceId() const;

    /**
     * @brief Returns whether this storage represents host memory.
     */
    bool isHost() const;
};

}; // namespace llaisys::core
