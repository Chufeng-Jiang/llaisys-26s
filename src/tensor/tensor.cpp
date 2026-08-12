#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "../ops/rearrange/op.hpp"
#include "../utils.hpp"
#include "tensor.hpp"

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(
    const std::vector<size_t> &shape,
    llaisysDataType_t dtype,
    llaisysDeviceType_t device_type,
    int device) {
    size_t ndim_ = shape.size();

    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;

    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }

    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU
        && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() { return _storage->memory() + _offset; }

const std::byte *Tensor::data() const { return _storage->memory() + _offset; }

size_t Tensor::ndim() const { return _meta.shape.size(); }

const std::vector<size_t> &Tensor::shape() const { return _meta.shape; }

const std::vector<ptrdiff_t> &Tensor::strides() const { return _meta.strides; }

llaisysDataType_t Tensor::dtype() const { return _meta.dtype; }

llaisysDeviceType_t Tensor::deviceType() const { return _storage->deviceType(); }

int Tensor::deviceId() const { return _storage->deviceId(); }

size_t Tensor::numel() const {
    return std::accumulate(
        _meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const { return utils::dsize(_meta.dtype); }

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: " << "shape[ ";
    for (auto s : this->shape()) { ss << s << " "; }
    ss << "] strides[ ";
    for (auto s : this->strides()) { ss << s << " "; }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(
    const T *data,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &strides,
    size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(
    const std::byte *data,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &strides,
    llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(), this->data(), this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

/**
 * @brief Determines whether tensor elements are stored in contiguous memory.
 *
 * Starting from the last dimension, the expected stride is initially 1.
 * For each non-singleton dimension, the actual stride must equal the
 * expected stride. The expected stride is then multiplied by the current
 * dimension size before checking the next dimension.
 *
 * For example, a tensor with shape [2, 3, 4] is contiguous when its strides
 * are [12, 4, 1].
 *
 * Dimensions of size 1 are skipped because changing their stride does not
 * change the physical memory layout.
 *
 * @return true if the tensor has a contiguous layout, false otherwise.
 */
bool Tensor::isContiguous() const {
    const auto &shape = _meta.shape;
    const auto &strides = _meta.strides;

    ASSERT(
        shape.size() == strides.size(),
        "Shape and strides must have the same number of dimensions");

    if (this->numel() == 0) { return true; }

    ptrdiff_t expected_stride = 1; // last dimension

    for (size_t i = shape.size(); i > 0; --i) {
        const size_t dim = i - 1;
        if (shape[dim] == 1) { continue; }
        if (strides[dim] != expected_stride) { return false; }
        expected_stride *= static_cast<ptrdiff_t>(shape[dim]);
    }

    return true;
}

/**
 * @brief Returns a new tensor view with its dimensions reordered.
 *
 * This operation does not copy or rearrange the underlying tensor data.
 * Instead, it creates a new TensorMeta by permuting the tensor's shape
 * and strides according to the given order, while sharing the same
 * Storage and offset with the original tensor.
 *
 * The `order` vector describes which original dimension becomes each
 * output dimension.
 *
 * Example:
 *
 *     Original tensor:
 *         shape   = [2, 3, 4]
 *         strides = [12, 4, 1]
 *
 *     permute({1, 0, 2})
 *
 *     means:
 *         output dim 0 <- original dim 1
 *         output dim 1 <- original dim 0
 *         output dim 2 <- original dim 2
 *
 *     New tensor:
 *         shape   = [3, 2, 4]
 *         strides = [4, 12, 1]
 *
 * Both tensors still refer to the same underlying memory. Only the
 * logical interpretation of that memory changes.
 *
 * A valid permutation must:
 *   1. Have exactly the same number of dimensions as the tensor.
 *   2. Contain only valid dimension indices.
 *   3. Contain each dimension exactly once.
 *
 * Since permuting dimensions often changes the stride pattern, the
 * returned tensor may no longer be contiguous even if the original
 * tensor was contiguous.
 *
 * @param order The new dimension order.
 * @return A new Tensor sharing the same storage and offset, but with
 *         permuted shape and strides.
 */
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const size_t ndim = this->ndim();

    CHECK_ARGUMENT(order.size() == ndim, "Permutation order must contain every dimension");

    std::vector<bool> seen(ndim, false);
    std::vector<size_t> new_shape(ndim);
    std::vector<ptrdiff_t> new_strides(ndim);

    for (size_t output_dim = 0; output_dim < ndim; ++output_dim) {
        const size_t input_dim = order[output_dim];

        CHECK_ARGUMENT(input_dim < ndim, "Permutation dimension is out of range");
        CHECK_ARGUMENT(!seen[input_dim], "Permutation dimensions must not be repeated");

        seen[input_dim] = true;
        new_shape[output_dim] = this->shape()[input_dim];
        new_strides[output_dim] = this->strides()[input_dim];
    }

    TensorMeta meta{
        this->dtype(),
        std::move(new_shape),
        std::move(new_strides),
    };

    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), this->_storage, this->_offset));
}

/**
 * @brief Creates a new tensor view with a different shape.
 *
 * view() changes only the tensor metadata (shape and strides).
 * It does not copy, move, or rearrange the underlying tensor data.
 *
 * Because no data rearrangement is performed, the original tensor must
 * be contiguous in memory. Otherwise, simply assigning a new contiguous
 * stride pattern could change the logical-to-physical memory mapping.
 *
 * The new shape must contain exactly the same number of elements as the
 * original tensor.
 *
 * Example:
 *
 *     Original tensor:
 *         shape   = [2, 3, 4]
 *         strides = [12, 4, 1]
 *         numel   = 24
 *
 *     view({6, 4})
 *
 *     New tensor:
 *         shape   = [6, 4]
 *         strides = [4, 1]
 *         numel   = 24
 *
 * Both tensors share the same underlying Storage and offset:
 *
 *     Original Tensor ──┐
 *                       ├──> same Storage
 *     Viewed Tensor   ──┘
 *
 * Only the logical interpretation of the same contiguous memory changes.
 *
 * The new strides are generated using the standard row-major contiguous
 * layout. Starting from the last dimension, its stride is 1. Moving toward
 * the first dimension, each stride equals the product of the sizes of all
 * dimensions to its right.
 *
 * @param shape The desired new shape.
 * @return A new Tensor sharing the same storage and offset, but using the
 *         requested shape and newly computed contiguous strides.
 */
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    CHECK_ARGUMENT(this->isContiguous(), "View requires a contiguous tensor");

    const size_t new_numel
        = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());

    CHECK_ARGUMENT(new_numel == this->numel(), "View shape must have the same number of elements");

    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t expected_stride = 1;

    for (size_t i = shape.size(); i > 0; --i) {
        const size_t dim = i - 1;
        new_strides[dim] = expected_stride;
        expected_stride *= static_cast<ptrdiff_t>(shape[dim]);
    }

    TensorMeta meta{
        this->dtype(),
        shape,
        std::move(new_strides),
    };

    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), this->_storage, this->_offset));
}

/**
 * @brief Creates a tensor view representing a slice of one dimension.
 *
 * slice() selects the range [start, end) along the specified dimension.
 * The `start` index is included, while the `end` index is excluded.
 *
 * This operation does not copy or rearrange the underlying tensor data.
 * Instead, it:
 *
 *   1. Changes the size of the sliced dimension.
 *   2. Keeps the original strides unchanged.
 *   3. Moves the tensor's byte offset to the first element of the slice.
 *   4. Shares the same underlying Storage with the original tensor.
 *
 * Example:
 *
 *     Original tensor:
 *         shape   = [2, 3, 4]
 *         strides = [12, 4, 1]
 *
 *     slice(dim=1, start=1, end=3)
 *
 *     The second dimension changes from size 3 to size 2:
 *
 *         new_shape = [2, 2, 4]
 *
 *     Since stride[1] = 4, moving from index 0 to index 1 along
 *     dimension 1 skips 4 elements.
 *
 *         element_offset = start * stride[1]
 *                        = 1 * 4
 *                        = 4 elements
 *
 *     If the tensor stores float32 values:
 *
 *         elementSize() = 4 bytes
 *
 *     then:
 *
 *         byte_offset = 4 * 4
 *                     = 16 bytes
 *
 *     Therefore, the new tensor starts 16 bytes after the original
 *     tensor's current offset.
 *
 * The resulting tensor still shares the same Storage:
 *
 *     Original Tensor ──────┐
 *                           ├──> same Storage
 *     Sliced Tensor   ──────┘
 *
 * Only the logical starting position and shape are changed.
 *
 * This implementation supports forward slices only. Negative strides
 * are rejected because they would require signed offset handling.
 *
 * @param dim   Dimension along which to slice.
 * @param start Inclusive starting index.
 * @param end   Exclusive ending index.
 * @return A new Tensor sharing the same storage and strides, with an
 *         updated shape and byte offset.
 */
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < this->ndim(), "Slice dimension is out of range");
    CHECK_ARGUMENT(start <= end, "Slice start must not be greater than end");
    CHECK_ARGUMENT(end <= this->shape()[dim], "Slice end is out of range");
    CHECK_ARGUMENT(this->strides()[dim] >= 0, "Slice does not support negative strides");

    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;
    const ptrdiff_t element_offset = static_cast<ptrdiff_t>(start) * this->strides()[dim];
    const size_t byte_offset = static_cast<size_t>(element_offset) * this->elementSize();

    TensorMeta meta{
        this->dtype(),
        std::move(new_shape),
        this->strides(),
    };

    return std::shared_ptr<Tensor>(
        new Tensor(std::move(meta), this->_storage, this->_offset + byte_offset));
}

/**
 * @brief Loads data from a caller-provided host buffer into this tensor.
 *
 * The source pointer `src_` is expected to point to a contiguous block
 * of host memory containing exactly the logical elements of this tensor.
 *
 * This function copies:
 *
 *     numel() * elementSize()
 *
 * bytes into the tensor starting at `data()`.
 *
 * Note that `data()` already includes the tensor's offset inside the
 * shared Storage:
 *
 *     data() = _storage->memory() + _offset
 *
 * Therefore, if this Tensor is a slice or view with a non-zero offset,
 * the copied data starts from that tensor's logical starting position
 * rather than from the beginning of the entire Storage.
 *
 * The tensor must be contiguous because this implementation performs
 * one single memory copy. A non-contiguous tensor may contain gaps
 * between logical elements and would require a stride-aware copy.
 *
 * Memory transfer depends on the tensor's device:
 *
 *     CPU tensor:
 *         host memory -> host memory
 *         std::memcpy()
 *
 *     Accelerator tensor:
 *         host memory -> device memory
 *         runtime memcpy_sync(..., LLAISYS_MEMCPY_H2D)
 *
 * Before performing an accelerator copy, the runtime is switched to
 * the tensor's device if necessary.
 *
 * @param src_ Pointer to the source buffer in host memory.
 */
void Tensor::load(const void *src_) {
    CHECK_ARGUMENT(src_ != nullptr, "Source pointer is null");
    ASSERT(_storage != nullptr, "Tensor storage is null");
    CHECK_ARGUMENT(this->data() != nullptr, "Tensor data pointer is null");
    CHECK_ARGUMENT(this->isContiguous(), "Tensor must be contiguous when loading data");

    const size_t nbytes = this->numel() * this->elementSize();

    if (nbytes == 0) { return; }

    const auto device_type = this->deviceType();
    const int device_id = this->deviceId();
    auto &runtime = core::context().runtime();

    if (runtime.deviceType() != device_type || runtime.deviceId() != device_id) {
        core::context().setDevice(device_type, device_id);
    }

    if (device_type == LLAISYS_DEVICE_CPU) {
        std::memcpy(this->data(), src_, nbytes);
    } else {
        core::context().runtime().api()->memcpy_sync(
            this->data(), src_, nbytes, LLAISYS_MEMCPY_H2D);
    }
}

/**
 * @brief Returns a tensor whose elements are stored contiguously in memory.
 *
 * If the current tensor is already contiguous, no data rearrangement is
 * necessary. A new Tensor object is created that shares the same metadata,
 * Storage, and byte offset with the original tensor.
 *
 * If the current tensor is non-contiguous, a new contiguous tensor is
 * allocated with the same shape, dtype, and device. The elements are then
 * copied from the original tensor according to its strides and written into
 * the new tensor using the standard contiguous row-major layout.
 *
 * Example:
 *
 *     Original tensor after permute:
 *
 *         shape   = [3, 2, 4]
 *         strides = [4, 12, 1]
 *
 *     This tensor is non-contiguous because a contiguous tensor with
 *     shape [3, 2, 4] should have:
 *
 *         strides = [8, 4, 1]
 *
 *     Calling:
 *
 *         tensor->contiguous()
 *
 *     allocates a new tensor:
 *
 *         shape   = [3, 2, 4]
 *         strides = [8, 4, 1]
 *
 *     rearrange() then reads the original tensor according to its
 *     non-contiguous strides [4, 12, 1] and writes the elements into
 *     the new contiguous memory layout.
 *
 * In summary:
 *
 *     Already contiguous:
 *         same Storage
 *         same offset
 *         no data copy
 *
 *     Non-contiguous:
 *         new Storage
 *         new contiguous strides
 *         data rearrangement required
 *
 * @return A Tensor with contiguous memory layout.
 */
tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(this->_meta, this->_storage, this->_offset));
    }

    tensor_t contiguous_tensor
        = Tensor::create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());

    tensor_t source_tensor
        = std::shared_ptr<Tensor>(new Tensor(this->_meta, this->_storage, this->_offset));

    llaisys::ops::rearrange(contiguous_tensor, source_tensor);

    return contiguous_tensor;
}

/**
 * @brief Returns a tensor with the requested shape.
 *
 * reshape() changes the logical shape of a tensor while preserving
 * the same element order and total number of elements.
 *
 * If the current tensor is already contiguous, reshape() can directly
 * use view(), because the underlying data is already stored as one
 * continuous row-major sequence.
 *
 * If the current tensor is non-contiguous, reshape() first calls
 * contiguous() to rearrange the elements into a new contiguous memory
 * layout, and then calls view() to assign the requested shape.
 *
 * Therefore:
 *
 *     contiguous tensor:
 *         reshape(shape)
 *             -> view(shape)
 *             -> no data copy
 *
 *     non-contiguous tensor:
 *         reshape(shape)
 *             -> contiguous()
 *             -> data rearrangement / copy
 *             -> view(shape)
 *
 * @param shape The desired new shape. It must contain the same number
 *              of elements as the original tensor.
 * @return A tensor with the requested shape.
 */
tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    if (this->isContiguous()) { return this->view(shape); }

    return this->contiguous()->view(shape);
}

/**
 * @brief Moves a tensor to the specified device.
 *
 * to() returns a tensor located on the requested device type and device ID.
 *
 * If the tensor is already on the requested device, no memory transfer is
 * performed. A new Tensor object is returned that shares the same metadata,
 * Storage, and byte offset with the original tensor.
 *
 * If the target device is different, the source tensor is first converted
 * to a contiguous layout. This is necessary because the implementation uses
 * a single flat memory copy, which requires all logical tensor elements to
 * occupy one continuous memory region.
 *
 * A new contiguous tensor is then allocated on the destination device with
 * the same shape and dtype as the source tensor.
 *
 * The memory copy direction depends on the source and destination devices:
 *
 *     CPU -> CPU
 *         Host-to-host copy using std::memcpy().
 *
 *     CPU -> Accelerator
 *         Host-to-device copy (H2D).
 *
 *     Accelerator -> CPU
 *         Device-to-host copy (D2H).
 *
 *     Accelerator -> Accelerator
 *         Device-to-device copy (D2D).
 *
 * For a non-contiguous tensor, such as a tensor produced by permute(),
 * contiguous() first rearranges the logical elements into a dense
 * row-major memory layout before the device transfer is performed.
 *
 * @param device_type Target device type, such as CPU, NVIDIA, or MetaX.
 * @param device_id   Target device ID, such as GPU 0 or GPU 1.
 * @return A tensor located on the requested device.
 */
tensor_t Tensor::to(llaisysDeviceType_t device_type, int device_id) const {
    const auto source_device_type = this->deviceType();
    const int source_device_id = this->deviceId();

    if (device_type == source_device_type && device_id == source_device_id) {
        return std::shared_ptr<Tensor>(new Tensor(this->_meta, this->_storage, this->_offset));
    }

    tensor_t source = this->contiguous();
    tensor_t destination = Tensor::create(source->shape(), source->dtype(), device_type, device_id);
    const size_t total_bytes = source->numel() * source->elementSize();

    if (total_bytes == 0) { return destination; }

    if (source_device_type == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        std::memcpy(destination->data(), source->data(), total_bytes);
    } else if (source_device_type == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        core::context().setDevice(device_type, device_id);
        core::context().runtime().api()->memcpy_sync(
            destination->data(), source->data(), total_bytes, LLAISYS_MEMCPY_H2D);
    } else if (source_device_type != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        core::context().setDevice(source_device_type, source_device_id);
        core::context().runtime().api()->memcpy_sync(
            destination->data(), source->data(), total_bytes, LLAISYS_MEMCPY_D2H);
    } else {
        core::context().setDevice(source_device_type, source_device_id);
        core::context().runtime().api()->memcpy_sync(
            destination->data(), source->data(), total_bytes, LLAISYS_MEMCPY_D2D);
    }

    return destination;
}

} // namespace llaisys
