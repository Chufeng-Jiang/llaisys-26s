#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();

    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;

    // 从最后一维开始，向前计算 strides
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i]; // 计算当前维度的元素数量，并更新 stride
    }

    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
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

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
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
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

/*
Check shape and strides of the tensor, and tell wether it is contiguous in memory.
*/
bool Tensor::isContiguous() const {
    ptrdiff_t expected_stride = 1; // The expected stride for the last dimension is always 1

    for (size_t i = this->ndim(); i > 0; --i) {
        const size_t dim = i - 1;

        if (this->shape()[dim] == 1) { // Skip dimensions of size 1, as they do not affect contiguity
            continue;
        }

        // Check if the actual stride matches the expected stride for this dimension
        if (this->strides()[dim] != expected_stride) {
            return false;
        }

        // Update the expected stride for the next dimension (moving backwards)
        expected_stride *= static_cast<ptrdiff_t>(this->shape()[dim]); // size_t to ptrdiff_t conversion is safe here because shape values are non-negative
    }

    return true;
}

/*
Create a new tensor which changes the order of the dimensions of original tensor. 
Transpose can be achieved by this function without moving data around.
*/
tensor_t Tensor::permute(const std::vector<size_t> &order) const {

    if (order.size() != this->ndim()) {
        throw std::runtime_error(
            "Permutation order must contain every dimension");
    }

    // Check for the dimension has been used or not
    std::vector<bool> seen(this->ndim(), false);

    std::vector<size_t> new_shape(this->ndim());

    // Initialize new strides for the permuted tensor. 
    //We must ensure that the new strides correspond to the new shape and the original tensor's strides.
    std::vector<ptrdiff_t> new_strides(this->ndim());

    for (size_t i = 0; i < order.size(); ++i) {
        const size_t dim = order[i];

        if (dim >= this->ndim()) {
            throw std::runtime_error(
                "Permutation dimension is out of range");
        }

        if (seen[dim]) {
            throw std::runtime_error(
                "Permutation dimensions must not be repeated");
        }

        seen[dim] = true;
        new_shape[i] = this->shape()[dim];
        new_strides[i] = this->strides()[dim];
    }

    TensorMeta meta{
        this->dtype(),
        std::move(new_shape),
        std::move(new_strides),
    };

    return std::shared_ptr<Tensor>(
        new Tensor(std::move(meta), _storage, _offset));
}

/*
Create a new tensor which reshapes the original tensor to the given shape by splitting or merging the original dimensions.
No data transfer is involved. For example change a tensor of shape (2, 3, 5) to (2, 15) by merging the last two dimensions.

This function is not as easy as simply changing the shape of the tensor, although the test will pass.
It should raise an error if new view is not compatible with the original tensor.
Think about a tensor of shape (2, 3, 5) and strides (30, 10, 1).
Can you still reshape it to (2, 15) without data transfer?
*/
tensor_t Tensor::view(const std::vector<size_t> &shape) const {

    // Check if the new shape has the same number of elements as the original tensor
    const size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());

    if (new_numel != this->numel()) {
        throw std::runtime_error(
            "View shape must have the same number of elements");
    }

    // Initialize new strides for the view tensor
    std::vector<ptrdiff_t> new_strides(shape.size());

    // Handle scalar tensors and empty tensors separately.
    if (this->ndim() == 0 || this->numel() == 0) {
        ptrdiff_t stride = 1;

        for (size_t i = shape.size(); i > 0; --i) {
            const size_t dim = i - 1;
            new_strides[dim] = stride;
            stride *= static_cast<ptrdiff_t>(shape[dim]);
        }

        TensorMeta meta{
            this->dtype(),
            shape,
            new_strides,
        };

        return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
    }

    // For non-scalar tensors, we need to check if the new shape is compatible with the original tensor's strides.
    // view_dim is the current dimension in the new view shape that we are trying to fill with strides.
    ptrdiff_t view_dim = static_cast<ptrdiff_t>(shape.size()) - 1;

    ptrdiff_t chunk_base_stride = this->strides().back(); // Last dimension's stride is the base stride for the current chunk of dimensions in the original tensor.

    size_t tensor_chunk_numel = 1; // The number of elements in the current chunk of dimensions in the original tensor.
    size_t view_chunk_numel = 1;   // The number of elements dispatched in the current chunk of dimensions.

    // Iterate over the original tensor's dimensions in reverse order to check compatibility with the new view shape.
    for (ptrdiff_t tensor_dim = static_cast<ptrdiff_t>(this->ndim()) - 1; tensor_dim >= 0; --tensor_dim) {

        // Update the number of elements in the current chunk of dimensions in the original tensor.
        tensor_chunk_numel *= this->shape()[static_cast<size_t>(tensor_dim)];

        // Case 1: if the current dimension is 0
        // Case 2: if the current dimension is not 0, but the next dimension is not 1 and the stride of the next dimension is not equal to the product of the number of elements in the current chunk and the base stride of the current chunk.
        // If the previous dimension is 1, we can skip it because it does not affect the contiguity of the tensor.
        const bool end_of_chunk = tensor_dim == 0 || 
                                  (this->shape()[static_cast<size_t>(tensor_dim - 1)] != 1 && 
                                  this->strides()[static_cast<size_t>(tensor_dim - 1)] != static_cast<ptrdiff_t>(tensor_chunk_numel) * chunk_base_stride);

        if (!end_of_chunk) {
            continue;
        }

        // Start filling the new strides for the view tensor from the current view dimension downwards, 
        // as long as the number of elements in the current chunk of dimensions in the view tensor is less than 
        // the number of elements in the current chunk of dimensions in the original tensor, 
        //or if the size of the current dimension in the view tensor is 1 (which can be broadcasted).
        while (view_dim >= 0 && 
               (view_chunk_numel < tensor_chunk_numel || shape[static_cast<size_t>(view_dim)] == 1)) {
            
            // Set the stride for the current view dimension based on the number of elements 
            //in the current chunk of dimensions in the view tensor and the base stride of the current chunk in the original tensor.
            new_strides[static_cast<size_t>(view_dim)] = static_cast<ptrdiff_t>(view_chunk_numel) * chunk_base_stride;

            // Update the number of elements in the current chunk of dimensions in the view tensor.
            view_chunk_numel *= shape[static_cast<size_t>(view_dim)];
            
            // Move to the next dimension in the view tensor.
            --view_dim;
        }

        // After filling the new strides for the view tensor, check if the number of elements in the current chunk of dimensions in the view tensor matches that of the original tensor.
        if (view_chunk_numel != tensor_chunk_numel) {
            throw std::runtime_error(
                "View shape is incompatible with tensor strides");
        }

        // Reset the number of elements in the current chunk of dimensions in the view tensor for the next chunk.
        if (tensor_dim > 0) {
            chunk_base_stride = this->strides()[static_cast<size_t>(tensor_dim - 1)];

            tensor_chunk_numel = 1;
            view_chunk_numel = 1;
        }
    }

    if (view_dim != -1) {
        throw std::runtime_error(
            "View shape is incompatible with tensor strides");
    }

    TensorMeta meta{
        this->dtype(),
        shape,
        new_strides,
    };

    return std::shared_ptr<Tensor>(
        new Tensor(std::move(meta), _storage, _offset));
}


/*
Create a new tensor which slices the original tensor along the given dimension, start (inclusive) and end (exclusive) indices.
*/
tensor_t Tensor::slice(
    size_t dim,
    size_t start,
    size_t end) const {

    if (dim >= this->ndim()) {
        throw std::runtime_error(
            "Slice dimension is out of range");
    }

    if (start > end) {
        throw std::runtime_error(
            "Slice start must not be greater than end");
    }

    if (end > this->shape()[dim]) {
        throw std::runtime_error(
            "Slice end is out of range");
    }

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
        new Tensor(
            std::move(meta),
            _storage,
            _offset + byte_offset));
}

/*
Load host (cpu) data to the tensor (can be on device).
Check contructor to see how to get runtime apis of the current device context,
and do a memcpy from host to device.
*/
void Tensor::load(const void *src_) {
    const size_t nbytes = this->numel() * this->elementSize();

    core::context().setDevice(
        this->deviceType(),
        this->deviceId());

    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(
            this->data(),
            src_,
            nbytes);
    } else {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            nbytes,
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
