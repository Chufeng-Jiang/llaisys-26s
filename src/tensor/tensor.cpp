#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "../utils.hpp"
#include "tensor.hpp"

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type, int device) {
  size_t ndim_ = shape.size();

  std::vector<ptrdiff_t> strides(ndim_);
  size_t stride = 1;

  // 从最后一维开始，向前计算 strides
  for (size_t i = 1; i <= ndim_; i++) {
    strides[ndim_ - i] = stride;
    stride *= shape[ndim_ - i];  // 计算当前维度的元素数量，并更新 stride
  }

  TensorMeta meta{dtype, shape, strides};
  size_t total_elems = stride;
  size_t dtype_size = utils::dsize(dtype);

  if (device_type == LLAISYS_DEVICE_CPU &&
      core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
    auto storage =
        core::context().runtime().allocateHostStorage(total_elems * dtype_size);
    return std::shared_ptr<Tensor>(new Tensor(meta, storage));
  } else {
    core::context().setDevice(device_type, device);
    auto storage = core::context().runtime().allocateDeviceStorage(total_elems *
                                                                   dtype_size);
    return std::shared_ptr<Tensor>(new Tensor(meta, storage));
  }
}

std::byte *Tensor::data() { return _storage->memory() + _offset; }

const std::byte *Tensor::data() const { return _storage->memory() + _offset; }

size_t Tensor::ndim() const { return _meta.shape.size(); }

const std::vector<size_t> &Tensor::shape() const { return _meta.shape; }

const std::vector<ptrdiff_t> &Tensor::strides() const { return _meta.strides; }

llaisysDataType_t Tensor::dtype() const { return _meta.dtype; }

llaisysDeviceType_t Tensor::deviceType() const {
  return _storage->deviceType();
}

int Tensor::deviceId() const { return _storage->deviceId(); }

size_t Tensor::numel() const {
  return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1),
                         std::multiplies<size_t>());
}

size_t Tensor::elementSize() const { return utils::dsize(_meta.dtype); }

std::string Tensor::info() const {
  std::stringstream ss;

  ss << "Tensor: " << "shape[ ";
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
void print_data(const T *data, const std::vector<size_t> &shape,
                const std::vector<ptrdiff_t> &strides, size_t dim) {
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

void debug_print(const std::byte *data, const std::vector<size_t> &shape,
                 const std::vector<ptrdiff_t> &strides,
                 llaisysDataType_t dtype) {
  switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
      return print_data(reinterpret_cast<const char *>(data), shape, strides,
                        0);
    case LLAISYS_DTYPE_BOOL:
      return print_data(reinterpret_cast<const bool *>(data), shape, strides,
                        0);
    case LLAISYS_DTYPE_I8:
      return print_data(reinterpret_cast<const int8_t *>(data), shape, strides,
                        0);
    case LLAISYS_DTYPE_I16:
      return print_data(reinterpret_cast<const int16_t *>(data), shape, strides,
                        0);
    case LLAISYS_DTYPE_I32:
      return print_data(reinterpret_cast<const int32_t *>(data), shape, strides,
                        0);
    case LLAISYS_DTYPE_I64:
      return print_data(reinterpret_cast<const int64_t *>(data), shape, strides,
                        0);
    case LLAISYS_DTYPE_U8:
      return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides,
                        0);
    case LLAISYS_DTYPE_U16:
      return print_data(reinterpret_cast<const uint16_t *>(data), shape,
                        strides, 0);
    case LLAISYS_DTYPE_U32:
      return print_data(reinterpret_cast<const uint32_t *>(data), shape,
                        strides, 0);
    case LLAISYS_DTYPE_U64:
      return print_data(reinterpret_cast<const uint64_t *>(data), shape,
                        strides, 0);
    case LLAISYS_DTYPE_F16:
      return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides,
                        0);
    case LLAISYS_DTYPE_F32:
      return print_data(reinterpret_cast<const float *>(data), shape, strides,
                        0);
    case LLAISYS_DTYPE_F64:
      return print_data(reinterpret_cast<const double *>(data), shape, strides,
                        0);
    case LLAISYS_DTYPE_BF16:
      return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides,
                        0);
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
    debug_print(tmp_tensor->data(), this->shape(), this->strides(),
                this->dtype());
  }
}

bool Tensor::isContiguous() const {
  const auto &shape = _meta.shape;
  const auto &strides = _meta.strides;

  // Shape and stride metadata must describe the same number of dimensions.
  ASSERT(shape.size() == strides.size(),
         "Shape and strides must have the same number of dimensions");

  // Empty tensors have no elements whose physical layout can violate
  // contiguity, so they are conventionally treated as contiguous.
  if (this->numel() == 0) {
    return true;
  }

  ptrdiff_t expected_stride = 1;

  // Traverse dimensions from the innermost dimension to the outermost
  // dimension, following the standard row-major memory layout.
  for (size_t i = shape.size(); i > 0; --i) {
    const size_t dim = i - 1;

    // A dimension of size 1 has only index 0, so its stride does not
    // affect the actual memory addresses accessed by the tensor.
    if (shape[dim] == 1) {
      continue;
    }

    // A dense row-major tensor must use the expected stride for every
    // non-singleton dimension.
    if (strides[dim] != expected_stride) {
      return false;
    }

    // The next outer dimension skips all elements contained in the
    // current dimension.
    expected_stride *= static_cast<ptrdiff_t>(shape[dim]);
  }

  return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
  const size_t ndim = this->ndim();

  // A valid permutation must specify exactly one entry
  // for every dimension of the tensor.
  CHECK_ARGUMENT(order.size() == ndim,
                 "Permutation order must contain every dimension");

  std::vector<bool> seen(ndim, false);
  std::vector<size_t> new_shape(ndim);
  std::vector<ptrdiff_t> new_strides(ndim);

  for (size_t output_dim = 0; output_dim < ndim; ++output_dim) {
    // order[output_dim] identifies which original dimension
    // becomes this output dimension.
    const size_t input_dim = order[output_dim];

    CHECK_ARGUMENT(input_dim < ndim, "Permutation dimension is out of range");

    // Each original dimension must appear exactly once.
    CHECK_ARGUMENT(!seen[input_dim],
                   "Permutation dimensions must not be repeated");

    seen[input_dim] = true;

    // Permute both shape and stride together so that the new
    // tensor preserves the original mapping to physical memory.
    new_shape[output_dim] = this->shape()[input_dim];
    new_strides[output_dim] = this->strides()[input_dim];
  }

  TensorMeta meta{
      this->dtype(),
      std::move(new_shape),
      std::move(new_strides),
  };

  return std::shared_ptr<Tensor>(
      new Tensor(std::move(meta), this->_storage, this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
  // view() only changes metadata and does not rearrange data.
  // Therefore, the original tensor must have a dense memory layout.
  CHECK_ARGUMENT(this->isContiguous(), "View requires a contiguous tensor");

  // The new shape must describe exactly the same number of elements.
  const size_t new_numel = std::accumulate(
      shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());

  CHECK_ARGUMENT(new_numel == this->numel(),
                 "View shape must have the same number of elements");

  // Generate standard row-major strides for the new shape.
  std::vector<ptrdiff_t> new_strides(shape.size());

  ptrdiff_t expected_stride = 1;

  for (size_t i = shape.size(); i > 0; --i) {
    const size_t dim = i - 1;

    new_strides[dim] = expected_stride;

    // Each outer dimension skips all elements contained
    // in the dimensions to its right.
    expected_stride *= static_cast<ptrdiff_t>(shape[dim]);
  }

  TensorMeta meta{
      this->dtype(),
      shape,
      std::move(new_strides),
  };

  return std::shared_ptr<Tensor>(
      new Tensor(std::move(meta), this->_storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
  // The sliced dimension must exist in the tensor.
  CHECK_ARGUMENT(dim < this->ndim(), "Slice dimension is out of range");

  // Allow start == end so that an empty slice can be represented.
  CHECK_ARGUMENT(start <= end, "Slice start must not be greater than end");

  // The end index is exclusive and may equal the dimension size.
  CHECK_ARGUMENT(end <= this->shape()[dim], "Slice end is out of range");

  // This implementation assumes that strides describe forward memory
  // traversal. Negative strides would require signed offset handling.
  CHECK_ARGUMENT(this->strides()[dim] >= 0,
                 "Slice does not support negative strides");

  // Only the size of the selected dimension changes.
  // All other dimensions retain their original sizes.
  std::vector<size_t> new_shape = this->shape();
  new_shape[dim] = end - start;

  // Calculate how many elements must be skipped along the selected
  // dimension before reaching the first element of the slice.
  const ptrdiff_t element_offset =
      static_cast<ptrdiff_t>(start) * this->strides()[dim];

  // Tensor offsets are stored in bytes, so convert the element offset
  // using the size of one tensor element.
  const size_t byte_offset =
      static_cast<size_t>(element_offset) * this->elementSize();

  TensorMeta meta{
      this->dtype(),
      std::move(new_shape),
      this->strides(),
  };

  return std::shared_ptr<Tensor>(
      new Tensor(std::move(meta), this->_storage, this->_offset + byte_offset));
}

void Tensor::load(const void *src_) {
  // The source buffer is provided by the caller and must be valid.
  CHECK_ARGUMENT(src_ != nullptr, "Source pointer is null");

  // Every valid tensor should own or reference an allocated storage object.
  ASSERT(_storage != nullptr, "Tensor storage is null");

  // data() includes the tensor's offset inside the underlying storage.
  // This is important for sliced tensors or tensor views.
  CHECK_ARGUMENT(this->data() != nullptr, "Tensor data pointer is null");

  // A single memcpy assumes that logical tensor elements are stored
  // consecutively in memory. Non-contiguous tensors require strided copying.
  CHECK_ARGUMENT(this->isContiguous(),
                 "Tensor must be contiguous when loading data");

  // Copy only the bytes belonging to this tensor, rather than the entire
  // underlying storage, which may be shared with another tensor or view.
  const size_t nbytes = this->numel() * this->elementSize();

  // Nothing needs to be copied for an empty tensor.
  if (nbytes == 0) {
    return;
  }

  const auto device_type = this->deviceType();
  const int device_id = this->deviceId();

  auto &runtime = core::context().runtime();

  // Switch the active runtime device only when it does not already match
  // the device on which this tensor is allocated.
  if (runtime.deviceType() != device_type || runtime.deviceId() != device_id) {
    core::context().setDevice(device_type, device_id);
  }

  if (device_type == LLAISYS_DEVICE_CPU) {
    // CPU tensor memory is directly accessible from the host.
    std::memcpy(this->data(), src_, nbytes);
  } else {
    // For accelerator tensors, use the runtime API to perform a
    // synchronous host-to-device memory transfer.
    core::context().runtime().api()->memcpy_sync(this->data(), src_, nbytes,
                                                 LLAISYS_MEMCPY_H2D);
  }
}

tensor_t Tensor::contiguous() const {
  // if (this->isContiguous()) {
  //   return std::shared_ptr<Tensor>(
  //       new Tensor(this->_meta, this->_storage, this->_offset));
  // }

  // tensor_t contiguous_tensor = Tensor::create(
  //     this->shape(), this->dtype(), this->deviceType(), this->deviceId());

  // // Create a wrapper for the current tensor so it can be passed to
  // // rearrange(). The wrapper preserves the original shape, strides,
  // // storage, and byte offset.
  // tensor_t source_tensor = std::shared_ptr<Tensor>(
  //     new Tensor(this->_meta, this->_storage, this->_offset));

  // // Copy elements according to the original strides and store them
  // // in standard contiguous row-major order.
  // llaisys::ops::rearrange(contiguous_tensor, source_tensor);

  // return contiguous_tensor;

  return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
  if (this->isContiguous()) {
    return this->view(shape);
  }

  return this->contiguous()->view(shape);
}

// tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
tensor_t Tensor::to(llaisysDeviceType_t device_type, int device_id) const {
  const auto source_device_type = this->deviceType();
  const int source_device_id = this->deviceId();

  // No transfer is needed when the requested device already matches.
  // Return another tensor wrapper sharing the same storage and offset.
  if (device_type == source_device_type && device_id == source_device_id) {
    return std::shared_ptr<Tensor>(
        new Tensor(this->_meta, this->_storage, this->_offset));
  }

  // A raw device copy requires a dense source memory region.
  // contiguous() preserves logical element order and handles non-contiguous
  // tensors such as permuted or sliced views.
  tensor_t source = this->contiguous();

  // create() allocates storage on the target device and generates
  // standard row-major strides for the destination tensor.
  tensor_t destination =
      Tensor::create(source->shape(), source->dtype(), device_type, device_id);

  const size_t total_bytes = source->numel() * source->elementSize();

  if (total_bytes == 0) {
    return destination;
  }

  if (source_device_type == LLAISYS_DEVICE_CPU &&
      device_type == LLAISYS_DEVICE_CPU) {
    // Host-to-host memory is directly accessible.
    std::memcpy(destination->data(), source->data(), total_bytes);
  } else if (source_device_type == LLAISYS_DEVICE_CPU &&
             device_type != LLAISYS_DEVICE_CPU) {
    // Activate the destination accelerator before H2D transfer.
    core::context().setDevice(device_type, device_id);

    core::context().runtime().api()->memcpy_sync(
        destination->data(), source->data(), total_bytes, LLAISYS_MEMCPY_H2D);
  } else if (source_device_type != LLAISYS_DEVICE_CPU &&
             device_type == LLAISYS_DEVICE_CPU) {
    // Activate the source accelerator before D2H transfer.
    core::context().setDevice(source_device_type, source_device_id);

    core::context().runtime().api()->memcpy_sync(
        destination->data(), source->data(), total_bytes, LLAISYS_MEMCPY_D2H);
  } else {
    // This assumes that the runtime supports direct device-to-device
    // transfer between the selected source and destination devices.
    core::context().setDevice(source_device_type, source_device_id);

    core::context().runtime().api()->memcpy_sync(
        destination->data(), source->data(), total_bytes, LLAISYS_MEMCPY_D2D);
  }

  return destination;
}

}  // namespace llaisys
