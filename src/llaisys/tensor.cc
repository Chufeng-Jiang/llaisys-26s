#include "llaisys/tensor.h"

#include "error.hpp"
#include "llaisys_tensor.hpp"

#include "../tensor/tensor.hpp"

#include <algorithm>
#include <vector>

__C {
    llaisysTensor_t tensorCreate(
        size_t *shape,
        size_t ndim,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id) {
        return llaisys::c_api::guard_result<llaisysTensor_t>(
            [&]() {
                std::vector<size_t> shape_vec(shape, shape + ndim);

                return new LlaisysTensor{
                    llaisys::Tensor::create(shape_vec, dtype, device_type, device_id)};
            },
            nullptr);
    }

    int tensorDestroy(llaisysTensor_t tensor) {
        return llaisys::c_api::guard([&]() { delete tensor; }) ? 0 : -1;
    }

    int tensorGetData(llaisysTensor_t tensor, void **data) {
        return llaisys::c_api::guard([&]() {
            if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

            if (data == nullptr) {
                throw std::invalid_argument("data output pointer must not be null");
            }

            *data = tensor->tensor->data();
        })
                 ? 0
                 : -1;
    }

    int tensorGetNdim(llaisysTensor_t tensor, size_t *ndim) {
        return llaisys::c_api::guard([&]() {
            if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

            if (ndim == nullptr) {
                throw std::invalid_argument("ndim output pointer must not be null");
            }

            *ndim = tensor->tensor->ndim();
        })
                 ? 0
                 : -1;
    }

    int tensorGetShape(llaisysTensor_t tensor, size_t *shape) {
        return llaisys::c_api::guard([&]() {
            if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

            if (shape == nullptr && tensor->tensor->ndim() != 0) {
                throw std::invalid_argument("shape output pointer must not be null");
            }

            std::copy(tensor->tensor->shape().begin(), tensor->tensor->shape().end(), shape);
        })
                 ? 0
                 : -1;
    }

    int tensorGetStrides(llaisysTensor_t tensor, ptrdiff_t *strides) {
        return llaisys::c_api::guard([&]() {
            if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

            if (strides == nullptr && tensor->tensor->ndim() != 0) {
                throw std::invalid_argument("strides output pointer must not be null");
            }

            std::copy(tensor->tensor->strides().begin(), tensor->tensor->strides().end(), strides);
        })
                 ? 0
                 : -1;
    }

    int tensorGetDataType(llaisysTensor_t tensor, llaisysDataType_t * dtype) {
        return llaisys::c_api::guard([&]() {
            if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

            if (dtype == nullptr) {
                throw std::invalid_argument("dtype output pointer must not be null");
            }

            *dtype = tensor->tensor->dtype();
        })
                 ? 0
                 : -1;
    }

    int tensorGetDeviceType(llaisysTensor_t tensor, llaisysDeviceType_t * device_type) {
        return llaisys::c_api::guard([&]() {
            if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

            if (device_type == nullptr) {
                throw std::invalid_argument("device type output pointer must not be null");
            }

            *device_type = tensor->tensor->deviceType();
        })
                 ? 0
                 : -1;
    }

    int tensorGetDeviceId(llaisysTensor_t tensor, int *device_id) {
        return llaisys::c_api::guard([&]() {
            if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

            if (device_id == nullptr) {
                throw std::invalid_argument("device id output pointer must not be null");
            }

            *device_id = tensor->tensor->deviceId();
        })
                 ? 0
                 : -1;
    }

    int tensorDebug(llaisysTensor_t tensor) {
        return llaisys::c_api::guard([&]() {
            if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

            tensor->tensor->debug();
        })
                 ? 0
                 : -1;
    }

    int tensorIsContiguous(llaisysTensor_t tensor, uint8_t *is_contiguous) {
        return llaisys::c_api::guard([&]() {
            if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

            if (is_contiguous == nullptr) {
                throw std::invalid_argument("is_contiguous output pointer must not be null");
            }

            *is_contiguous = static_cast<uint8_t>(tensor->tensor->isContiguous());
        })
                 ? 0
                 : -1;
    }

    int tensorLoad(llaisysTensor_t tensor, const void *data) {
        return llaisys::c_api::guard([&]() {
            if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

            if (data == nullptr) { throw std::invalid_argument("data must not be null"); }

            tensor->tensor->load(data);
        })
                 ? 0
                 : -1;
    }

    llaisysTensor_t tensorView(llaisysTensor_t tensor, size_t *shape, size_t ndim) {
        return llaisys::c_api::guard_result<llaisysTensor_t>(
            [&]() {
                if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

                if (shape == nullptr && ndim != 0) {
                    throw std::invalid_argument("shape must not be null");
                }

                std::vector<size_t> shape_vec(shape, shape + ndim);

                return new LlaisysTensor{tensor->tensor->view(shape_vec)};
            },
            nullptr);
    }

    llaisysTensor_t tensorPermute(llaisysTensor_t tensor, size_t *order) {
        return llaisys::c_api::guard_result<llaisysTensor_t>(
            [&]() {
                if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

                const size_t ndim = tensor->tensor->ndim();

                if (order == nullptr && ndim != 0) {
                    throw std::invalid_argument("order must not be null");
                }

                std::vector<size_t> order_vec(order, order + ndim);

                return new LlaisysTensor{tensor->tensor->permute(order_vec)};
            },
            nullptr);
    }

    llaisysTensor_t tensorSlice(llaisysTensor_t tensor, size_t dim, size_t start, size_t end) {
        return llaisys::c_api::guard_result<llaisysTensor_t>(
            [&]() {
                if (tensor == nullptr) { throw std::invalid_argument("tensor must not be null"); }

                return new LlaisysTensor{tensor->tensor->slice(dim, start, end)};
            },
            nullptr);
    }
}