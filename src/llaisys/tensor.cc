#include "llaisys/tensor.h"

#include "error.hpp"
#include "llaisys_tensor.hpp"

#include "../tensor/tensor.hpp"

#include <vector>
__C {
	llaisysTensor_t tensorCreate(size_t* shape, size_t ndim, llaisysDataType_t dtype, llaisysDeviceType_t device_type, int device_id) {
	return llaisys::c_api::guard_result<llaisysTensor_t>(
		[&]() {
			std::vector<size_t> shape_vec(shape, shape + ndim);
			return new LlaisysTensor{llaisys::Tensor::create(shape_vec, dtype, device_type, device_id)};
		},
		nullptr);
	}
	
	void tensorDestroy(
		llaisysTensor_t tensor) {
		delete tensor;
	}

	void *tensorGetData(
		llaisysTensor_t tensor) {
		return tensor->tensor->data();
	}

	size_t tensorGetNdim(
		llaisysTensor_t tensor) {
		return tensor->tensor->ndim();
	}

	void tensorGetShape(
		llaisysTensor_t tensor,
		size_t * shape) {
		std::copy(tensor->tensor->shape().begin(), tensor->tensor->shape().end(), shape);
	}

	void tensorGetStrides(
		llaisysTensor_t tensor,
		ptrdiff_t * strides) {
		std::copy(tensor->tensor->strides().begin(), tensor->tensor->strides().end(), strides);
	}

	llaisysDataType_t tensorGetDataType(
		llaisysTensor_t tensor) {
		return tensor->tensor->dtype();
	}

	llaisysDeviceType_t tensorGetDeviceType(
		llaisysTensor_t tensor) {
		return tensor->tensor->deviceType();
	}

	int tensorGetDeviceId(
		llaisysTensor_t tensor) {
		return tensor->tensor->deviceId();
	}

	int tensorDebug(
		llaisysTensor_t tensor) {
		return llaisys::c_api::guard(
			[&]() {
				tensor->tensor->debug();
			}
		)
			? 0
			: -1;
	}

	uint8_t tensorIsContiguous(
		llaisysTensor_t tensor) {
		return uint8_t(tensor->tensor->isContiguous());
	}

	int tensorLoad(
		llaisysTensor_t tensor,
		const void *data
	) {
		return llaisys::c_api::guard(
			[&]() {
				tensor->tensor->load(
					data
				);
			}
		)
			? 0
			: -1;
	}

	llaisysTensor_t tensorView(
		llaisysTensor_t tensor,
		size_t * shape,
		size_t ndim) {
		return llaisys::c_api::guard_result<llaisysTensor_t>(
			[&]() {
				std::vector<size_t> shape_vec(
					shape,
					shape + ndim
				);

				return new LlaisysTensor{
					tensor->tensor->view(
						shape_vec
					)
				};
			},
			nullptr
		);
	}

	llaisysTensor_t tensorPermute(
		llaisysTensor_t tensor,
		size_t * order) {
		return llaisys::c_api::guard_result<llaisysTensor_t>(
			[&]() {
				const size_t ndim =
					tensor->tensor->ndim();

				std::vector<size_t> order_vec(
					order,
					order + ndim
				);

				return new LlaisysTensor{
					tensor->tensor->permute(
						order_vec
					)
				};
			},
			nullptr
		);
	}

	llaisysTensor_t tensorSlice(
		llaisysTensor_t tensor,
		size_t dim,
		size_t start,
		size_t end) {
		return llaisys::c_api::guard_result<llaisysTensor_t>(
			[&]() {
				return new LlaisysTensor{
					tensor->tensor->slice(
						dim,
						start,
						end
					)
				};
			},
			nullptr
		);
	}
}
