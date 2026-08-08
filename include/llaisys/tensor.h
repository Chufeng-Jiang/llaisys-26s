#ifndef LLAISYS_TENSOR_H
#define LLAISYS_TENSOR_H

#include "../llaisys.h"

__C {

	typedef struct LlaisysTensor *llaisysTensor_t;

	__export llaisysTensor_t tensorCreate(
		size_t *shape,
		size_t ndim,
		llaisysDataType_t dtype,
		llaisysDeviceType_t device_type,
		int device_id
	);

	__export int tensorDestroy(
		llaisysTensor_t tensor
	);

	__export int tensorGetData(
		llaisysTensor_t tensor,
		void **data
	);

	__export int tensorGetNdim(
		llaisysTensor_t tensor,
		size_t *ndim
	);

	__export int tensorGetShape(
		llaisysTensor_t tensor,
		size_t *shape
	);

	__export int tensorGetStrides(
		llaisysTensor_t tensor,
		ptrdiff_t *strides
	);

	__export int tensorGetDataType(
		llaisysTensor_t tensor,
		llaisysDataType_t *dtype
	);

	__export int tensorGetDeviceType(
		llaisysTensor_t tensor,
		llaisysDeviceType_t *device_type
	);

	__export int tensorGetDeviceId(
		llaisysTensor_t tensor,
		int *device_id
	);

	__export int tensorDebug(
		llaisysTensor_t tensor
	);

	__export int tensorIsContiguous(
		llaisysTensor_t tensor,
		uint8_t *is_contiguous
	);

	__export int tensorLoad(
		llaisysTensor_t tensor,
		const void *data
	);

	__export llaisysTensor_t tensorView(
		llaisysTensor_t tensor,
		size_t *shape,
		size_t ndim
	);

	__export llaisysTensor_t tensorPermute(
		llaisysTensor_t tensor,
		size_t *order
	);

	__export llaisysTensor_t tensorSlice(
		llaisysTensor_t tensor,
		size_t dim,
		size_t start,
		size_t end
	);

}

#endif // LLAISYS_TENSOR_H