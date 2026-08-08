from ctypes import (
	byref,
	c_int,
	c_size_t,
	c_ssize_t,
	c_uint8,
	c_void_p,
)

from typing import (
	Sequence,
	Tuple,
)

from .libllaisys import (
	LIB_LLAISYS,
	llaisysTensor_t,
	llaisysDeviceType_t,
	DeviceType,
	llaisysDataType_t,
	DataType,
)


class Tensor:

	def __init__(
		self,
		shape: Sequence[int] = None,
		dtype: DataType = DataType.F32,
		device: DeviceType = DeviceType.CPU,
		device_id: int = 0,
		tensor: llaisysTensor_t = None,
	):
		if tensor:
			self._tensor = tensor
		else:
			_ndim = (
				0
				if shape is None
				else len(shape)
			)

			_shape = (
				None
				if shape is None
				else (c_size_t * len(shape))(
					*shape
				)
			)

			self._tensor: llaisysTensor_t = (
				LIB_LLAISYS.tensorCreate(
					_shape,
					c_size_t(_ndim),
					llaisysDataType_t(dtype),
					llaisysDeviceType_t(device),
					c_int(device_id),
				)
			)


	def __del__(self):
		if (
			hasattr(self, "_tensor")
			and self._tensor is not None
		):
			try:
				LIB_LLAISYS.tensorDestroy(
					self._tensor
				)
			except Exception:
				# Never propagate exceptions from __del__.
				pass

			self._tensor = None


	def shape(self) -> Tuple[int, ...]:
		ndim = self.ndim()

		buf = (
			c_size_t * ndim
		)()

		LIB_LLAISYS.tensorGetShape(
			self._tensor,
			buf,
		)

		return tuple(
			buf[i]
			for i in range(ndim)
		)


	def strides(self) -> Tuple[int, ...]:
		ndim = self.ndim()

		buf = (
			c_ssize_t * ndim
		)()

		LIB_LLAISYS.tensorGetStrides(
			self._tensor,
			buf,
		)

		return tuple(
			buf[i]
			for i in range(ndim)
		)


	def ndim(self) -> int:
		value = c_size_t()

		LIB_LLAISYS.tensorGetNdim(
			self._tensor,
			byref(value),
		)

		return int(
			value.value
		)


	def dtype(self) -> DataType:
		value = llaisysDataType_t()

		LIB_LLAISYS.tensorGetDataType(
			self._tensor,
			byref(value),
		)

		return DataType(
			value.value
		)


	def device_type(self) -> DeviceType:
		value = llaisysDeviceType_t()

		LIB_LLAISYS.tensorGetDeviceType(
			self._tensor,
			byref(value),
		)

		return DeviceType(
			value.value
		)


	def device_id(self) -> int:
		value = c_int()

		LIB_LLAISYS.tensorGetDeviceId(
			self._tensor,
			byref(value),
		)

		return int(
			value.value
		)


	def data_ptr(self) -> c_void_p:
		value = c_void_p()

		LIB_LLAISYS.tensorGetData(
			self._tensor,
			byref(value),
		)

		return value


	def lib_tensor(self) -> llaisysTensor_t:
		return self._tensor


	def debug(self):
		LIB_LLAISYS.tensorDebug(
			self._tensor
		)


	def __repr__(self):
		return (
			f"<Tensor "
			f"shape={self.shape()} "
			f"dtype={self.dtype()} "
			f"device={self.device_type()}:{self.device_id()}>"
		)


	def load(
		self,
		data: c_void_p
	):
		LIB_LLAISYS.tensorLoad(
			self._tensor,
			data,
		)


	def is_contiguous(self) -> bool:
		value = c_uint8()

		LIB_LLAISYS.tensorIsContiguous(
			self._tensor,
			byref(value),
		)

		return bool(
			value.value
		)


	def view(
		self,
		*shape: int
	):
		_shape = (
			c_size_t * len(shape)
		)(
			*shape
		)

		return Tensor(
			tensor=LIB_LLAISYS.tensorView(
				self._tensor,
				_shape,
				c_size_t(
					len(shape)
				),
			)
		)


	def permute(
		self,
		*perm: int
	):
		assert len(perm) == self.ndim()

		_perm = (
			c_size_t * len(perm)
		)(
			*perm
		)

		return Tensor(
			tensor=LIB_LLAISYS.tensorPermute(
				self._tensor,
				_perm,
			)
		)


	def slice(
		self,
		dim: int,
		start: int,
		end: int,
	):
		return Tensor(
			tensor=LIB_LLAISYS.tensorSlice(
				self._tensor,
				c_size_t(dim),
				c_size_t(start),
				c_size_t(end),
			)
		)