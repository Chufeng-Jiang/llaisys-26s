from collections.abc import Sequence
from ctypes import (
    byref,
    c_int,
    c_size_t,
    c_ssize_t,
    c_uint8,
    c_void_p,
)

from .libllaisys import (
    LIB_LLAISYS,
    DataType,
    DeviceType,
    llaisysDataType_t,
    llaisysDeviceType_t,
    llaisysTensor_t,
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
        # ========================================================
        # Create / wrap native tensor
        # ========================================================

        if tensor is not None:
            self._tensor = tensor

        else:
            _ndim = 0 if shape is None else len(shape)

            _shape = None if shape is None else (c_size_t * len(shape))(*shape)

            self._tensor: llaisysTensor_t = LIB_LLAISYS.tensorCreate(
                _shape,
                c_size_t(_ndim),
                llaisysDataType_t(dtype),
                llaisysDeviceType_t(device),
                c_int(device_id),
            )

        # ========================================================
        # Cache immutable tensor metadata
        # ========================================================
        #
        # Important:
        #
        # Metadata is fetched from C++ exactly once when this
        # Python Tensor wrapper is created.
        #
        # Operations such as:
        #
        #     view
        #     permute
        #     slice
        #
        # return NEW Tensor wrappers, so those wrappers obtain
        # their own metadata cache from the returned C++ Tensor.
        #
        # data_ptr() is intentionally NOT cached.
        # ========================================================

        self._cache_metadata()

    # ============================================================
    # Metadata cache
    # ============================================================

    def _cache_metadata(self):
        # --------------------------------------------------------
        # ndim
        # --------------------------------------------------------

        ndim_value = c_size_t()

        LIB_LLAISYS.tensorGetNdim(
            self._tensor,
            byref(ndim_value),
        )

        self._cached_ndim = int(ndim_value.value)

        # --------------------------------------------------------
        # shape
        # --------------------------------------------------------

        if self._cached_ndim == 0:
            self._cached_shape = ()
        else:
            shape_buf = (c_size_t * self._cached_ndim)()

            LIB_LLAISYS.tensorGetShape(
                self._tensor,
                shape_buf,
            )

            self._cached_shape = tuple(int(shape_buf[i]) for i in range(self._cached_ndim))

        # --------------------------------------------------------
        # strides
        # --------------------------------------------------------

        if self._cached_ndim == 0:
            self._cached_strides = ()
        else:
            strides_buf = (c_ssize_t * self._cached_ndim)()

            LIB_LLAISYS.tensorGetStrides(
                self._tensor,
                strides_buf,
            )

            self._cached_strides = tuple(int(strides_buf[i]) for i in range(self._cached_ndim))

        # --------------------------------------------------------
        # dtype
        # --------------------------------------------------------

        dtype_value = llaisysDataType_t()

        LIB_LLAISYS.tensorGetDataType(
            self._tensor,
            byref(dtype_value),
        )

        self._cached_dtype = DataType(dtype_value.value)

        # --------------------------------------------------------
        # device type
        # --------------------------------------------------------

        device_type_value = llaisysDeviceType_t()

        LIB_LLAISYS.tensorGetDeviceType(
            self._tensor,
            byref(device_type_value),
        )

        self._cached_device_type = DeviceType(device_type_value.value)

        # --------------------------------------------------------
        # device id
        # --------------------------------------------------------

        device_id_value = c_int()

        LIB_LLAISYS.tensorGetDeviceId(
            self._tensor,
            byref(device_id_value),
        )

        self._cached_device_id = int(device_id_value.value)

    # ============================================================
    # Lifetime
    # ============================================================

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor is not None:
            try:
                LIB_LLAISYS.tensorDestroy(self._tensor)

            except Exception:
                # Never propagate exceptions from __del__.
                pass

            self._tensor = None

    # ============================================================
    # Cached metadata access
    # ============================================================

    def shape(self) -> tuple[int, ...]:
        return self._cached_shape

    def strides(self) -> tuple[int, ...]:
        return self._cached_strides

    def ndim(self) -> int:
        return self._cached_ndim

    def dtype(self) -> DataType:
        return self._cached_dtype

    def device_type(self) -> DeviceType:
        return self._cached_device_type

    def device_id(self) -> int:
        return self._cached_device_id

    # ============================================================
    # Raw data pointer
    # ============================================================
    #
    # Do NOT cache this yet.
    #
    # Pointer semantics depend on the underlying storage and
    # offset. Keeping this query through the C API is safer while
    # we are validating the metadata-cache optimization.
    # ============================================================

    def data_ptr(self) -> int:
        value = c_void_p()

        LIB_LLAISYS.tensorGetData(
            self._tensor,
            byref(value),
        )

        return 0 if value.value is None else int(value.value)

    # ============================================================
    # Native tensor handle
    # ============================================================

    def lib_tensor(self) -> llaisysTensor_t:
        return self._tensor

    # ============================================================
    # Debug
    # ============================================================

    def debug(self):
        LIB_LLAISYS.tensorDebug(self._tensor)

    def __repr__(self):
        return f"<Tensor shape={self.shape()} dtype={self.dtype()} device={self.device_type()}:{self.device_id()}>"

    # ============================================================
    # Data loading
    # ============================================================

    def load(
        self,
        data: c_void_p,
    ):
        LIB_LLAISYS.tensorLoad(
            self._tensor,
            data,
        )

    # ============================================================
    # Layout
    # ============================================================

    def is_contiguous(self) -> bool:
        value = c_uint8()

        LIB_LLAISYS.tensorIsContiguous(
            self._tensor,
            byref(value),
        )

        return bool(value.value)

    # ============================================================
    # View
    # ============================================================
    #
    # tensorView creates a NEW native Tensor.
    #
    # Tensor(tensor=...) therefore runs _cache_metadata() on the
    # new Tensor and obtains its new shape/strides.
    # ============================================================

    def view(
        self,
        *shape: int,
    ):
        _shape = (c_size_t * len(shape))(*shape)

        return Tensor(
            tensor=LIB_LLAISYS.tensorView(
                self._tensor,
                _shape,
                c_size_t(len(shape)),
            )
        )

    # ============================================================
    # Permute
    # ============================================================
    #
    # The returned Tensor has different shape/strides, therefore
    # its constructor obtains a fresh metadata cache.
    # ============================================================

    def permute(
        self,
        *perm: int,
    ):
        assert len(perm) == self.ndim()

        _perm = (c_size_t * len(perm))(*perm)

        return Tensor(
            tensor=LIB_LLAISYS.tensorPermute(
                self._tensor,
                _perm,
            )
        )

    # ============================================================
    # Slice
    # ============================================================
    #
    # The returned Tensor may have a new shape and data offset.
    #
    # Metadata is refreshed in the new wrapper.
    # data_ptr() remains dynamic and therefore reflects the new
    # underlying Tensor's actual data address.
    # ============================================================

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
