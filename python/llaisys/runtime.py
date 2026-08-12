from ctypes import byref, c_void_p

from . import libllaisys
from .libllaisys import LIB_LLAISYS


class RuntimeAPI:
    def __init__(self, device_type: libllaisys.DeviceType):
        self._device_type = device_type
        self._api = LIB_LLAISYS.llaisysGetRuntimeAPI(libllaisys.llaisysDeviceType_t(device_type))

    def get_device_count(self) -> int:
        result = self._api.contents.get_device_count()
        return result

    def set_device(self, device_id: int) -> None:
        self._api.contents.set_device(device_id)

    def device_synchronize(self) -> None:
        self._api.contents.device_synchronize()

    def create_stream(self) -> libllaisys.llaisysStream_t:
        stream = self._api.contents.create_stream()
        return stream

    def destroy_stream(self, stream: libllaisys.llaisysStream_t) -> None:
        self._api.contents.destroy_stream(stream)

    def stream_synchronize(self, stream: libllaisys.llaisysStream_t) -> None:
        self._api.contents.stream_synchronize(stream)

    def malloc_device(self, size: int) -> c_void_p:
        ptr = self._api.contents.malloc_device(size)
        return ptr

    def free_device(self, ptr: c_void_p) -> None:
        print(f"[llaisys] free_device({ptr})")
        self._api.contents.free_device(ptr)

    def malloc_host(self, size: int) -> c_void_p:
        ptr = self._api.contents.malloc_host(size)
        return ptr

    def free_host(self, ptr: c_void_p) -> None:
        self._api.contents.free_host(ptr)

    def memcpy_sync(
        self,
        dst: c_void_p,
        src: c_void_p,
        size: int,
        kind: libllaisys.MemcpyKind,
    ) -> None:
        self._api.contents.memcpy_sync(dst, src, size, libllaisys.llaisysMemcpyKind_t(kind))

    def memcpy_async(
        self,
        dst: c_void_p,
        src: c_void_p,
        size: int,
        kind: libllaisys.MemcpyKind,
        stream: libllaisys.llaisysStream_t,
    ) -> None:
        self._api.contents.memcpy_async(dst, src, size, libllaisys.llaisysMemcpyKind_t(kind), stream)

    def get_context_stream(
        self,
        device_id: int = 0,
    ) -> int:
        stream = libllaisys.llaisysStream_t()

        status = LIB_LLAISYS.llaisysGetContextStream(
            libllaisys.llaisysDeviceType_t(self._device_type),
            device_id,
            byref(stream),
        )

        if status != 0:
            raise RuntimeError("Failed to get LLAISYS context stream")

        return 0 if stream.value is None else int(stream.value)
