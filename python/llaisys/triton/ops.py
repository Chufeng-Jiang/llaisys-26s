import triton

from ..libllaisys import DeviceType
from ..runtime import RuntimeAPI
from .backends.nvidia import NvidiaTritonBackend
from .kernels.add import add_kernel
from .tensor import as_nvidia_triton_tensor

# ============================================================
# NVIDIA Triton backend / runtime
# ============================================================

_nvidia_backend = NvidiaTritonBackend()
_nvidia_runtime = RuntimeAPI(DeviceType.NVIDIA)


def add(c, a, b):
    # ============================================================
    # Fetch metadata once
    # ============================================================

    c_shape = c.shape()
    a_shape = a.shape()
    b_shape = b.shape()

    c_dtype = c.dtype()
    a_dtype = a.dtype()
    b_dtype = b.dtype()

    c_device_type = c.device_type()
    a_device_type = a.device_type()
    b_device_type = b.device_type()

    c_device_id = c.device_id()
    a_device_id = a.device_id()
    b_device_id = b.device_id()

    # ============================================================
    # Validate
    # ============================================================

    if c_shape != a_shape or c_shape != b_shape:
        raise ValueError("Triton Add requires tensors with the same shape")

    if c_dtype != a_dtype or c_dtype != b_dtype:
        raise ValueError("Triton Add requires tensors with the same dtype")

    if c_device_type != DeviceType.NVIDIA or a_device_type != DeviceType.NVIDIA or b_device_type != DeviceType.NVIDIA:
        raise ValueError("NVIDIA Triton Add requires NVIDIA tensors")

    if c_device_id != a_device_id or c_device_id != b_device_id:
        raise ValueError("Triton Add requires tensors on the same device")

    # ============================================================
    # Numel
    # ============================================================

    numel = 1

    for dim in c_shape:
        numel *= dim

    if numel == 0:
        return c

    # ============================================================
    # Configuration
    # ============================================================

    config = _nvidia_backend.add_config(numel)

    block_size = config["BLOCK_SIZE"]

    grid = (triton.cdiv(numel, block_size),)

    # ============================================================
    # Triton wrappers
    # ============================================================

    c_triton = as_nvidia_triton_tensor(c)

    a_triton = as_nvidia_triton_tensor(a)

    b_triton = as_nvidia_triton_tensor(b)

    # ============================================================
    # LLAISYS stream
    # ============================================================

    stream_ptr = _nvidia_runtime.get_context_stream(c_device_id)

    # ============================================================
    # Launch
    #
    # Case 1:
    # An execution-level stream context is already active.
    #
    # Do NOT enter torch.cuda.stream(...) again.
    #
    # Case 2:
    # add() is called standalone.
    #
    # Enter the LLAISYS stream context locally so standalone
    # calls remain correct.
    # ============================================================

    if _nvidia_backend.in_execution_context(stream_ptr, c_device_id):
        add_kernel[grid](c_triton, a_triton, b_triton, numel, BLOCK_SIZE=block_size, num_warps=config["num_warps"])

    else:
        with _nvidia_backend.stream_context(stream_ptr, c_device_id):
            add_kernel[grid](c_triton, a_triton, b_triton, numel, BLOCK_SIZE=block_size, num_warps=config["num_warps"])

    return c
