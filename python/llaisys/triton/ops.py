import math

import triton

from .backends.nvidia import NvidiaTritonBackend

from .kernels.add import add_kernel

from .kernels.argmax import argmax_stage1_kernel, argmax_stage_n_kernel

from .kernels.embedding import embedding_kernel

from .kernels.rms_norm import rms_norm_kernel

from .kernels.rope import rope_kernel

from .kernels.swiglu import swiglu_kernel

from .tensor import as_nvidia_triton_tensor

from ..tensor import Tensor

from ..runtime import RuntimeAPI

from ..libllaisys import DataType, DeviceType

from .kernels.linear import linear_kernel, linear_zero_k_kernel
from .kernels.self_attention import self_attention_kernel

_nvidia_backend = NvidiaTritonBackend()

_nvidia_runtime = RuntimeAPI(DeviceType.NVIDIA)


# ============================================================
# Argmax workspace cache
# ============================================================

_argmax_workspace_cache = {}


# ============================================================
# General helpers
# ============================================================


def _numel(shape):
    result = 1

    for dim in shape:
        result *= dim

    return result


def _is_contiguous(shape, strides):
    if len(shape) != len(strides):
        return False

    numel = _numel(shape)

    if numel == 0:
        return True

    expected_stride = 1

    for index in range(len(shape) - 1, -1, -1):
        if shape[index] == 1:
            continue

        if strides[index] != expected_stride:
            return False

        expected_stride *= shape[index]

    return True


# ============================================================
# Argmax workspace
# ============================================================


def _get_argmax_workspace(num_blocks, dtype, device_id):
    key = (device_id, dtype, num_blocks)

    workspace = _argmax_workspace_cache.get(key)

    if workspace is not None:
        return workspace

    values_a = Tensor((num_blocks,), dtype=dtype, device=DeviceType.NVIDIA, device_id=device_id)

    indices_a = Tensor((num_blocks,), dtype=DataType.I64, device=DeviceType.NVIDIA, device_id=device_id)

    values_b = Tensor((num_blocks,), dtype=dtype, device=DeviceType.NVIDIA, device_id=device_id)

    indices_b = Tensor((num_blocks,), dtype=DataType.I64, device=DeviceType.NVIDIA, device_id=device_id)

    workspace = (values_a, indices_a, values_b, indices_b)

    _argmax_workspace_cache[key] = workspace

    return workspace


# ============================================================
# Add
# ============================================================


def add(c, a, b):
    c_shape = c.shape()
    a_shape = a.shape()
    b_shape = b.shape()

    c_strides = c.strides()
    a_strides = a.strides()
    b_strides = b.strides()

    c_dtype = c.dtype()
    a_dtype = a.dtype()
    b_dtype = b.dtype()

    c_device_type = c.device_type()
    a_device_type = a.device_type()
    b_device_type = b.device_type()

    c_device_id = c.device_id()
    a_device_id = a.device_id()
    b_device_id = b.device_id()

    if c_shape != a_shape or c_shape != b_shape:
        raise ValueError("Triton Add requires tensors with the same shape")

    if c_dtype != a_dtype or c_dtype != b_dtype:
        raise ValueError("Triton Add requires tensors with the same dtype")

    if c_device_type != DeviceType.NVIDIA or a_device_type != DeviceType.NVIDIA or b_device_type != DeviceType.NVIDIA:
        raise ValueError("NVIDIA Triton Add requires NVIDIA tensors")

    if c_device_id != a_device_id or c_device_id != b_device_id:
        raise ValueError("Triton Add requires tensors on the same device")

    if not _is_contiguous(c_shape, c_strides):
        raise ValueError("Triton Add output must be contiguous")

    if not _is_contiguous(a_shape, a_strides):
        raise ValueError("Triton Add left input must be contiguous")

    if not _is_contiguous(b_shape, b_strides):
        raise ValueError("Triton Add right input must be contiguous")

    numel = _numel(c_shape)

    if numel == 0:
        return c

    config = _nvidia_backend.add_config(numel)
    block_size = config["BLOCK_SIZE"]
    grid = (triton.cdiv(numel, block_size),)

    c_triton = as_nvidia_triton_tensor(c)
    a_triton = as_nvidia_triton_tensor(a)
    b_triton = as_nvidia_triton_tensor(b)

    stream_ptr = _nvidia_runtime.get_context_stream(c_device_id)

    def launch():
        add_kernel[grid](c_triton, a_triton, b_triton, numel, BLOCK_SIZE=block_size, num_warps=config["num_warps"])

    if _nvidia_backend.in_execution_context(stream_ptr, c_device_id):
        launch()
    else:
        with _nvidia_backend.stream_context(stream_ptr, c_device_id):
            launch()

    return c


# ============================================================
# SwiGLU
# ============================================================


def swiglu(out, gate, up):
    out_shape = out.shape()
    gate_shape = gate.shape()
    up_shape = up.shape()

    out_strides = out.strides()
    gate_strides = gate.strides()
    up_strides = up.strides()

    out_dtype = out.dtype()
    gate_dtype = gate.dtype()
    up_dtype = up.dtype()

    out_device_type = out.device_type()
    gate_device_type = gate.device_type()
    up_device_type = up.device_type()

    out_device_id = out.device_id()
    gate_device_id = gate.device_id()
    up_device_id = up.device_id()

    if out_shape != gate_shape or out_shape != up_shape:
        raise ValueError("Triton SwiGLU requires tensors with the same shape")

    if out_dtype != gate_dtype or out_dtype != up_dtype:
        raise ValueError("Triton SwiGLU requires tensors with the same dtype")

    if (
        out_device_type != DeviceType.NVIDIA
        or gate_device_type != DeviceType.NVIDIA
        or up_device_type != DeviceType.NVIDIA
    ):
        raise ValueError("NVIDIA Triton SwiGLU requires NVIDIA tensors")

    if out_device_id != gate_device_id or out_device_id != up_device_id:
        raise ValueError("Triton SwiGLU requires tensors on the same device")

    if not _is_contiguous(out_shape, out_strides):
        raise ValueError("Triton SwiGLU output must be contiguous")

    if not _is_contiguous(gate_shape, gate_strides):
        raise ValueError("Triton SwiGLU gate input must be contiguous")

    if not _is_contiguous(up_shape, up_strides):
        raise ValueError("Triton SwiGLU up input must be contiguous")

    numel = _numel(out_shape)

    if numel == 0:
        return out

    config = _nvidia_backend.swiglu_config(numel)
    block_size = config["BLOCK_SIZE"]
    grid = (triton.cdiv(numel, block_size),)

    out_triton = as_nvidia_triton_tensor(out)
    gate_triton = as_nvidia_triton_tensor(gate)
    up_triton = as_nvidia_triton_tensor(up)

    stream_ptr = _nvidia_runtime.get_context_stream(out_device_id)

    def launch():
        swiglu_kernel[grid](
            out_triton, gate_triton, up_triton, numel, BLOCK_SIZE=block_size, num_warps=config["num_warps"]
        )

    if _nvidia_backend.in_execution_context(stream_ptr, out_device_id):
        launch()
    else:
        with _nvidia_backend.stream_context(stream_ptr, out_device_id):
            launch()

    return out


# ============================================================
# RMSNorm
# ============================================================


def rms_norm(out, x, weight, eps):
    out_shape = out.shape()
    x_shape = x.shape()
    weight_shape = weight.shape()

    out_strides = out.strides()
    x_strides = x.strides()
    weight_strides = weight.strides()

    out_dtype = out.dtype()
    x_dtype = x.dtype()
    weight_dtype = weight.dtype()

    out_device_type = out.device_type()
    x_device_type = x.device_type()
    weight_device_type = weight.device_type()

    out_device_id = out.device_id()
    x_device_id = x.device_id()
    weight_device_id = weight.device_id()

    if len(out_shape) != 2:
        raise ValueError("Triton RMSNorm output tensor must be two-dimensional")

    if len(x_shape) != 2:
        raise ValueError("Triton RMSNorm input tensor must be two-dimensional")

    if len(weight_shape) != 1:
        raise ValueError("Triton RMSNorm weight tensor must be one-dimensional")

    nrow = x_shape[0]
    ncol = x_shape[1]

    if ncol <= 0:
        raise ValueError("Triton RMSNorm input row length must be greater than zero")

    if out_shape != x_shape:
        raise ValueError("Triton RMSNorm output shape must match input shape")

    if weight_shape[0] != ncol:
        raise ValueError("Triton RMSNorm weight length must match input row length")

    eps = float(eps)

    if not math.isfinite(eps):
        raise ValueError("Triton RMSNorm epsilon must be finite")

    if eps < 0.0:
        raise ValueError("Triton RMSNorm epsilon must not be negative")

    if out_dtype != x_dtype or out_dtype != weight_dtype:
        raise ValueError("Triton RMSNorm output, input, and weight must use the same dtype")

    if (
        out_device_type != DeviceType.NVIDIA
        or x_device_type != DeviceType.NVIDIA
        or weight_device_type != DeviceType.NVIDIA
    ):
        raise ValueError("NVIDIA Triton RMSNorm requires NVIDIA tensors")

    if out_device_id != x_device_id or out_device_id != weight_device_id:
        raise ValueError("Triton RMSNorm tensors must be on the same device")

    if not _is_contiguous(out_shape, out_strides):
        raise ValueError("Triton RMSNorm output must be contiguous")

    if not _is_contiguous(x_shape, x_strides):
        raise ValueError("Triton RMSNorm input must be contiguous")

    if not _is_contiguous(weight_shape, weight_strides):
        raise ValueError("Triton RMSNorm weight must be contiguous")

    if nrow == 0:
        return out

    config = _nvidia_backend.rms_norm_config(ncol)
    block_size = config["BLOCK_SIZE"]
    grid = (nrow,)

    out_triton = as_nvidia_triton_tensor(out)
    x_triton = as_nvidia_triton_tensor(x)
    weight_triton = as_nvidia_triton_tensor(weight)

    stream_ptr = _nvidia_runtime.get_context_stream(out_device_id)

    def launch():
        rms_norm_kernel[grid](
            out_triton,
            x_triton,
            weight_triton,
            eps,
            ncol,
            x_strides[0],
            x_strides[1],
            out_strides[0],
            out_strides[1],
            weight_strides[0],
            BLOCK_SIZE=block_size,
            num_warps=config["num_warps"],
        )

    if _nvidia_backend.in_execution_context(stream_ptr, out_device_id):
        launch()
    else:
        with _nvidia_backend.stream_context(stream_ptr, out_device_id):
            launch()

    return out


# ============================================================
# RoPE
# ============================================================


def rope(out, x, pos_ids, theta):
    out_shape = out.shape()
    x_shape = x.shape()
    pos_shape = pos_ids.shape()

    out_strides = out.strides()
    x_strides = x.strides()
    pos_strides = pos_ids.strides()

    out_dtype = out.dtype()
    x_dtype = x.dtype()
    pos_dtype = pos_ids.dtype()

    out_device_type = out.device_type()
    x_device_type = x.device_type()
    pos_device_type = pos_ids.device_type()

    out_device_id = out.device_id()
    x_device_id = x.device_id()
    pos_device_id = pos_ids.device_id()

    if len(out_shape) != 3:
        raise ValueError("Triton RoPE output tensor must be three-dimensional")

    if len(x_shape) != 3:
        raise ValueError("Triton RoPE input tensor must be three-dimensional")

    if len(pos_shape) != 1:
        raise ValueError("Triton RoPE position IDs must be one-dimensional")

    if out_shape != x_shape:
        raise ValueError("Triton RoPE output and input must have the same shape")

    sequence_length = x_shape[0]
    head_count = x_shape[1]
    head_dim = x_shape[2]

    if pos_shape[0] != sequence_length:
        raise ValueError("Triton RoPE position-id length must match sequence length")

    if sequence_length > 0 and head_count <= 0:
        raise ValueError("Triton RoPE head count must be positive for a nonempty sequence")

    if head_dim <= 0:
        raise ValueError("Triton RoPE head dimension must be greater than zero")

    if head_dim % 2 != 0:
        raise ValueError("Triton RoPE head dimension must be even")

    theta = float(theta)

    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("Triton RoPE theta must be finite and greater than zero")

    if out_dtype != x_dtype:
        raise ValueError("Triton RoPE output and input must use the same dtype")

    if pos_dtype != DataType.I64:
        raise ValueError("Triton RoPE position IDs must use Int64")

    if (
        out_device_type != DeviceType.NVIDIA
        or x_device_type != DeviceType.NVIDIA
        or pos_device_type != DeviceType.NVIDIA
    ):
        raise ValueError("NVIDIA Triton RoPE requires NVIDIA tensors")

    if out_device_id != x_device_id or out_device_id != pos_device_id:
        raise ValueError("Triton RoPE tensors must be on the same device")

    if not _is_contiguous(out_shape, out_strides):
        raise ValueError("Triton RoPE output must be contiguous")

    if not _is_contiguous(x_shape, x_strides):
        raise ValueError("Triton RoPE input must be contiguous")

    if not _is_contiguous(pos_shape, pos_strides):
        raise ValueError("Triton RoPE position IDs must be contiguous")

    if sequence_length == 0 or head_count == 0:
        return out

    config = _nvidia_backend.rope_config(head_dim)
    block_size = config["BLOCK_SIZE"]
    half_dim = head_dim // 2

    grid = (sequence_length, head_count, triton.cdiv(half_dim, block_size))

    out_triton = as_nvidia_triton_tensor(out)
    x_triton = as_nvidia_triton_tensor(x)
    pos_triton = as_nvidia_triton_tensor(pos_ids)

    stream_ptr = _nvidia_runtime.get_context_stream(out_device_id)

    def launch():
        rope_kernel[grid](
            out_triton,
            x_triton,
            pos_triton,
            theta,
            x_strides[0],
            x_strides[1],
            x_strides[2],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            pos_strides[0],
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            num_warps=config["num_warps"],
        )

    if _nvidia_backend.in_execution_context(stream_ptr, out_device_id):
        launch()
    else:
        with _nvidia_backend.stream_context(stream_ptr, out_device_id):
            launch()

    return out


# ============================================================
# Argmax
# ============================================================


def argmax(max_idx, max_val, vals):
    vals_shape = vals.shape()
    max_idx_shape = max_idx.shape()
    max_val_shape = max_val.shape()

    vals_strides = vals.strides()
    max_idx_strides = max_idx.strides()
    max_val_strides = max_val.strides()

    vals_dtype = vals.dtype()
    max_idx_dtype = max_idx.dtype()
    max_val_dtype = max_val.dtype()

    vals_device_type = vals.device_type()
    max_idx_device_type = max_idx.device_type()
    max_val_device_type = max_val.device_type()

    vals_device_id = vals.device_id()
    max_idx_device_id = max_idx.device_id()
    max_val_device_id = max_val.device_id()

    if len(vals_shape) != 1:
        raise ValueError("Triton Argmax input tensor must be one-dimensional")

    if max_idx_shape != (1,):
        raise ValueError("Triton Argmax max_idx must have shape (1,)")

    if max_val_shape != (1,):
        raise ValueError("Triton Argmax max_val must have shape (1,)")

    numel = _numel(vals_shape)

    if numel <= 0:
        raise ValueError("Triton Argmax input tensor must not be empty")

    if numel > 0xFFFFFFFF:
        raise ValueError("Triton Argmax baseline currently supports at most UINT32_MAX elements")

    supported_value_dtypes = (DataType.F32, DataType.F16, DataType.BF16)

    if vals_dtype not in supported_value_dtypes:
        raise TypeError("Triton Argmax supports F32, F16, and BF16 inputs")

    if max_val_dtype != vals_dtype:
        raise ValueError("Triton Argmax max_val dtype must match input dtype")

    if max_idx_dtype != DataType.I64:
        raise ValueError("Triton Argmax max_idx must use Int64")

    if (
        vals_device_type != DeviceType.NVIDIA
        or max_idx_device_type != DeviceType.NVIDIA
        or max_val_device_type != DeviceType.NVIDIA
    ):
        raise ValueError("NVIDIA Triton Argmax requires NVIDIA tensors")

    if vals_device_id != max_idx_device_id or vals_device_id != max_val_device_id:
        raise ValueError("Triton Argmax tensors must be on the same device")

    if not _is_contiguous(vals_shape, vals_strides):
        raise ValueError("Triton Argmax input must be contiguous")

    if not _is_contiguous(max_idx_shape, max_idx_strides):
        raise ValueError("Triton Argmax max_idx must be contiguous")

    if not _is_contiguous(max_val_shape, max_val_strides):
        raise ValueError("Triton Argmax max_val must be contiguous")

    config = _nvidia_backend.argmax_config(numel)

    stage1_block = config["STAGE1_BLOCK_SIZE"]

    stage1_warps = config["STAGE1_NUM_WARPS"]

    stagen_block = config["STAGEN_BLOCK_SIZE"]

    stagen_warps = config["STAGEN_NUM_WARPS"]

    num_blocks = (numel + stage1_block - 1) // stage1_block

    vals_triton = as_nvidia_triton_tensor(vals)

    max_idx_triton = as_nvidia_triton_tensor(max_idx)

    max_val_triton = as_nvidia_triton_tensor(max_val)

    def launch():
        if num_blocks == 1:
            argmax_stage1_kernel[(1,)](
                vals_triton, max_val_triton, max_idx_triton, numel, BLOCK_SIZE=stage1_block, num_warps=stage1_warps
            )

            return

        (values_a, indices_a, values_b, indices_b) = _get_argmax_workspace(num_blocks, vals_dtype, vals_device_id)

        values_a_triton = as_nvidia_triton_tensor(values_a)

        indices_a_triton = as_nvidia_triton_tensor(indices_a)

        values_b_triton = as_nvidia_triton_tensor(values_b)

        indices_b_triton = as_nvidia_triton_tensor(indices_b)

        argmax_stage1_kernel[(num_blocks,)](
            vals_triton, values_a_triton, indices_a_triton, numel, BLOCK_SIZE=stage1_block, num_warps=stage1_warps
        )

        cur_values = values_a_triton
        cur_indices = indices_a_triton
        cur_n = num_blocks
        current_is_a = True

        while cur_n > 1:
            next_n = (cur_n + stagen_block - 1) // stagen_block

            if next_n == 1:
                out_values = max_val_triton
                out_indices = max_idx_triton

            elif current_is_a:
                out_values = values_b_triton
                out_indices = indices_b_triton

            else:
                out_values = values_a_triton
                out_indices = indices_a_triton

            argmax_stage_n_kernel[(next_n,)](
                cur_values, cur_indices, out_values, out_indices, cur_n, BLOCK_SIZE=stagen_block, num_warps=stagen_warps
            )

            cur_values = out_values
            cur_indices = out_indices
            cur_n = next_n

            if cur_n > 1:
                current_is_a = not current_is_a

    stream_ptr = _nvidia_runtime.get_context_stream(vals_device_id)

    if _nvidia_backend.in_execution_context(stream_ptr, vals_device_id):
        launch()

    else:
        with _nvidia_backend.stream_context(stream_ptr, vals_device_id):
            launch()

    return (max_idx, max_val)


# ============================================================
# Embedding
# ============================================================


def embedding(out, index, weight):
    # ========================================================
    # Metadata
    # ========================================================

    out_shape = out.shape()
    index_shape = index.shape()
    weight_shape = weight.shape()

    out_strides = out.strides()
    index_strides = index.strides()
    weight_strides = weight.strides()

    out_dtype = out.dtype()
    index_dtype = index.dtype()
    weight_dtype = weight.dtype()

    out_device_type = out.device_type()
    index_device_type = index.device_type()
    weight_device_type = weight.device_type()

    out_device_id = out.device_id()
    index_device_id = index.device_id()
    weight_device_id = weight.device_id()

    # ========================================================
    # Shape
    #
    #     index  [N]
    #     weight [V, D]
    #     out    [N, D]
    # ========================================================

    if len(index_shape) != 1:
        raise ValueError("Triton Embedding index tensor must be one-dimensional")

    if len(weight_shape) != 2:
        raise ValueError("Triton Embedding weight tensor must be two-dimensional")

    if len(out_shape) != 2:
        raise ValueError("Triton Embedding output tensor must be two-dimensional")

    index_count = index_shape[0]
    vocabulary_size = weight_shape[0]
    embedding_dim = weight_shape[1]

    if out_shape[0] != index_count:
        raise ValueError("Triton Embedding output row count must match index count")

    if out_shape[1] != embedding_dim:
        raise ValueError("Triton Embedding output column count must match embedding dimension")

    # ========================================================
    # DType
    # ========================================================

    if index_dtype != DataType.I64:
        raise ValueError("Triton Embedding index tensor must use Int64")

    supported_value_dtypes = (DataType.F32, DataType.F16, DataType.BF16)

    if weight_dtype not in supported_value_dtypes:
        raise TypeError("Triton Embedding supports F32, F16, and BF16 weights")

    if out_dtype != weight_dtype:
        raise ValueError("Triton Embedding output and weight must use the same dtype")

    # ========================================================
    # Device
    # ========================================================

    if (
        out_device_type != DeviceType.NVIDIA
        or index_device_type != DeviceType.NVIDIA
        or weight_device_type != DeviceType.NVIDIA
    ):
        raise ValueError("NVIDIA Triton Embedding requires NVIDIA tensors")

    if out_device_id != index_device_id or out_device_id != weight_device_id:
        raise ValueError("Triton Embedding tensors must be on the same device")

    # ========================================================
    # Contiguity
    # ========================================================

    if not _is_contiguous(out_shape, out_strides):
        raise ValueError("Triton Embedding output must be contiguous")

    if not _is_contiguous(index_shape, index_strides):
        raise ValueError("Triton Embedding index must be contiguous")

    if not _is_contiguous(weight_shape, weight_strides):
        raise ValueError("Triton Embedding weight must be contiguous")

    # ========================================================
    # Empty work
    # ========================================================

    if index_count == 0 or embedding_dim == 0:
        return out

    # ========================================================
    # A zero-row table means every index is invalid.
    #
    # Current LLAISYS semantics:
    #
    #     invalid index
    #         ↓
    #     output row untouched
    #
    # Therefore no kernel is necessary.
    # ========================================================

    if vocabulary_size == 0:
        return out

    # ========================================================
    # Configuration
    # ========================================================

    config = _nvidia_backend.embedding_config(embedding_dim)

    block_size = config["BLOCK_SIZE"]

    # ========================================================
    # Grid
    # ========================================================

    grid = (index_count, triton.cdiv(embedding_dim, block_size))

    # ========================================================
    # Tensor bridge
    # ========================================================

    out_triton = as_nvidia_triton_tensor(out)

    index_triton = as_nvidia_triton_tensor(index)

    weight_triton = as_nvidia_triton_tensor(weight)

    # ========================================================
    # Runtime stream
    # ========================================================

    stream_ptr = _nvidia_runtime.get_context_stream(out_device_id)

    # ========================================================
    # Launch
    # ========================================================

    def launch():
        embedding_kernel[grid](
            out_triton,
            index_triton,
            weight_triton,
            vocabulary_size,
            embedding_dim,
            BLOCK_SIZE=block_size,
            num_warps=config["num_warps"],
        )

    if _nvidia_backend.in_execution_context(stream_ptr, out_device_id):
        launch()

    else:
        with _nvidia_backend.stream_context(stream_ptr, out_device_id):
            launch()

    return out


def linear(out, x, weight, bias=None):
    # ============================================================
    # Metadata
    # ============================================================

    out_shape = out.shape()
    x_shape = x.shape()
    weight_shape = weight.shape()

    out_strides = out.strides()
    x_strides = x.strides()
    weight_strides = weight.strides()

    out_dtype = out.dtype()
    x_dtype = x.dtype()
    weight_dtype = weight.dtype()

    out_device_type = out.device_type()
    x_device_type = x.device_type()
    weight_device_type = weight.device_type()

    out_device_id = out.device_id()
    x_device_id = x.device_id()
    weight_device_id = weight.device_id()

    # ============================================================
    # Shape contract
    #
    #     X:      [M, K]
    #     Weight: [N, K]
    #     Out:    [M, N]
    # ============================================================

    if len(out_shape) != 2:
        raise ValueError("Triton Linear output tensor must be two-dimensional")

    if len(x_shape) != 2:
        raise ValueError("Triton Linear input tensor must be two-dimensional")

    if len(weight_shape) != 2:
        raise ValueError("Triton Linear weight tensor must be two-dimensional")

    m = x_shape[0]
    k = x_shape[1]
    n = weight_shape[0]

    if weight_shape[1] != k:
        raise ValueError("Triton Linear input feature count must match weight row length")

    if out_shape[0] != m:
        raise ValueError("Triton Linear output row count must match input row count")

    if out_shape[1] != n:
        raise ValueError("Triton Linear output feature count must match weight row count")

    # ============================================================
    # DType
    # ============================================================

    supported_dtypes = (DataType.F32, DataType.F16, DataType.BF16)

    if x_dtype not in supported_dtypes:
        raise TypeError("Triton Linear supports F32, F16, and BF16")

    if out_dtype != x_dtype or weight_dtype != x_dtype:
        raise ValueError("Triton Linear output, input, and weight must use the same dtype")

    # ============================================================
    # Device
    # ============================================================

    if (
        out_device_type != DeviceType.NVIDIA
        or x_device_type != DeviceType.NVIDIA
        or weight_device_type != DeviceType.NVIDIA
    ):
        raise ValueError("NVIDIA Triton Linear requires NVIDIA tensors")

    if out_device_id != x_device_id or out_device_id != weight_device_id:
        raise ValueError("Triton Linear output, input, and weight must be on the same device")

    # ============================================================
    # Contiguity
    # ============================================================

    if not _is_contiguous(out_shape, out_strides):
        raise ValueError("Triton Linear output must be contiguous")

    if not _is_contiguous(x_shape, x_strides):
        raise ValueError("Triton Linear input must be contiguous")

    if not _is_contiguous(weight_shape, weight_strides):
        raise ValueError("Triton Linear weight must be contiguous")

    # ============================================================
    # Optional bias
    # ============================================================

    if bias is not None:
        bias_shape = bias.shape()
        bias_strides = bias.strides()
        bias_dtype = bias.dtype()
        bias_device_type = bias.device_type()
        bias_device_id = bias.device_id()

        if len(bias_shape) != 1:
            raise ValueError("Triton Linear bias must be one-dimensional")

        if bias_shape[0] != n:
            raise ValueError("Triton Linear bias length must match output feature count")

        if bias_dtype != out_dtype:
            raise ValueError("Triton Linear bias must use the same dtype as output")

        if bias_device_type != DeviceType.NVIDIA:
            raise ValueError("NVIDIA Triton Linear bias must be an NVIDIA tensor")

        if bias_device_id != out_device_id:
            raise ValueError("Triton Linear bias must be on the same device as output")

        if not _is_contiguous(bias_shape, bias_strides):
            raise ValueError("Triton Linear bias must be contiguous")

    # ============================================================
    # Empty output
    # ============================================================

    if m == 0 or n == 0:
        return out

    # ============================================================
    # Configuration
    # ============================================================

    config = _nvidia_backend.linear_config(m, n, k)

    # ============================================================
    # Tensor bridge
    # ============================================================

    out_triton = as_nvidia_triton_tensor(out)

    # ============================================================
    # Bias
    #
    # Triton still needs a pointer argument even when HAS_BIAS
    # is False. Use output as a harmless dummy pointer.
    # ============================================================

    has_bias = bias is not None

    if has_bias:
        bias_triton = as_nvidia_triton_tensor(bias)

        stride_bias = bias.strides()[0]

    else:
        bias_triton = out_triton

        stride_bias = 0

    # ============================================================
    # Runtime stream
    # ============================================================

    stream_ptr = _nvidia_runtime.get_context_stream(out_device_id)

    # ============================================================
    # K == 0
    #
    # Native semantics:
    #
    #     no bias:
    #         output = 0
    #
    #     bias:
    #         output = broadcast(bias)
    # ============================================================

    if k == 0:
        block_size = config["ZERO_K_BLOCK_SIZE"]

        grid = (triton.cdiv(m * n, block_size),)

        def launch_zero_k():
            linear_zero_k_kernel[grid](
                out_triton,
                bias_triton,
                m,
                n,
                out_strides[0],
                out_strides[1],
                stride_bias,
                HAS_BIAS=has_bias,
                BLOCK_SIZE=block_size,
                num_warps=4,
            )

        if _nvidia_backend.in_execution_context(stream_ptr, out_device_id):
            launch_zero_k()

        else:
            with _nvidia_backend.stream_context(stream_ptr, out_device_id):
                launch_zero_k()

        return out

    # ============================================================
    # Normal GEMM path
    # ============================================================

    x_triton = as_nvidia_triton_tensor(x)

    weight_triton = as_nvidia_triton_tensor(weight)

    block_m = config["BLOCK_M"]

    block_n = config["BLOCK_N"]

    block_k = config["BLOCK_K"]

    group_m = config["GROUP_M"]

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    def launch():
        linear_kernel[grid](
            out_triton,
            x_triton,
            weight_triton,
            bias_triton,
            m,
            n,
            k,
            x_strides[0],
            x_strides[1],
            weight_strides[0],
            weight_strides[1],
            out_strides[0],
            out_strides[1],
            stride_bias,
            HAS_BIAS=has_bias,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )

    # ============================================================
    # Same LLAISYS CUDA stream
    # ============================================================

    if _nvidia_backend.in_execution_context(stream_ptr, out_device_id):
        launch()

    else:
        with _nvidia_backend.stream_context(stream_ptr, out_device_id):
            launch()

    return out


def self_attention(attn_val, q, k, v, scale):
    out_shape = attn_val.shape()
    q_shape = q.shape()
    k_shape = k.shape()
    v_shape = v.shape()

    out_strides = attn_val.strides()
    q_strides = q.strides()
    k_strides = k.strides()
    v_strides = v.strides()

    out_dtype = attn_val.dtype()
    q_dtype = q.dtype()
    k_dtype = k.dtype()
    v_dtype = v.dtype()

    out_device_type = attn_val.device_type()
    q_device_type = q.device_type()
    k_device_type = k.device_type()
    v_device_type = v.device_type()

    out_device_id = attn_val.device_id()
    q_device_id = q.device_id()
    k_device_id = k.device_id()
    v_device_id = v.device_id()

    if len(out_shape) != 3:
        raise ValueError("Triton Self-Attention output must be three-dimensional")

    if len(q_shape) != 3:
        raise ValueError("Triton Self-Attention query must be three-dimensional")

    if len(k_shape) != 3:
        raise ValueError("Triton Self-Attention key must be three-dimensional")

    if len(v_shape) != 3:
        raise ValueError("Triton Self-Attention value must be three-dimensional")

    seqlen = q_shape[0]
    nhead = q_shape[1]
    qk_dim = q_shape[2]

    total_len = k_shape[0]
    nkvhead = k_shape[1]
    value_dim = v_shape[2]

    if k_shape[2] != qk_dim:
        raise ValueError("Triton Self-Attention query/key dimensions must match")

    if v_shape[0] != total_len:
        raise ValueError("Triton Self-Attention key/value sequence lengths must match")

    if v_shape[1] != nkvhead:
        raise ValueError("Triton Self-Attention key/value head counts must match")

    if out_shape != (seqlen, nhead, value_dim):
        raise ValueError("Triton Self-Attention output shape must be [seqlen, nhead, value_dim]")

    if nhead <= 0:
        raise ValueError("Triton Self-Attention query head count must be positive")

    if nkvhead <= 0:
        raise ValueError("Triton Self-Attention KV head count must be positive")

    if nhead % nkvhead != 0:
        raise ValueError("Triton Self-Attention query head count must be a multiple of KV head count")

    if total_len < seqlen:
        raise ValueError("Triton Self-Attention total KV length must not be smaller than query length")

    if qk_dim <= 0:
        raise ValueError("Triton Self-Attention Q/K dimension must be positive")

    if value_dim <= 0:
        raise ValueError("Triton Self-Attention value dimension must be positive")

    scale = float(scale)

    if not math.isfinite(scale):
        raise ValueError("Triton Self-Attention scale must be finite")

    supported_dtypes = (DataType.F32, DataType.F16, DataType.BF16)

    if q_dtype not in supported_dtypes:
        raise TypeError("Triton Self-Attention supports F32, F16, and BF16")

    if not (out_dtype == q_dtype == k_dtype == v_dtype):
        raise ValueError("Triton Self-Attention output, query, key, and value must use the same dtype")

    if not (
        out_device_type == DeviceType.NVIDIA
        and q_device_type == DeviceType.NVIDIA
        and k_device_type == DeviceType.NVIDIA
        and v_device_type == DeviceType.NVIDIA
    ):
        raise ValueError("NVIDIA Triton Self-Attention requires NVIDIA tensors")

    if not (out_device_id == q_device_id == k_device_id == v_device_id):
        raise ValueError("Triton Self-Attention tensors must be on the same device")

    if not _is_contiguous(out_shape, out_strides):
        raise ValueError("Triton Self-Attention output must be contiguous")

    if not _is_contiguous(q_shape, q_strides):
        raise ValueError("Triton Self-Attention query must be contiguous")

    if not _is_contiguous(k_shape, k_strides):
        raise ValueError("Triton Self-Attention key must be contiguous")

    if not _is_contiguous(v_shape, v_strides):
        raise ValueError("Triton Self-Attention value must be contiguous")

    if seqlen == 0:
        return attn_val

    group_size = nhead // nkvhead

    config = _nvidia_backend.self_attention_config(qk_dim, value_dim, total_len)

    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    block_d = config["BLOCK_D"]
    block_v = config["BLOCK_V"]

    grid = (triton.cdiv(seqlen, block_m), nhead, triton.cdiv(value_dim, block_v))

    out_triton = as_nvidia_triton_tensor(attn_val)
    q_triton = as_nvidia_triton_tensor(q)
    k_triton = as_nvidia_triton_tensor(k)
    v_triton = as_nvidia_triton_tensor(v)

    stream_ptr = _nvidia_runtime.get_context_stream(out_device_id)

    def launch():
        self_attention_kernel[grid](
            out_triton,
            q_triton,
            k_triton,
            v_triton,
            q_strides[0],
            q_strides[1],
            q_strides[2],
            k_strides[0],
            k_strides[1],
            k_strides[2],
            v_strides[0],
            v_strides[1],
            v_strides[2],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            seqlen,
            total_len,
            scale,
            GROUP_SIZE=group_size,
            QK_DIM=qk_dim,
            V_DIM=value_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            BLOCK_V=block_v,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )

    if _nvidia_backend.in_execution_context(stream_ptr, out_device_id):
        launch()
    else:
        with _nvidia_backend.stream_context(stream_ptr, out_device_id):
            launch()

    return attn_val
