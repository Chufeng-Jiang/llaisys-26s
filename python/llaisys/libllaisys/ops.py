from ctypes import c_float, c_int

from .tensor import llaisysTensor_t


def load_ops(lib):
    def check_status(
        result,
        func,
        arguments,
    ):
        if result == 0:
            return result

        message = lib.llaisysGetLastError()

        if message:
            raise RuntimeError(
                message.decode(
                    "utf-8",
                    errors="replace",
                )
            )

        raise RuntimeError("Unknown LLAISYS error")

    # Add
    lib.llaisysAdd.argtypes = [
        llaisysTensor_t,
        llaisysTensor_t,
        llaisysTensor_t,
    ]
    lib.llaisysAdd.restype = c_int
    lib.llaisysAdd.errcheck = check_status

    # Argmax
    lib.llaisysArgmax.argtypes = [
        llaisysTensor_t,
        llaisysTensor_t,
        llaisysTensor_t,
    ]
    lib.llaisysArgmax.restype = c_int
    lib.llaisysArgmax.errcheck = check_status

    # Embedding
    lib.llaisysEmbedding.argtypes = [
        llaisysTensor_t,
        llaisysTensor_t,
        llaisysTensor_t,
    ]
    lib.llaisysEmbedding.restype = c_int
    lib.llaisysEmbedding.errcheck = check_status

    # Linear
    lib.llaisysLinear.argtypes = [
        llaisysTensor_t,
        llaisysTensor_t,
        llaisysTensor_t,
        llaisysTensor_t,
    ]
    lib.llaisysLinear.restype = c_int
    lib.llaisysLinear.errcheck = check_status

    # Rearrange
    lib.llaisysRearrange.argtypes = [
        llaisysTensor_t,
        llaisysTensor_t,
    ]
    lib.llaisysRearrange.restype = c_int
    lib.llaisysRearrange.errcheck = check_status

    # RMSNorm
    lib.llaisysRmsNorm.argtypes = [
        llaisysTensor_t,
        llaisysTensor_t,
        llaisysTensor_t,
        c_float,
    ]
    lib.llaisysRmsNorm.restype = c_int
    lib.llaisysRmsNorm.errcheck = check_status

    # RoPE
    lib.llaisysROPE.argtypes = [
        llaisysTensor_t,
        llaisysTensor_t,
        llaisysTensor_t,
        c_float,
    ]
    lib.llaisysROPE.restype = c_int
    lib.llaisysROPE.errcheck = check_status

    # Self Attention
    lib.llaisysSelfAttention.argtypes = [
        llaisysTensor_t,
        llaisysTensor_t,
        llaisysTensor_t,
        llaisysTensor_t,
        c_float,
    ]
    lib.llaisysSelfAttention.restype = c_int
    lib.llaisysSelfAttention.errcheck = check_status

    # SwiGLU
    lib.llaisysSwiGLU.argtypes = [
        llaisysTensor_t,
        llaisysTensor_t,
        llaisysTensor_t,
    ]
    lib.llaisysSwiGLU.restype = c_int
    lib.llaisysSwiGLU.errcheck = check_status
