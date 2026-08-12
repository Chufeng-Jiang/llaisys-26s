import torch

DTYPE_MAP = {
    # 根据你自己的 enum 实际名字改
    LLAISYS_DTYPE_F32: torch.float32,
    LLAISYS_DTYPE_F16: torch.float16,
    LLAISYS_DTYPE_BF16: torch.bfloat16,
}
