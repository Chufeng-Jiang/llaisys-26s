from llaisys.libllaisys import DataType, DeviceType
from llaisys.tensor import Tensor
from llaisys.triton.tensor import as_nvidia_triton_tensor

x = Tensor(shape=(2, 3), dtype=DataType.F32, device=DeviceType.NVIDIA, device_id=0)

print("LLAISYS Tensor:")
print("  shape:", x.shape())
print("  dtype:", x.dtype())
print("  device:", x.device_type(), x.device_id())
print("  pointer:", hex(x.data_ptr()))

tx = as_nvidia_triton_tensor(x)

print("Triton Tensor:")
print("  dtype:", tx.dtype)
print("  pointer:", hex(tx.data_ptr()))

assert x.data_ptr() != 0
assert tx.data_ptr() == x.data_ptr()

print("Pointer bridge OK")
