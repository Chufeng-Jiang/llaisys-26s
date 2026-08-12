import llaisys
import torch


runtime = llaisys.RuntimeAPI(
    llaisys.DeviceType.NVIDIA
)

stream_ptr = runtime.get_context_stream(
    device_id=0
)

print(
    "LLAISYS runtime stream:",
    hex(stream_ptr),
)

assert stream_ptr != 0


external_stream = torch.cuda.ExternalStream(
    stream_ptr,
    device=0,
)

print(
    "PyTorch external stream:",
    hex(external_stream.cuda_stream),
)

assert (
    int(external_stream.cuda_stream)
    == stream_ptr
)

print("Stream pointer bridge OK")