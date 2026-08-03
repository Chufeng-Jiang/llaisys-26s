#include "nvidia_resource.cuh"

#include "nvidia_common.cuh"

#include <cuda_runtime.h>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id)
    : DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {}

Resource::~Resource() {
  if (_argmax_packed_workspace == nullptr) {
    return;
  }

  // A destructor should not throw exceptions.
  (void)cudaSetDevice(getDeviceId());

  (void)cudaFree(_argmax_packed_workspace);

  _argmax_packed_workspace = nullptr;
}

unsigned long long *Resource::argmaxPackedWorkspace() {
  if (_argmax_packed_workspace != nullptr) {
    return _argmax_packed_workspace;
  }

  CUDA_CHECK(cudaSetDevice(getDeviceId()));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&_argmax_packed_workspace),
                        sizeof(unsigned long long)));

  return _argmax_packed_workspace;
}

} // namespace llaisys::device::nvidia