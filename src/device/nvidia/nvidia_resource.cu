#include "nvidia_resource.cuh"

#include "nvidia_common.cuh"

#include <cuda_runtime.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

// One pool exists per host thread. Each CUDA device used by that thread gets
// its own cuBLAS handle because a handle is associated with the CUDA device
// that is current when cublasCreate() is called.
class CublasResource final {
public:
    explicit CublasResource(int device_id) : _device_id(device_id) {
        CUDA_CHECK(cudaSetDevice(_device_id));

        cublasHandle_t handle = nullptr;

        CUBLAS_CHECK(cublasCreate(&handle));

        try {
            CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

            // Use explicit FP32 compute modes selected by operators instead of
            // enabling a global fast-math mode on the handle.
            CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        } catch (...) {
            // The constructor is still running on _device_id here.
            // Best-effort cleanup is sufficient before rethrowing.
            (void)cublasDestroy(handle);

            throw;
        }

        _handle = handle;
    }

    ~CublasResource() noexcept {
        if (_handle == nullptr) { return; }

        llaisys::device::nvidia::run_on_cuda_device_noexcept(_device_id, [this]() noexcept {
            (void)cublasDestroy(_handle);

            _handle = nullptr;
        });
    }

    CublasResource(const CublasResource &) = delete;
    CublasResource &operator=(const CublasResource &) = delete;

    CublasResource(CublasResource &&) = delete;
    CublasResource &operator=(CublasResource &&) = delete;

    cublasHandle_t handle() const noexcept { return _handle; }

private:
    int _device_id{-1};
    cublasHandle_t _handle{nullptr};
};

thread_local std::unordered_map<int, std::unique_ptr<CublasResource>> cublas_resources;

// CUDA device capabilities are effectively static. Keep one cached copy
// per host thread and device, matching the thread-local NVIDIA resource
// model used by the cuBLAS handle cache.
thread_local std::unordered_map<int, cudaDeviceProp> device_properties_cache;

} // namespace

namespace llaisys::device::nvidia {

// ============================================================
// Runtime-owned Resource implementation
// ============================================================

Resource::Resource(int device_id) : DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {}

Resource::~Resource() noexcept {
    if (_argmax_packed_workspace == nullptr) { return; }

    run_on_cuda_device_noexcept(getDeviceId(), [this]() noexcept {
        (void)cudaFree(_argmax_packed_workspace);

        _argmax_packed_workspace = nullptr;
    });
}

unsigned long long *Resource::argmaxPackedWorkspace() {
    if (_argmax_packed_workspace != nullptr) { return _argmax_packed_workspace; }

    CUDA_CHECK(cudaSetDevice(getDeviceId()));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&_argmax_packed_workspace), sizeof(unsigned long long)));

    return _argmax_packed_workspace;
}

// ============================================================
// cuBLAS error handling
// ============================================================

const char *cublas_status_name(cublasStatus_t status) noexcept {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

void check_cublas(
    cublasStatus_t status,
    const char *expression,
    const char *file,
    int line,
    const char *function) {
    if (status == CUBLAS_STATUS_SUCCESS) { return; }

    std::ostringstream message;

    message << "cuBLAS error: " << cublas_status_name(status) << " while executing " << expression
            << " in " << function << " at " << file << ':' << line;

    throw std::runtime_error(message.str());
}

// ============================================================
// Reusable cuBLAS handle
// ============================================================

cublasHandle_t get_cublas_handle(cudaStream_t stream) {
    int device_id = -1;

    CUDA_CHECK(cudaGetDevice(&device_id));

    auto iterator = cublas_resources.find(device_id);

    if (iterator == cublas_resources.end()) {
        auto resource = std::make_unique<CublasResource>(device_id);

        iterator = cublas_resources.emplace(device_id, std::move(resource)).first;
    }

    cublasHandle_t handle = iterator->second->handle();

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    return handle;
}

const cudaDeviceProp &get_device_properties(int device_id) {
    auto iterator = device_properties_cache.find(device_id);

    if (iterator != device_properties_cache.end()) { return iterator->second; }

    cudaDeviceProp properties{};

    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));

    auto [inserted_iterator, inserted] = device_properties_cache.emplace(device_id, properties);

    (void)inserted;

    return inserted_iterator->second;
}

} // namespace llaisys::device::nvidia