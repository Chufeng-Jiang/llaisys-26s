#include "nvidia_blas.cuh"

#include "nvidia_common.cuh"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

class CublasResource final {
public:
    explicit CublasResource(int device_id) : _device_id(device_id) {
        CUDA_CHECK(cudaSetDevice(_device_id));

        cublasHandle_t handle = nullptr;

        CUBLAS_CHECK(cublasCreate(&handle));

        try {
            CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

            // Operators select their required compute mode explicitly.
            CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        } catch (...) {
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

} // namespace

namespace llaisys::device::nvidia {

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

} // namespace llaisys::device::nvidia