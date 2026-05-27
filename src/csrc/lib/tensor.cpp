#include "minicpmv/tensor.h"
#include "minicpmv/acl_context.h"

#include <numeric>
#include <stdexcept>

namespace minicpmv {

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Float32: return 4;
        case DType::Int32: return 4;
        case DType::Int64: return 8;
        case DType::UInt8: return 1;
        case DType::Int8: return 1;
    }
    throw std::runtime_error("unknown dtype");
}

aclDataType to_acl_dtype(DType dtype) {
    switch (dtype) {
        case DType::Float16: return ACL_FLOAT16;
        case DType::BFloat16: return ACL_BF16;
        case DType::Float32: return ACL_FLOAT;
        case DType::Int32: return ACL_INT32;
        case DType::Int64: return ACL_INT64;
        case DType::UInt8: return ACL_UINT8;
        case DType::Int8: return ACL_INT8;
    }
    throw std::runtime_error("unknown dtype");
}

DeviceBuffer::DeviceBuffer(size_t bytes) {
    allocate(bytes);
}

DeviceBuffer::~DeviceBuffer() {
    reset();
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept {
    data_ = other.data_;
    size_bytes_ = other.size_bytes_;
    other.data_ = nullptr;
    other.size_bytes_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = other.data_;
        size_bytes_ = other.size_bytes_;
        other.data_ = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

void DeviceBuffer::allocate(size_t bytes) {
    reset();
    if (bytes == 0) {
        return;
    }
    check_acl(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    size_bytes_ = bytes;
}

void DeviceBuffer::reset() {
    if (data_ != nullptr) {
        aclrtFree(data_);
        data_ = nullptr;
        size_bytes_ = 0;
    }
}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype) : shape_(std::move(shape)), dtype_(dtype) {}

void Tensor::allocate() {
    buffer_.allocate(size_bytes());
}

void Tensor::copy_from_host(const void* src, size_t bytes) {
    if (buffer_.data() == nullptr) {
        allocate();
    }
    check_acl(aclrtMemcpy(buffer_.data(), size_bytes(), src, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
}

void Tensor::copy_to_host(void* dst, size_t bytes) const {
    check_acl(aclrtMemcpy(dst, bytes, buffer_.data(), bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
}

size_t Tensor::numel() const {
    if (shape_.empty()) {
        return 0;
    }
    return std::accumulate(shape_.begin(), shape_.end(), size_t{1}, [](size_t a, int64_t b) {
        return a * static_cast<size_t>(b);
    });
}

size_t Tensor::size_bytes() const {
    return numel() * dtype_size(dtype_);
}

}  // namespace minicpmv
