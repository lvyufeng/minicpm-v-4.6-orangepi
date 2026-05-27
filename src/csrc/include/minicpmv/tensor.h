#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minicpmv {

enum class DType {
    Float16,
    BFloat16,
    Float32,
    Int32,
    Int64,
    UInt8,
    Int8,
};

size_t dtype_size(DType dtype);
aclDataType to_acl_dtype(DType dtype);

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void allocate(size_t bytes);
    void reset();

    void* data() const { return data_; }
    size_t size_bytes() const { return size_bytes_; }

private:
    void* data_{nullptr};
    size_t size_bytes_{0};
};

class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<int64_t> shape, DType dtype);

    void allocate();
    void copy_from_host(const void* src, size_t bytes);
    void copy_to_host(void* dst, size_t bytes) const;

    const std::vector<int64_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t numel() const;
    size_t size_bytes() const;
    void* data() const { return buffer_.data(); }

private:
    std::vector<int64_t> shape_;
    DType dtype_{DType::Float16};
    DeviceBuffer buffer_;
};

}  // namespace minicpmv
