#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <stdexcept>
#include <string>

namespace minicpmv {

inline void check_acl(aclError code, const char* expr) {
    if (code != ACL_SUCCESS) {
        throw std::runtime_error(std::string(expr) + " failed with aclError=" + std::to_string(code));
    }
}

class AclContext {
public:
    explicit AclContext(int device_id = 0);
    ~AclContext();

    AclContext(const AclContext&) = delete;
    AclContext& operator=(const AclContext&) = delete;

    int device_id() const { return device_id_; }
    aclrtStream stream() const { return stream_; }

private:
    int device_id_;
    aclrtStream stream_{};
    bool initialized_{false};
    bool device_set_{false};
};

}  // namespace minicpmv
