// tensor_file.h: named-tensor container for golden test data (ROADMAP §10).
//
// Written by the dump_init app (C++) and dev_only/gen_golden.py (offline
// Python); read by tests with no Python at test time. Little-endian:
//   "TSGD"  u32 version (=1)  u32 count
//   count x { u32 name_len, name bytes (UTF-8, no NUL),
//             u32 dtype (0 = f32, 1 = f64, 2 = i32, 3 = i64),
//             u32 ndim, ndim x u64 dims, prod(dims) x element (LE) }
// Names are unique. A 0-dim tensor (ndim = 0) holds one element. Host-only.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tslm {

enum class DType : uint32_t { F32 = 0, F64 = 1, I32 = 2, I64 = 3 };

struct NamedTensor {
    std::string name;
    DType dtype = DType::F64;
    std::vector<uint64_t> dims;
    std::vector<unsigned char> data;  // raw little-endian elements

    uint64_t numel() const;
    // Element i converted to double / int64 (CHECK-fails on a dtype mismatch).
    double f(uint64_t i) const;   // F32 or F64
    int64_t i(uint64_t i) const;  // I32 or I64
};

struct TensorFile {
    std::vector<NamedTensor> tensors;

    const NamedTensor* find(const std::string& name) const;  // nullptr if absent
    const NamedTensor& get(const std::string& name) const;   // CHECK-fails if absent

    void add_f32(const std::string& name, const std::vector<uint64_t>& dims, const float* v);
    void add_f64(const std::string& name, const std::vector<uint64_t>& dims, const double* v);
    void add_i32(const std::string& name, const std::vector<uint64_t>& dims, const int32_t* v);
    void add_i64(const std::string& name, const std::vector<uint64_t>& dims, const int64_t* v);
};

bool save_tensor_file(const std::string& path, const TensorFile& tf, std::string* err);
bool load_tensor_file(const std::string& path, TensorFile* tf, std::string* err);

}  // namespace tslm
