// tensor_file.cpp: named-tensor container I/O (see include/tslm/tensor_file.h).
// Values are serialized byte by byte in little-endian order, so files are
// identical on any host.
#include "tslm/tensor_file.h"

#include "tslm/common.h"

#include <cstdio>
#include <cstring>

namespace tslm {

namespace {

size_t elem_size(DType t) {
    switch (t) {
        case DType::F32: return 4;
        case DType::F64: return 8;
        case DType::I32: return 4;
        case DType::I64: return 8;
    }
    return 0;
}

uint64_t load_le(const unsigned char* p, size_t n) {
    uint64_t v = 0;
    for (size_t k = 0; k < n; ++k) v |= static_cast<uint64_t>(p[k]) << (8 * k);
    return v;
}

void store_le(unsigned char* p, uint64_t v, size_t n) {
    for (size_t k = 0; k < n; ++k) p[k] = static_cast<unsigned char>(v >> (8 * k));
}

template <typename T, typename U>
void add_raw(TensorFile* tf, const std::string& name, DType dt, const std::vector<uint64_t>& dims, const T* v) {
    CHECK(tf->find(name) == nullptr, "tensor_file: duplicate tensor '%s'", name.c_str());
    NamedTensor t;
    t.name = name;
    t.dtype = dt;
    t.dims = dims;
    const uint64_t n = t.numel();
    t.data.resize(n * sizeof(T));
    for (uint64_t k = 0; k < n; ++k) {
        U bits;
        std::memcpy(&bits, &v[k], sizeof(T));  // bit pattern of one host scalar
        store_le(&t.data[k * sizeof(T)], static_cast<uint64_t>(bits), sizeof(T));
    }
    tf->tensors.push_back(std::move(t));
}

}  // namespace

uint64_t NamedTensor::numel() const {
    uint64_t n = 1;
    for (uint64_t d : dims) n *= d;
    return n;
}

double NamedTensor::f(uint64_t k) const {
    CHECK(k < numel(), "tensor '%s': index %llu out of range", name.c_str(), static_cast<unsigned long long>(k));
    if (dtype == DType::F32) {
        const uint32_t bits = static_cast<uint32_t>(load_le(&data[k * 4], 4));
        float v;
        std::memcpy(&v, &bits, 4);
        return v;
    }
    CHECK(dtype == DType::F64, "tensor '%s' is not floating point", name.c_str());
    const uint64_t bits = load_le(&data[k * 8], 8);
    double v;
    std::memcpy(&v, &bits, 8);
    return v;
}

int64_t NamedTensor::i(uint64_t k) const {
    CHECK(k < numel(), "tensor '%s': index %llu out of range", name.c_str(), static_cast<unsigned long long>(k));
    if (dtype == DType::I32) return static_cast<int32_t>(static_cast<uint32_t>(load_le(&data[k * 4], 4)));
    CHECK(dtype == DType::I64, "tensor '%s' is not an integer tensor", name.c_str());
    return static_cast<int64_t>(load_le(&data[k * 8], 8));
}

const NamedTensor* TensorFile::find(const std::string& name) const {
    for (const NamedTensor& t : tensors)
        if (t.name == name) return &t;
    return nullptr;
}

const NamedTensor& TensorFile::get(const std::string& name) const {
    const NamedTensor* t = find(name);
    CHECK(t != nullptr, "tensor_file: missing tensor '%s'", name.c_str());
    return *t;
}

void TensorFile::add_f32(const std::string& n, const std::vector<uint64_t>& d, const float* v) {
    add_raw<float, uint32_t>(this, n, DType::F32, d, v);
}
void TensorFile::add_f64(const std::string& n, const std::vector<uint64_t>& d, const double* v) {
    add_raw<double, uint64_t>(this, n, DType::F64, d, v);
}
void TensorFile::add_i32(const std::string& n, const std::vector<uint64_t>& d, const int32_t* v) {
    add_raw<int32_t, uint32_t>(this, n, DType::I32, d, v);
}
void TensorFile::add_i64(const std::string& n, const std::vector<uint64_t>& d, const int64_t* v) {
    add_raw<int64_t, uint64_t>(this, n, DType::I64, d, v);
}

bool save_tensor_file(const std::string& path, const TensorFile& tf, std::string* err) {
    std::vector<unsigned char> out = {'T', 'S', 'G', 'D'};
    auto put = [&](uint64_t v, size_t n) {
        const size_t at = out.size();
        out.resize(at + n);
        store_le(&out[at], v, n);
    };
    put(1, 4);
    put(tf.tensors.size(), 4);
    for (const NamedTensor& t : tf.tensors) {
        put(t.name.size(), 4);
        out.insert(out.end(), t.name.begin(), t.name.end());
        put(static_cast<uint32_t>(t.dtype), 4);
        put(t.dims.size(), 4);
        for (uint64_t d : t.dims) put(d, 8);
        out.insert(out.end(), t.data.begin(), t.data.end());
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        if (err) *err = "cannot open '" + path + "' for writing";
        return false;
    }
    const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
    if (std::fclose(f) != 0 || !ok) {
        if (err) *err = "error writing '" + path + "'";
        return false;
    }
    return true;
}

bool load_tensor_file(const std::string& path, TensorFile* tf, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = path + ": " + m;
        return false;
    };
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail("cannot open");
    std::vector<unsigned char> b;
    unsigned char buf[1 << 15];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) b.insert(b.end(), buf, buf + n);
    std::fclose(f);
    size_t p = 0;
    auto need = [&](size_t k) { return p + k <= b.size(); };
    auto get = [&](size_t k) {
        const uint64_t v = load_le(&b[p], k);
        p += k;
        return v;
    };
    if (!need(12) || std::memcmp(b.data(), "TSGD", 4) != 0) return fail("bad magic");
    p = 4;
    if (get(4) != 1) return fail("unsupported version");
    const uint64_t count = get(4);
    tf->tensors.clear();
    for (uint64_t k = 0; k < count; ++k) {
        NamedTensor t;
        if (!need(4)) return fail("truncated");
        const uint64_t nl = get(4);
        if (!need(nl)) return fail("truncated name");
        t.name.assign(reinterpret_cast<const char*>(&b[p]), nl);
        p += nl;
        if (!need(8)) return fail("truncated");
        const uint64_t dt = get(4);
        if (dt > 3) return fail("bad dtype for '" + t.name + "'");
        t.dtype = static_cast<DType>(dt);
        const uint64_t nd = get(4);
        if (nd > 8 || !need(8 * nd)) return fail("bad dims for '" + t.name + "'");
        for (uint64_t d = 0; d < nd; ++d) t.dims.push_back(get(8));
        const uint64_t bytes = t.numel() * elem_size(t.dtype);
        if (!need(bytes)) return fail("truncated data for '" + t.name + "'");
        t.data.assign(b.begin() + static_cast<std::ptrdiff_t>(p), b.begin() + static_cast<std::ptrdiff_t>(p + bytes));
        p += bytes;
        if (tf->find(t.name)) return fail("duplicate tensor '" + t.name + "'");
        tf->tensors.push_back(std::move(t));
    }
    if (p != b.size()) return fail("trailing bytes");
    return true;
}

}  // namespace tslm
