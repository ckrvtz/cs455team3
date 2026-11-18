// gpu_check.cu: B0.1 machine verification for the Phase B target GPU
// (ROADMAP_CUDA.md §2.1, task B0.1). Built and run by scripts/gpu_check.sh
// with the project's planned nvcc flags; NOT part of the library or of
// `make` (B0.2 adds the real CUDA build).
//
// It prints "key: value" lines and ends with "RESULT: PASS" or "RESULT: FAIL"
// (exit status 0 / 1):
//   1. versions: CUDA runtime, driver (max supported runtime), cuBLAS;
//   2. device 0 properties: name, compute capability, SMs, global and free
//      memory, shared memory per block, kernel exec timeout (display
//      watchdog), clocks and bus width; a derived FP32 peak for cc 5.x;
//   3. a trivial grid-stride kernel (y = a*x + y) on 1M elements, checked
//      EXACTLY against the host (small-integer data, so FMA contraction
//      cannot change any result);
//   4. one cublasSgemm with the row-major mapping of ROADMAP_CUDA.md §5.4
//      for matmul_forward, out[N,OC] = inp[N,C] * W[OC,C]^T, on non-square
//      shapes (N=37, C=19, OC=23), checked EXACTLY against a host reference
//      (small-integer data: every product and sum is exact in float);
//   5. the W-A head shape (N=8192, C=128, OC=10000): three timed runs
//      (informational GFLOP/s, cudaEvent timing) and a bit-wise comparison
//      of the outputs (cuBLAS determinism on one stream, D-B3).
// Checks against D-B1: compute capability 5.0 and >= 3.5 GiB global memory
// are required; differences are reported as FAIL with the reason.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call)                                                                                     \
    do {                                                                                             \
        cudaError_t e_ = (call);                                                                     \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "CUDA error: %s\n  at %s:%d\n  %s\n", #call, __FILE__, __LINE__,    \
                         cudaGetErrorString(e_));                                                    \
            std::printf("RESULT: FAIL (CUDA error, see stderr)\n");                                  \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

#define CB(call)                                                                                     \
    do {                                                                                             \
        cublasStatus_t s_ = (call);                                                                  \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                           \
            std::fprintf(stderr, "cuBLAS error %d: %s\n  at %s:%d\n", static_cast<int>(s_), #call,   \
                         __FILE__, __LINE__);                                                        \
            std::printf("RESULT: FAIL (cuBLAS error, see stderr)\n");                                \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

namespace {

__global__ void saxpy_kernel(int n, float a, const float* x, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = a * x[i] + y[i];
}

int attr(cudaDeviceAttr a, int dev) {
    int v = 0;
    CK(cudaDeviceGetAttribute(&v, a, dev));
    return v;
}

// Host reference for the forward mapping: out[n,o] = sum_c inp[n,c] * W[o,c].
void host_forward(const std::vector<float>& inp, const std::vector<float>& W, std::vector<float>& out, int N, int C,
                  int OC) {
    for (int n = 0; n < N; ++n)
        for (int o = 0; o < OC; ++o) {
            float s = 0.0f;
            for (int c = 0; c < C; ++c) s += inp[size_t(n) * C + c] * W[size_t(o) * C + c];
            out[size_t(n) * OC + o] = s;
        }
}

// The §5.4 forward mapping: cuBLAS computes out^T[OC,N] = W * inp^T in its
// column-major view, i.e. opA = T on W (lda = C), opB = N on inp (ldb = C).
void sgemm_forward(cublasHandle_t h, const float* inp, const float* W, float* out, int N, int C, int OC) {
    const float one = 1.0f, zero = 0.0f;
    CB(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, OC, N, C, &one, W, C, inp, C, &zero, out, OC));
}

}  // namespace

int main() {
    bool ok = true;
    auto fail = [&](const char* why) {
        std::printf("check_failed: %s\n", why);
        ok = false;
    };

    // ---- 1. versions
    int rt = 0, drv = 0;
    CK(cudaRuntimeGetVersion(&rt));
    CK(cudaDriverGetVersion(&drv));
    std::printf("cuda_runtime_version: %d.%d\n", rt / 1000, (rt % 1000) / 10);
    std::printf("cuda_driver_max_runtime: %d.%d\n", drv / 1000, (drv % 1000) / 10);
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    std::printf("device_count: %d\n", ndev);
    if (ndev < 1) {
        std::printf("RESULT: FAIL (no CUDA device)\n");
        return 1;
    }
    const int dev = 0;
    CK(cudaSetDevice(dev));
    cublasHandle_t h;
    CB(cublasCreate(&h));
    int cublas_ver = 0;
    CB(cublasGetVersion(h, &cublas_ver));
    std::printf("cublas_version: %d (major %d, minor %d, patch %d)\n", cublas_ver, cublas_ver / 10000,
                (cublas_ver / 100) % 100, cublas_ver % 100);
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

    // ---- 2. device properties
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    const int sms = p.multiProcessorCount;
    const int clock_khz = attr(cudaDevAttrClockRate, dev);
    const int mem_clock_khz = attr(cudaDevAttrMemoryClockRate, dev);
    const int bus_bits = attr(cudaDevAttrGlobalMemoryBusWidth, dev);
    const int timeout = attr(cudaDevAttrKernelExecTimeout, dev);
    std::printf("device_name: %s\n", p.name);
    std::printf("compute_capability: %d.%d\n", p.major, p.minor);
    std::printf("multiprocessors: %d\n", sms);
    std::printf("global_memory_MiB: %.0f\n", double(total_b) / (1024.0 * 1024.0));
    std::printf("free_memory_MiB: %.0f\n", double(free_b) / (1024.0 * 1024.0));
    std::printf("shared_memory_per_block_KiB: %.0f\n", double(p.sharedMemPerBlock) / 1024.0);
    std::printf("max_threads_per_block: %d\n", p.maxThreadsPerBlock);
    std::printf("warp_size: %d\n", p.warpSize);
    std::printf("kernel_exec_timeout: %s\n", timeout ? "yes (display watchdog active)" : "no");
    std::printf("core_clock_MHz: %.0f\n", clock_khz / 1000.0);
    std::printf("memory_clock_MHz: %.0f\n", mem_clock_khz / 1000.0);
    std::printf("memory_bus_bits: %d\n", bus_bits);
    // Peak memory bandwidth = 2 (DDR) * memory clock * bus width / 8.
    std::printf("derived_peak_bandwidth_GBps: %.1f\n", 2.0 * mem_clock_khz * 1e3 * bus_bits / 8.0 / 1e9);
    if (p.major == 5) {  // Maxwell: 128 FP32 cores per SM, 2 FLOP per FMA
        std::printf("derived_fp32_peak_GFLOPs: %.0f (128 cores/SM x %d SMs x 2 x %.0f MHz)\n",
                    2.0 * 128 * sms * clock_khz / 1e6, sms, clock_khz / 1000.0);
    }
    if (!(p.major == 5 && p.minor == 0)) fail("compute capability is not 5.0 (D-B1 expects sm_50)");
    if (total_b < size_t(3584) * 1024 * 1024) fail("less than 3.5 GiB of global memory (D-B1 expects 4 GB)");
    if (free_b < size_t(2400) * 1024 * 1024)
        std::printf("note: less than 2.4 GiB free; `base` at B=32 (2,050 MB + context) may not fit (D-B1)\n");

    // ---- 3. trivial kernel, exact check
    {
        const int n = 1 << 20;
        std::vector<float> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = float(i % 97);
            y[i] = float((i * 7) % 101);
        }
        float *dx = nullptr, *dy = nullptr;
        CK(cudaMalloc(&dx, n * sizeof(float)));
        CK(cudaMalloc(&dy, n * sizeof(float)));
        CK(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dy, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        saxpy_kernel<<<120, 256>>>(n, 2.0f, dx, dy);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<float> got(n);
        CK(cudaMemcpy(got.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += got[i] != 2.0f * x[i] + y[i];
        std::printf("kernel_saxpy_1M: %s (%d mismatches)\n", bad ? "FAIL" : "exact", bad);
        if (bad) fail("trivial kernel result differs from the host");
        CK(cudaFree(dx));
        CK(cudaFree(dy));
    }

    // ---- 4. cublasSgemm, §5.4 forward mapping, exact check on small integers
    {
        const int N = 37, C = 19, OC = 23;
        std::vector<float> inp(size_t(N) * C), W(size_t(OC) * C), ref(size_t(N) * OC), got(size_t(N) * OC);
        for (size_t i = 0; i < inp.size(); ++i) inp[i] = float(int(i * 5 % 7) - 3);
        for (size_t i = 0; i < W.size(); ++i) W[i] = float(int(i * 3 % 7) - 3);
        host_forward(inp, W, ref, N, C, OC);
        float *di = nullptr, *dw = nullptr, *dout = nullptr;
        CK(cudaMalloc(&di, inp.size() * sizeof(float)));
        CK(cudaMalloc(&dw, W.size() * sizeof(float)));
        CK(cudaMalloc(&dout, got.size() * sizeof(float)));
        CK(cudaMemcpy(di, inp.data(), inp.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));
        sgemm_forward(h, di, dw, dout, N, C, OC);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (size_t i = 0; i < got.size(); ++i) bad += got[i] != ref[i];
        std::printf("sgemm_forward_mapping_37x19x23: %s (%d mismatches)\n", bad ? "FAIL" : "exact", bad);
        if (bad) fail("cublasSgemm with the §5.4 forward mapping differs from the host reference");
        CK(cudaFree(di));
        CK(cudaFree(dw));
        CK(cudaFree(dout));
    }

    // ---- 5. W-A head shape: timing (informational) and bit-wise repeatability
    {
        const int N = 8192, C = 128, OC = 10000;
        std::vector<float> inp(size_t(N) * C), W(size_t(OC) * C);
        unsigned s = 12345u;  // deterministic pseudo-random values in [-1, 1)
        auto rnd = [&]() {
            s = s * 1664525u + 1013904223u;
            return float(s >> 8) / float(1u << 23) - 1.0f;
        };
        for (float& v : inp) v = rnd();
        for (float& v : W) v = rnd();
        float *di = nullptr, *dw = nullptr, *dout = nullptr;
        const size_t out_n = size_t(N) * OC;
        CK(cudaMalloc(&di, inp.size() * sizeof(float)));
        CK(cudaMalloc(&dw, W.size() * sizeof(float)));
        CK(cudaMalloc(&dout, out_n * sizeof(float)));
        CK(cudaMemcpy(di, inp.data(), inp.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        std::vector<float> first(out_n), last(out_n);
        std::vector<float> ms(3);
        for (int r = 0; r < 3; ++r) {
            CK(cudaEventRecord(e0));
            sgemm_forward(h, di, dw, dout, N, C, OC);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms[r], e0, e1));
            if (r == 0) CK(cudaMemcpy(first.data(), dout, out_n * sizeof(float), cudaMemcpyDeviceToHost));
            if (r == 2) CK(cudaMemcpy(last.data(), dout, out_n * sizeof(float), cudaMemcpyDeviceToHost));
        }
        std::sort(ms.begin(), ms.end());
        const double flop = 2.0 * N * C * OC;
        std::printf("sgemm_head_8192x128x10000_ms: %.3f %.3f %.3f (sorted; first run includes warm-up)\n", ms[0],
                    ms[1], ms[2]);
        std::printf("sgemm_head_median_GFLOPs: %.0f (informational; not a benchmark)\n", flop / (ms[1] * 1e6));
        const bool same = std::memcmp(first.data(), last.data(), out_n * sizeof(float)) == 0;
        std::printf("sgemm_head_bitwise_repeatable: %s\n", same ? "yes" : "NO");
        if (!same) fail("cublasSgemm results differ between identical runs (D-B3)");
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
        CK(cudaFree(di));
        CK(cudaFree(dw));
        CK(cudaFree(dout));
    }

    CB(cublasDestroy(h));
    std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
