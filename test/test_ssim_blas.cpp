// test_ssim_blas.cpp
// Benchmark and correctness check: original SSIM vs BLAS-flat-buffer vs SAT.
//
// Usage:
//   ./test_ssim_blas <file_a.f32> <file_b.f32> <nx> <ny> <nz>
//
// Example (two nyx 512³ fields):
//   ./test_ssim_blas /home/jp/data/nyx_512x512x512/velocity_x.f32 \
//                   /home/jp/data/nyx_512x512x512/velocity_y.f32  \
//                   512 512 512

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "SZ3/utils/FileUtil.hpp"
#include "utils/timer.hpp"
#include "utils/qcat_ssim_blas.hpp"

int main(int argc, char **argv)
{
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <file_a.f32> <file_b.f32> <nx> <ny> <nz>\n";
        return 1;
    }

    const char *file_a = argv[1];
    const char *file_b = argv[2];
    const size_t nx = std::atoi(argv[3]);
    const size_t ny = std::atoi(argv[4]);
    const size_t nz = std::atoi(argv[5]);
    const size_t N  = nx * ny * nz;

    std::vector<float> a(N), b(N);
    SZ3::readfile(file_a, N, a.data());
    SZ3::readfile(file_b, N, b.data());
    std::cout << "Loaded " << N << " elements (" << N * 4 / 1024 / 1024 << " MB each)\n";

    size_t dims[3] = {nz, ny, nx};  // SZ/QCAT convention: dims[0] is outermost
    Timer t;

    // ------------------------------------------------------------------
    // 1. Original (2-pass per window, OMP parallel outer loops)
    // ------------------------------------------------------------------
    std::cout << "\n=== Original SSIM ===\n";
    t.start();
    double ssim_orig = PM::calculateSSIM(a.data(), b.data(), 3, dims);
    double time_orig = t.stop();
    std::printf("  SSIM = %.8f   time = %.3f s\n", ssim_orig, time_orig);

    // ------------------------------------------------------------------
    // 2. BLAS flat-buffer (same per-window range for c1/c2)
    // ------------------------------------------------------------------
    std::cout << "\n=== BLAS flat-buffer SSIM ===\n";
    t.start();
    double ssim_blas = PM::calculateSSIM_blas(a.data(), b.data(), 3, dims);
    double time_blas = t.stop();
    std::printf("  SSIM = %.8f   time = %.3f s\n", ssim_blas, time_blas);
    std::printf("  delta vs original = %.2e\n", ssim_blas - ssim_orig);
    std::printf("  speedup = %.2fx\n", time_orig / time_blas);

    // ------------------------------------------------------------------
    // 3. SAT / integral-image (global range for c1/c2)
    // ------------------------------------------------------------------
    std::cout << "\n=== SAT (sliding slab integral image) SSIM ===\n";
    t.start();
    double ssim_sat = PM::calculateSSIM_sat(a.data(), b.data(), 3, dims);
    double time_sat = t.stop();
    std::printf("  SSIM = %.8f   time = %.3f s\n", ssim_sat, time_sat);
    std::printf("  delta vs original = %.2e  (expected: small, c1/c2 differ slightly)\n",
                ssim_sat - ssim_orig);
    std::printf("  speedup = %.2fx\n", time_orig / time_sat);

    return 0;
}
