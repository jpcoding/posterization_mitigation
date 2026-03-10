#ifndef PM_QCAT_SSIM_BLAS_HPP
#define PM_QCAT_SSIM_BLAS_HPP

#include <cblas.h>
#include "utils/qcat_ssim.hpp"

namespace PM
{
    // =========================================================================
    // BLAS-accelerated SSIM (3-D).  Two variants:
    //
    //  1. SSIM_3d_windowed_blas  – same window-by-window loop as the original,
    //     but copies each window into a flat contiguous buffer and uses
    //     cblas_ddot for the variance / covariance terms.  Per-window dynamic
    //     range is kept identical to the original.
    //
    //  2. SSIM_3d_sat  – uses a sliding 2-D summed-area table (integral image)
    //     so that each window query is O(1) after an O(N²) slab-preprocessing
    //     step.  Reduces total work from O(nw·W³) to O(N³), which is roughly
    //     a 10-40× algorithmic speedup for W=7, shift=2.
    //     NOTE: uses the GLOBAL data range (max−min over the whole field) for
    //     the SSIM stability constants c1/c2, matching the standard SSIM
    //     definition rather than the per-window range used by the original.
    // =========================================================================

    // ------------------------------------------------------------------
    // Helper: compute a single 3-D window using flat buffers + cblas_ddot
    // xbuf/ybuf must have capacity >= windowSize0*windowSize1*windowSize2
    // ones must be a ones-vector of the same length
    // ------------------------------------------------------------------
    template <class T>
    double SSIM_3d_calcWindow_blas(
        const T *data, const T *other,
        size_t size1, size_t size0,
        int offset0, int offset1, int offset2,
        int windowSize0, int windowSize1, int windowSize2,
        double *xbuf, double *ybuf, const double *ones)
    {
        const int np = windowSize0 * windowSize1 * windowSize2;
        int idx = 0;

        // Copy window to contiguous buffers; accumulate sum and min/max in one pass
        double xMin, xMax, xSum = 0, ySum = 0;
        xMin = xMax = (double)data[offset0 + size0 * (offset1 + size1 * (size_t)offset2)];

        for (int i2 = offset2; i2 < offset2 + windowSize2; i2++) {
            for (int i1 = offset1; i1 < offset1 + windowSize1; i1++) {
                const T *xrow = &data[offset0 + size0 * (i1 + size1 * i2)];
                const T *yrow = &other[offset0 + size0 * (i1 + size1 * i2)];
                for (int i0 = 0; i0 < windowSize0; i0++, idx++) {
                    double xv = xrow[i0];
                    double yv = yrow[i0];
                    xbuf[idx] = xv;
                    ybuf[idx] = yv;
                    xSum += xv;
                    ySum += yv;
                    if (xv < xMin) xMin = xv;
                    if (xv > xMax) xMax = xv;
                }
            }
        }

        double xMean = xSum / np;
        double yMean = ySum / np;

        // Subtract means: buf[i] += alpha * ones[i]  →  buf[i] -= mean
        cblas_daxpy(np, -xMean, ones, 1, xbuf, 1);
        cblas_daxpy(np, -yMean, ones, 1, ybuf, 1);

        // Variance & covariance via BLAS ddot (uses AVX2/AVX512 internally)
        double var_x  = cblas_ddot(np, xbuf, 1, xbuf, 1) / np;
        double var_y  = cblas_ddot(np, ybuf, 1, ybuf, 1) / np;
        double cov_xy = cblas_ddot(np, xbuf, 1, ybuf, 1) / np;

        double xSigma = std::sqrt(std::max(0.0, var_x));
        double ySigma = std::sqrt(std::max(0.0, var_y));

        double c1, c2;
        double range = xMax - xMin;
        if (range == 0.0) { c1 = K1 * K1; c2 = K2 * K2; }
        else              { c1 = K1 * K1 * range * range; c2 = K2 * K2 * range * range; }
        double c3 = c2 / 2.0;

        double luminance = (2.0 * xMean * yMean + c1) / (xMean * xMean + yMean * yMean + c1);
        double contrast  = (2.0 * xSigma * ySigma + c2) / (var_x + var_y + c2);
        double structure = (cov_xy + c3) / (xSigma * ySigma + c3);
        return luminance * contrast * structure;
    }

    // ------------------------------------------------------------------
    // Windowed 3-D SSIM using BLAS for per-window variance/covariance
    // ------------------------------------------------------------------
    template <class T>
    double SSIM_3d_windowed_blas(
        T *oriData, T *decData,
        size_t size2, size_t size1, size_t size0,
        int windowSize0, int windowSize1, int windowSize2,
        int windowShift0, int windowShift1, int windowShift2)
    {
        const int np = windowSize0 * windowSize1 * windowSize2;
        const int num_threads = 8;

        const size_t max_offset2 = size2 - windowSize2;
        const size_t max_offset1 = size1 - windowSize1;
        const size_t max_offset0 = size0 - windowSize0;

        double ssimSum = 0;
        size_t nw = 0;

        // Tell OpenBLAS to use a single thread per call so OMP controls parallelism
        openblas_set_num_threads(1);

        #pragma omp parallel num_threads(num_threads) reduction(+:ssimSum,nw)
        {
            // One set of flat buffers per thread, reused across all windows
            std::vector<double> xbuf(np), ybuf(np), ones(np, 1.0);

            #pragma omp for collapse(3) schedule(static)
            for (size_t offset2 = 0; offset2 <= max_offset2; offset2 += windowShift2) {
                for (size_t offset1 = 0; offset1 <= max_offset1; offset1 += windowShift1) {
                    for (size_t offset0 = 0; offset0 <= max_offset0; offset0 += windowShift0) {
                        nw++;
                        ssimSum += SSIM_3d_calcWindow_blas(
                            oriData, decData, size1, size0,
                            offset0, offset1, offset2,
                            windowSize0, windowSize1, windowSize2,
                            xbuf.data(), ybuf.data(), ones.data());
                    }
                }
            }
        }
        std::cout << "nw = " << nw << std::endl;
        return ssimSum / nw;
    }

    // ------------------------------------------------------------------
    // SSIM_3d_sat: sliding 2-D summed-area table approach.
    // Complexity: O(N³) vs O(nw·W³) for the original.
    // Uses global data range for c1/c2 (standard SSIM definition).
    // ------------------------------------------------------------------
    template <class T>
    double SSIM_3d_sat(
        const T *oriData, const T *decData,
        size_t size2, size_t size1, size_t size0,
        int windowSize0, int windowSize1, int windowSize2,
        int windowShift0, int windowShift1, int windowShift2)
    {
        const size_t NXY = size0 * size1;
        const size_t N   = size0 * size1 * size2;
        const int np     = windowSize0 * windowSize1 * windowSize2;
        const int num_threads = 8;

        // Global data range for c1, c2 (standard SSIM)
        T gMin = oriData[0], gMax = oriData[0];
        #pragma omp parallel for num_threads(num_threads) reduction(min:gMin) reduction(max:gMax) schedule(static)
        for (size_t i = 0; i < N; i++) {
            if (oriData[i] < gMin) gMin = oriData[i];
            if (oriData[i] > gMax) gMax = oriData[i];
        }
        double L  = (double)(gMax - gMin);
        double c1 = K1 * K1 * L * L;
        double c2 = K2 * K2 * L * L;
        double c3 = c2 / 2.0;

        // 2-D slab sum arrays: slab_*(y, x) = sum_{z in current window} data(z, y, x)
        std::vector<double> slab_x(NXY,0), slab_y(NXY,0),
                            slab_xx(NXY,0), slab_yy(NXY,0), slab_xy(NXY,0);

        // Initialise slab for z = 0 .. windowSize2-1
        for (int iz = 0; iz < windowSize2; iz++) {
            const T *xz = &oriData[iz * NXY];
            const T *yz = &decData[iz * NXY];
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (size_t i = 0; i < NXY; i++) {
                double xv = xz[i], yv = yz[i];
                slab_x[i]  += xv;
                slab_y[i]  += yv;
                slab_xx[i] += xv * xv;
                slab_yy[i] += yv * yv;
                slab_xy[i] += xv * yv;
            }
        }

        // SAT arrays (same footprint as slab, rebuilt each z-step)
        std::vector<double> sat_x(NXY), sat_y(NXY),
                            sat_xx(NXY), sat_yy(NXY), sat_xy(NXY);

        auto build_2d_sat = [&](std::vector<double> &sat, const std::vector<double> &src) {
            std::copy(src.begin(), src.end(), sat.begin());
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (size_t iy = 0; iy < size1; iy++) {
                double *row = &sat[iy * size0];
                for (size_t ix = 1; ix < size0; ix++) row[ix] += row[ix - 1];
            }
            for (size_t iy = 1; iy < size1; iy++) {
                const double *prev = &sat[(iy - 1) * size0];
                double       *curr = &sat[iy       * size0];
                for (size_t ix = 0; ix < size0; ix++) curr[ix] += prev[ix];
            }
        };

        auto sat_query = [&](const std::vector<double> &sat,
                             int x1, int x2, int y1, int y2) -> double {
            auto at = [&](int y, int x) -> double {
                if (y < 0 || x < 0) return 0.0;
                return sat[(size_t)y * size0 + (size_t)x];
            };
            return at(y2, x2) - at(y1 - 1, x2) - at(y2, x1 - 1) + at(y1 - 1, x1 - 1);
        };

        double ssimSum = 0.0;
        size_t nw = 0;

        const size_t max_offset2 = size2 - windowSize2;
        const size_t max_offset1 = size1 - windowSize1;
        const size_t max_offset0 = size0 - windowSize0;

        for (size_t offset2 = 0; offset2 <= max_offset2; offset2 += windowShift2) {
            build_2d_sat(sat_x,  slab_x);
            build_2d_sat(sat_y,  slab_y);
            build_2d_sat(sat_xx, slab_xx);
            build_2d_sat(sat_yy, slab_yy);
            build_2d_sat(sat_xy, slab_xy);

            double slice_ssim = 0.0;
            size_t slice_nw   = 0;

            #pragma omp parallel for collapse(2) num_threads(num_threads) reduction(+:slice_ssim,slice_nw) schedule(static)
            for (size_t offset1 = 0; offset1 <= max_offset1; offset1 += windowShift1) {
                for (size_t offset0 = 0; offset0 <= max_offset0; offset0 += windowShift0) {
                    int x1 = (int)offset0, x2 = x1 + windowSize0 - 1;
                    int y1 = (int)offset1, y2 = y1 + windowSize1 - 1;

                    double sum_x  = sat_query(sat_x,  x1, x2, y1, y2);
                    double sum_y  = sat_query(sat_y,  x1, x2, y1, y2);
                    double sum_xx = sat_query(sat_xx, x1, x2, y1, y2);
                    double sum_yy = sat_query(sat_yy, x1, x2, y1, y2);
                    double sum_xy = sat_query(sat_xy, x1, x2, y1, y2);

                    double mean_x = sum_x / np;
                    double mean_y = sum_y / np;
                    double var_x  = std::max(0.0, sum_xx / np - mean_x * mean_x);
                    double var_y  = std::max(0.0, sum_yy / np - mean_y * mean_y);
                    double cov_xy = sum_xy / np - mean_x * mean_y;

                    double xSigma = std::sqrt(var_x);
                    double ySigma = std::sqrt(var_y);

                    double luminance = (2.0 * mean_x * mean_y + c1) / (mean_x * mean_x + mean_y * mean_y + c1);
                    double contrast  = (2.0 * xSigma * ySigma + c2) / (var_x + var_y + c2);
                    double structure = (cov_xy + c3) / (xSigma * ySigma + c3);

                    slice_ssim += luminance * contrast * structure;
                    slice_nw++;
                }
            }

            ssimSum += slice_ssim;
            nw      += slice_nw;

            if (offset2 + windowShift2 <= max_offset2) {
                for (int s = 0; s < windowShift2; s++) {
                    size_t z_old = offset2 + s;
                    size_t z_new = offset2 + windowSize2 + s;
                    const T *xo = &oriData[z_old * NXY];
                    const T *yo = &decData[z_old * NXY];
                    const T *xn = &oriData[z_new * NXY];
                    const T *yn = &decData[z_new * NXY];
                    #pragma omp parallel for num_threads(num_threads) schedule(static)
                    for (size_t i = 0; i < NXY; i++) {
                        double xov = xo[i], yov = yo[i];
                        double xnv = xn[i], ynv = yn[i];
                        slab_x[i]  += xnv - xov;
                        slab_y[i]  += ynv - yov;
                        slab_xx[i] += xnv * xnv - xov * xov;
                        slab_yy[i] += ynv * ynv - yov * yov;
                        slab_xy[i] += xnv * ynv - xov * yov;
                    }
                }
            }
        }

        std::cout << "nw = " << nw << std::endl;
        return ssimSum / nw;
    }

    // ------------------------------------------------------------------
    // Top-level dispatcher (BLAS flat-buffer variant)
    // ------------------------------------------------------------------
    template <class T>
    double calculateSSIM_blas(T *oriData, T *decData, int dim, size_t *dims)
    {
        const int windowSize0 = 7, windowSize1 = 7, windowSize2 = 7;
        const int windowShift0 = 2, windowShift1 = 2, windowShift2 = 2;
        double result = -1;
        switch (dim) {
        case 1:
            result = SSIM_1d_windowed(oriData, decData, dims[0], windowSize0, windowShift0);
            break;
        case 2:
            result = SSIM_2d_windowed(oriData, decData, dims[0], dims[1], windowSize0, windowSize1, windowShift0, windowShift1);
            break;
        case 3:
            result = SSIM_3d_windowed_blas(oriData, decData, dims[0], dims[1], dims[2], windowSize0, windowSize1, windowSize2, windowShift0, windowShift1, windowShift2);
            break;
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Top-level dispatcher (SAT / integral-image variant)
    // ------------------------------------------------------------------
    template <class T>
    double calculateSSIM_sat(T *oriData, T *decData, int dim, size_t *dims)
    {
        const int windowSize0 = 7, windowSize1 = 7, windowSize2 = 7;
        const int windowShift0 = 2, windowShift1 = 2, windowShift2 = 2;
        double result = -1;
        switch (dim) {
        case 3:
            result = SSIM_3d_sat(oriData, decData, dims[0], dims[1], dims[2], windowSize0, windowSize1, windowSize2, windowShift0, windowShift1, windowShift2);
            break;
        default:
            result = calculateSSIM(oriData, decData, dim, dims);
            break;
        }
        return result;
    }

} // namespace PM
#endif
