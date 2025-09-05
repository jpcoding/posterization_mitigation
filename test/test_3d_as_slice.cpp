#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
// #include "SZ3/api/sz.hpp"
#include <omp.h>

#include <algorithm>
#include <vector>

#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "compensation.hpp"
#include "utils/qcat_ssim.hpp"
#include "utils/timer.hpp"
#include <unordered_map>

using Real = float;
namespace SZ = SZ3;

namespace fs = std::filesystem;

template <typename Type>
void verify(Type *ori_data, Type *data, size_t num_elements, double &psnr, double &nrmse, double &max_diff) {
    size_t i = 0;
    double Max = ori_data[0];
    double Min = ori_data[0];
    max_diff = fabs(data[0] - ori_data[0]);
    double diff_sum = 0;
    double maxpw_relerr = 0;
    double sum1 = 0, sum2 = 0, l2sum = 0;
    for (i = 0; i < num_elements; i++) {
        sum1 += ori_data[i];
        sum2 += data[i];
        l2sum += data[i] * data[i];
    }
    double mean1 = sum1 / num_elements;
    double mean2 = sum2 / num_elements;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0;

    double *diff = (double *)malloc(num_elements * sizeof(double));

    for (i = 0; i < num_elements; i++) {
        diff[i] = data[i] - ori_data[i];
        diff_sum += data[i] - ori_data[i];
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        double err = fabs(data[i] - ori_data[i]);
        if (ori_data[i] != 0) {
            relerr = err / fabs(ori_data[i]);
            if (maxpw_relerr < relerr) maxpw_relerr = relerr;
        }

        if (max_diff < err) max_diff = err;
        prodSum += (ori_data[i] - mean1) * (data[i] - mean2);
        sum3 += (ori_data[i] - mean1) * (ori_data[i] - mean1);
        sum4 += (data[i] - mean2) * (data[i] - mean2);
        sum += err * err;
    }
    double std1 = sqrt(sum3 / num_elements);
    double std2 = sqrt(sum4 / num_elements);
    double ee = prodSum / num_elements;
    double acEff = ee / std1 / std2;

    double mse = sum / num_elements;
    double sse = sum;  // sum of square error
    double range = Max - Min;
    psnr = 20 * log10(range) - 10 * log10(mse);
    nrmse = sqrt(mse) / range;

    double normErr = sqrt(sum);
    double normErr_norm = normErr / sqrt(l2sum);

    printf("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf("Max absolute error = %.2G\n", max_diff);
    printf("Max relative error = %.2G\n", max_diff / (Max - Min));
    printf("Max pw relative error = %.2G\n", maxpw_relerr);
    printf("PSNR = %f, NRMSE= %.10G\n", psnr, nrmse);
    printf("normError = %f, normErr_norm = %f\n", normErr, normErr_norm);
    printf("acEff=%f\n", acEff);
    printf("SSE=%f\n", sse);
    printf("MSE=%f\n", mse);
    //        printf("errAutoCorr=%.10f\n", autocorrelation1DLag1<double>(diff, num_elements, diff_sum / num_elements));
    free(diff);
}

int main(int argc, char **argv) {
    int N = atoi(argv[1]);
    std::vector<int> dims(N);
    size_t data_size = 1;
    for (int i = 0; i < N; i++) {
        dims[i] = atoi(argv[i + 2]);
        data_size *= dims[i];
    }

    bool use_rbf = atoi(argv[N+6]) != 0;



    std::vector<Real> original_data(data_size, 0);
    readfile(argv[N + 2], data_size, original_data.data());
    double rel_eb = atof(argv[N + 3]);

    // make a copy of the original data
    std::vector<Real> dec_data(data_size, 0);
    std::copy(original_data.begin(), original_data.end(), dec_data.begin());

    std::vector<int> quant_inds(data_size, 0);

    float max = *std::max_element(original_data.begin(), original_data.end());
    float min = *std::min_element(original_data.begin(), original_data.end());
    printf("max: %f, min: %f\n", max, min);
    double eb = rel_eb * (max - min);
    printf("relative eb: %.6f\n", rel_eb);
    printf("absolute eb: %.6f\n", eb);
    // create a linear quantizer
    auto quantizer = SZ::LinearQuantizer<float>();
    quantizer.set_eb(eb);
    // iterate the input data and quantize it
    // use a dict to get the most frequent quantization index
    std::unordered_map<int, size_t> quant_freq_map;
    for (size_t i = 0; i < data_size; i++) {
        quant_inds[i] = quantizer.quantize_and_overwrite(dec_data[i], 0) - 32768;
        if(quant_inds[i] == -32768){
            quant_inds[i] = 0; // if the quantization result is -32768, set it to 0 
        }
        // get the most frequent quantization index
        quant_freq_map[quant_inds[i]]++;
    }
    // find the most frequent quantization index
    int most_frequent_quant_index = 0;
    size_t max_freq = 0;
    for (const auto &pair : quant_freq_map) {
        if (pair.second > max_freq) {
            max_freq = pair.second;
            most_frequent_quant_index = pair.first;
        }
    }
    


    writefile(argv[N + 4], dec_data.data(), data_size);
    writefile("quant_index.i32", quant_inds.data(), data_size);
    // verify the data
    double psnr, nrmse, max_diff;
    verify(original_data.data(), dec_data.data(), data_size, psnr, nrmse, max_diff);

    // cast dims to size_t
    std::vector<size_t> dims_(data_size);
    for (int i = 0; i < N; i++) {
        dims_[i] = dims[i];
    }
    auto ssim = PM::calculateSSIM(original_data.data(), dec_data.data(), N, dims_.data());
    printf("SSIM = %f\n", ssim);

    double compensation_factor = 0.9;
    double max_error = max_diff;

    std::vector<size_t> dim_offsets = {(size_t)dims[1] * dims[2], (size_t)dims[2], 1};

    // compensation by slices
    // 1. slicing from the first dimension
    // dim1 slices
    std::vector<Real> compensated_data1(data_size, 0);
    std::copy(dec_data.begin(), dec_data.end(), compensated_data1.begin());
    std::vector<int> slice_dims = {dims[1], dims[2]};
    #pragma omp parallel for num_threads(1)
    for (int i = 0; i < dims[0]; i++) {
        std::vector<Real> slice_dec_data(dims[1] * dims[2], 0);
        std::vector<int> slice_quant_inds(dims[1] * dims[2], 0);
        size_t slice_offset = i * dim_offsets[0];
        Real slice_max_diff = 0;
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                size_t local_index = j * slice_dims[1] + k;
                size_t global_idx = slice_offset + j * dim_offsets[1] + k * dim_offsets[2];
                slice_dec_data[local_index] = dec_data[global_idx];
                slice_quant_inds[local_index] = quant_inds[global_idx];
                slice_max_diff = std::max(slice_max_diff, std::abs(original_data[global_idx] - dec_data[global_idx]));
            }
        }

        // if quant is meaningless, skip this slice
        int slice_quant_max, slice_quant_min;
        slice_quant_max = *std::max_element(slice_quant_inds.begin(), slice_quant_inds.end());
        slice_quant_min = *std::min_element(slice_quant_inds.begin(), slice_quant_inds.end());
        if (slice_quant_max -  slice_quant_min <= 0  ) {
            // printf("max quant = min quant = %d, will not process this slice. \n", slice_quant_max);
            continue;
        } else {
            // printf("slice #%d, max quant = %d, min quant = %d, max diff = %.6f\n", i, slice_quant_max,
                //    slice_quant_min, slice_max_diff);
        auto compensator = PM::Compensation<Real, int>(2, slice_dims.data(), slice_dec_data.data(),
                                                       slice_quant_inds.data(), slice_max_diff * compensation_factor);
            compensator.set_edt_thread_num(2);
            compensator.set_use_rbf(use_rbf);
            compensator.set_frequent_quant_index(most_frequent_quant_index);

            auto compensation_map = compensator.get_compensation_map();
            for (int j = 0; j < dims[1]; j++) {
                for (int k = 0; k < dims[2]; k++) {
                    size_t local_index = j * slice_dims[1] + k;
                    size_t global_idx = slice_offset + j * dim_offsets[1] + k * dim_offsets[2];
                    compensated_data1[global_idx] += compensation_map[local_index];
                }
            }
        }
    }

    // verify the compensated data
    verify(original_data.data(), compensated_data1.data(), data_size, psnr, nrmse, max_diff);
    ssim = PM::calculateSSIM(original_data.data(), compensated_data1.data(), N, dims_.data());
    printf("::DIM1 SSIM = %f\n", ssim);
    // write the compensation map to file
    std::string outfile = argv[N + 5] + std::string(".dir1.f32");
    writefile(outfile.c_str(), compensated_data1.data(), data_size);


    // dim2  slices
    std::vector<Real> compensated_data2(data_size, 0);
    std::copy(dec_data.begin(), dec_data.end(), compensated_data2.begin());
    slice_dims = {dims[0], dims[2]};
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < dims[1]; i++) {
        std::vector<Real> slice_dec_data(dims[0] * dims[2], 0);
        std::vector<int> slice_quant_inds(dims[0] * dims[2], 0);
        size_t slice_offset = i * dim_offsets[1];
        Real slice_max_diff = 0;
        for (int j = 0; j < dims[0]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                size_t local_index = j * slice_dims[1] + k;
                size_t global_idx = slice_offset + j * dim_offsets[0] + k * dim_offsets[2];
                slice_dec_data[local_index] = dec_data[global_idx];
                slice_quant_inds[local_index] = quant_inds[global_idx];
                slice_max_diff = std::max(slice_max_diff, std::abs(original_data[global_idx] - dec_data[global_idx]));
            }
        }
        int slice_quant_max, slice_quant_min;
        slice_quant_max = *std::max_element(slice_quant_inds.begin(), slice_quant_inds.end());
        slice_quant_min = *std::min_element(slice_quant_inds.begin(), slice_quant_inds.end());
        if (slice_quant_max -  slice_quant_min <= 0  ) {
            // printf("max quant = min quant = %d, will not process this slice. \n", slice_quant_max);
            continue;
        }
        auto compensator = PM::Compensation<Real, int>(2, slice_dims.data(), slice_dec_data.data(),
                                                       slice_quant_inds.data(), slice_max_diff * compensation_factor);
        compensator.set_use_rbf(use_rbf);
        compensator.set_frequent_quant_index(most_frequent_quant_index);

        auto compensation_map = compensator.get_compensation_map();

        for (int j = 0; j < dims[0]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                size_t local_index = j * slice_dims[1] + k;
                size_t global_idx = slice_offset + j * dim_offsets[0] + k * dim_offsets[2];
                compensated_data2[global_idx] += compensation_map[local_index];
            }
        }
    }

    // verify the compensated data
    verify(original_data.data(), compensated_data2.data(), data_size, psnr, nrmse, max_diff);
    ssim = PM::calculateSSIM(original_data.data(), compensated_data2.data(), N, dims_.data());
    printf("::DIM2 SSIM = %f\n", ssim);
    // write the compensation map to file
    outfile = argv[N + 5] + std::string(".dir2.f32");
    writefile(outfile.c_str(), compensated_data2.data(), data_size);

    // dim3  slices

    std::vector<Real> compensated_data3(data_size, 0);
    std::copy(dec_data.begin(), dec_data.end(), compensated_data3.begin());
    slice_dims = {dims[0], dims[1]};
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < dims[2]; i++) {
        std::vector<Real> slice_dec_data(dims[0] * dims[1], 0);
        std::vector<int> slice_quant_inds(dims[0] * dims[1], 0);
        size_t slice_offset = i * dim_offsets[2];
        Real slice_max_diff = 0;
        for (int j = 0; j < dims[0]; j++) {
            for (int k = 0; k < dims[1]; k++) {
                size_t local_index = j * slice_dims[1] + k;
                size_t global_idx = slice_offset + j * dim_offsets[0] + k * dim_offsets[1];
                slice_dec_data[local_index] = dec_data[global_idx];
                slice_quant_inds[local_index] = quant_inds[global_idx];
                slice_max_diff = std::max(slice_max_diff, std::abs(original_data[global_idx] - dec_data[global_idx]));
            }
        }
        int slice_quant_max, slice_quant_min;
        slice_quant_max = *std::max_element(slice_quant_inds.begin(), slice_quant_inds.end());
        slice_quant_min = *std::min_element(slice_quant_inds.begin(), slice_quant_inds.end());
        if (slice_quant_max -  slice_quant_min <= 0  ) {
            // printf("max quant = min quant = %d, will not process this slice. \n", slice_quant_max);
            continue;
        }
        auto compensator = PM::Compensation<Real, int>(2, slice_dims.data(), slice_dec_data.data(),
                                                       slice_quant_inds.data(), slice_max_diff * compensation_factor);
        compensator.set_use_rbf(use_rbf);
        compensator.set_frequent_quant_index(most_frequent_quant_index);

        auto compensation_map = compensator.get_compensation_map();
        for (int j = 0; j < dims[0]; j++) {
            for (int k = 0; k < dims[1]; k++) {
                size_t local_index = j * slice_dims[1] + k;
                size_t global_idx = slice_offset + j * dim_offsets[0] + k * dim_offsets[1];
                compensated_data3[global_idx] += compensation_map[local_index];
            }
        }
    }

    // verify the compensated data
    // verify(original_data.data(), compensated_data3.data(), data_size, psnr, nrmse, max_diff);
    // ssim = PM::calculateSSIM(original_data.data(), compensated_data3.data(), N, dims_.data());
    // printf("SSIM = %f\n", ssim);
    // write the compensation map to file
    outfile = argv[N + 5] + std::string(".dir3.f32");
    writefile(outfile.c_str(), compensated_data3.data(), data_size);

    // get the avg of three directions
    std::vector<Real> compensated_data_avg(data_size, 0);
    for (int i = 0; i < data_size; i++) {
        compensated_data_avg[i] = (compensated_data1[i] + compensated_data2[i] + compensated_data3[i]) / 3;
    }
    // verify the compensated data
    verify(original_data.data(), compensated_data_avg.data(), data_size, psnr, nrmse, max_diff);
    ssim = PM::calculateSSIM(original_data.data(), compensated_data_avg.data(), N, dims_.data());
    printf("DIM3 ::SSIM = %f\n", ssim);
    // write the compensation map to file
    outfile = argv[N + 5];
    writefile(outfile.c_str(), compensated_data_avg.data(), data_size);

    return 0;
}
