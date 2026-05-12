#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <iostream>
#include "utils/file_utils.hpp"
#include "compensation_cuda.hpp"
#include "boundary_cuda.hpp"
#include "edt.hpp"
#include "edt_jfa.hpp"
#include "edt_pba.hpp"

// Returns {sparsity, edge_density}. Both are in [0,1].
// sparsity = fraction of non-mode voxels; edge_density = fraction of boundary voxels.
static std::pair<double,double> compute_field_stats(
    const int* quant_inds, size_t N,
    uint width, uint height, uint depth)
{
    // --- sparsity: find mode quant index ---
    std::unordered_map<int,size_t> freq;
    freq.reserve(1 << 16);
    for (size_t i = 0; i < N; i++) freq[quant_inds[i]]++;
    int mode_q = 0; size_t mode_count = 0;
    for (auto& kv : freq) if (kv.second > mode_count) { mode_count = kv.second; mode_q = kv.first; }
    double sparsity = 1.0 - (double)mode_count / N;

    // --- edge density: voxels with at least one face-neighbor of different quant index ---
    size_t n_edge = 0;
    const int W = (int)width, H = (int)height, D = (int)depth;
    for (int z = 0; z < D; z++)
    for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) {
        int q = quant_inds[(size_t)z*H*W + y*W + x];
        bool is_edge = false;
        if (x>0   && quant_inds[(size_t)z*H*W + y*W + (x-1)] != q) is_edge = true;
        if (x<W-1 && quant_inds[(size_t)z*H*W + y*W + (x+1)] != q) is_edge = true;
        if (y>0   && quant_inds[(size_t)z*H*W + (y-1)*W + x] != q) is_edge = true;
        if (y<H-1 && quant_inds[(size_t)z*H*W + (y+1)*W + x] != q) is_edge = true;
        if (z>0   && quant_inds[(size_t)(z-1)*H*W + y*W + x] != q) is_edge = true;
        if (z<D-1 && quant_inds[(size_t)(z+1)*H*W + y*W + x] != q) is_edge = true;
        if (is_edge) n_edge++;
    }
    double edge_density = (double)n_edge / N;
    return {sparsity, edge_density};
}

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
    double sse = sum; // sum of square error
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

// edt_method: 0=chunk, 1=JFA, 2=PBA+, 3=PBA+ optimized (fused kernels + downsampled R2)
void run_cuda(
    int* quant_inds, float* quantized_data, size_t size, uint width, uint height, uint depth,
    double magnitude, bool use_chunck = true, int edt_method = 0, int jfa_level = 0)
{
    // allocate memory on the device
    int* d_quant_inds;
    float* d_quantized_data;
    char* d_boundary;
    char* d_boundary_neutral;
    float* distance_edge;
    int* index_edge;
    float* distance_neutral;
    int* index_neutral;
    char* d_sign_map;

    auto check_cuda = [](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            printf("CUDA ERROR at %s: %s\n", msg, cudaGetErrorString(err));
        }
    };

    check_cuda(cudaMalloc(&d_quant_inds, size*sizeof(int)), "malloc quant_inds");
    check_cuda(cudaMalloc(&d_quantized_data, size*sizeof(float)), "malloc quantized_data");
    check_cuda(cudaMalloc(&d_boundary, size*sizeof(char)), "malloc boundary");
    check_cuda(cudaMalloc(&d_boundary_neutral, size*sizeof(char)), "malloc boundary_neutral");
    // method=4 uses only coarse buffers allocated inside its own block — skip fine-grid allocs
    if (edt_method != 4) {
        check_cuda(cudaMalloc(&distance_edge,   size*sizeof(float)),     "malloc distance_edge");
        check_cuda(cudaMalloc(&index_edge,      size*sizeof(int)*3),     "malloc index_edge");
        check_cuda(cudaMalloc(&distance_neutral,size*sizeof(float)),     "malloc distance_neutral");
        check_cuda(cudaMalloc(&index_neutral,   size*sizeof(int)*3),     "malloc index_neutral");
    }
    check_cuda(cudaMalloc(&d_sign_map, size*sizeof(char)), "malloc sign_map");

    // copy the data to the device
    check_cuda(cudaMemcpy(d_quant_inds, quant_inds, size*sizeof(int), cudaMemcpyHostToDevice), "memcpy quant_inds");
    check_cuda(cudaMemcpy(d_quantized_data, quantized_data, size*sizeof(float), cudaMemcpyHostToDevice), "memcpy quantized_data");

    auto start = std::chrono::high_resolution_clock::now();
    double stage_boundary = 0.0;
    double stage_edt1 = 0.0;
    double stage_fill_sign = 0.0;
    double stage_neutral_boundary = 0.0;
    double stage_edt2 = 0.0;
    double stage_comp = 0.0;
    auto stage_start = std::chrono::high_resolution_clock::now();

    // Pre-allocate PBA+ ping-pong buffers (reused for both EDT rounds).
    // Method 4 allocates its own coarse-sized buffers inside its block.
    int* pba_buf0 = nullptr;
    int* pba_buf1 = nullptr;
    if (edt_method == 2 || edt_method == 3) {
        size_t pba_sz = pba_buffer_size(width, height, depth);
        check_cuda(cudaMalloc(&pba_buf0, pba_sz), "malloc pba_buf0");
        check_cuda(cudaMalloc(&pba_buf1, pba_sz), "malloc pba_buf1");
    }

    // === Method 4: both EDT rounds downsampled to 256³ ===
    // Eliminates fine-grid PBA+ (50ms) and 2 GB of fine-grid index/distance buffers.
    if (edt_method == 4) {
        int ds_w, ds_h, ds_d;
        pba_downsample_dims(width, height, depth, ds_w, ds_h, ds_d);
        size_t ds_size = (size_t)ds_w * ds_h * ds_d;

        size_t pba_sz_coarse = pba_buffer_size((uint)ds_w, (uint)ds_h, (uint)ds_d);
        check_cuda(cudaMalloc(&pba_buf0, pba_sz_coarse), "malloc pba_buf0_coarse");
        check_cuda(cudaMalloc(&pba_buf1, pba_sz_coarse), "malloc pba_buf1_coarse");

        char*         d_coarse_boundary = nullptr;
        char*         d_coarse_sign     = nullptr;
        float*        d_coarse_d1       = nullptr;
        unsigned int* d_coarse_packed   = nullptr;
        check_cuda(cudaMalloc(&d_coarse_boundary, ds_size),                       "coarse_boundary");
        check_cuda(cudaMalloc(&d_coarse_sign,     ds_size),                       "coarse_sign");
        check_cuda(cudaMalloc(&d_coarse_d1,       ds_size * sizeof(float)),       "coarse_d1");
        check_cuda(cudaMalloc(&d_coarse_packed,   ds_size * sizeof(unsigned int)),"coarse_packed");

        // Step 1: fine boundary detection + sign map (boundary voxels only)
        stage_start = std::chrono::high_resolution_clock::now();
        {
            dim3 block(8,8,8);
            dim3 grid((width+7)/8,(height+7)/8,(depth+7)/8);
            get_boundary_and_sign_map<int><<<grid,block>>>(
                d_quant_inds, d_boundary, d_sign_map, (char)1, 3, width, height, depth);
            cudaDeviceSynchronize();
        }
        stage_boundary = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stage_start).count();

        // Step 2+3: downsample boundary+sign then coarse PBA+ (EDT round 1)
        stage_start = std::chrono::high_resolution_clock::now();
        {
            dim3 block(8,8,8);
            dim3 grid((ds_w+7)/8,(ds_h+7)/8,(ds_d+7)/8);
            downsample_boundary_with_sign_2x<<<grid,block>>>(
                d_boundary, d_sign_map, d_coarse_boundary, d_coarse_sign,
                (int)width, (int)height, (int)depth, ds_w, ds_h, ds_d);
            cudaDeviceSynchronize();
        }
        edt_3d_pba_downsampled(d_coarse_boundary, d_coarse_d1,
                               (uint)ds_w, (uint)ds_h, (uint)ds_d,
                               pba_buf0, pba_buf1, d_coarse_packed);
        cudaDeviceSynchronize();
        stage_edt1 = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stage_start).count();

        // Step 4+5: fill sign on coarse grid, upsample to fine
        stage_start = std::chrono::high_resolution_clock::now();
        {
            dim3 block(8,8,8);
            dim3 grid((ds_w+7)/8,(ds_h+7)/8,(ds_d+7)/8);
            fill_sign_coarse<<<grid,block>>>(d_coarse_sign, d_coarse_packed, ds_w, ds_h, ds_d);
            cudaDeviceSynchronize();
        }
        cudaFree(d_coarse_packed);
        cudaFree(d_coarse_boundary);
        {
            dim3 block(8,8,8);
            dim3 grid((width+7)/8,(height+7)/8,(depth+7)/8);
            upsample_sign_2x<<<grid,block>>>(
                d_coarse_sign, d_sign_map,
                (int)width, (int)height, (int)depth, ds_w, ds_h, ds_d);
            cudaDeviceSynchronize();
        }
        stage_fill_sign = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stage_start).count();

        // Step 6: neutral boundary on fine sign map
        stage_start = std::chrono::high_resolution_clock::now();
        {
            dim3 block(8,8,8);
            dim3 grid((width+7)/8,(height+7)/8,(depth+7)/8);
            get_filtered_boundary<char,char><<<grid,block>>>(
                d_sign_map, d_boundary, d_boundary_neutral, (char)1, width, height, depth);
            cudaDeviceSynchronize();
        }
        stage_neutral_boundary = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stage_start).count();

        // Step 7: EDT round 2 downsampled (same as method=3)
        stage_start = std::chrono::high_resolution_clock::now();
        float* d_coarse_d2 = nullptr;
        check_cuda(cudaMalloc(&d_coarse_d2, ds_size * sizeof(float)), "coarse_d2");
        edt_3d_pba_downsampled(d_boundary_neutral, d_coarse_d2,
                               width, height, depth, pba_buf0, pba_buf1);
        cudaDeviceSynchronize();
        stage_edt2 = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stage_start).count();

        // Step 8: compensation — trilinear interp for both d1 and d2
        stage_start = std::chrono::high_resolution_clock::now();
        {
            dim3 block(8,8,8);
            dim3 grid((width+7)/8,(height+7)/8,(depth+7)/8);
            compensation_idw_both_downsample<<<grid,block>>>(
                d_coarse_d1, d_coarse_d2, ds_w, ds_h, ds_d,
                d_sign_map, d_quantized_data, (float)magnitude,
                (int)width, (int)height, (int)depth);
            cudaDeviceSynchronize();
        }
        stage_comp = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stage_start).count();

        cudaFree(d_coarse_d1); cudaFree(d_coarse_d2); cudaFree(d_coarse_sign);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        printf("Elapsed time: %f\n", diff.count());
        printf("StageTime boundary_detect: %f\n", stage_boundary);
        printf("StageTime edt_round1: %f\n", stage_edt1);
        printf("StageTime fill_sign: %f\n", stage_fill_sign);
        printf("StageTime neutral_boundary: %f\n", stage_neutral_boundary);
        printf("StageTime edt_round2: %f\n", stage_edt2);
        printf("StageTime compensation: %f\n", stage_comp);
        printf("StageTime edt_total: %f\n", stage_edt1 + stage_edt2);

        cudaFree(pba_buf0); cudaFree(pba_buf1);
        cudaMemcpy(quantized_data, d_quantized_data, size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_quant_inds); cudaFree(d_quantized_data);
        cudaFree(d_boundary); cudaFree(d_boundary_neutral);
        cudaFree(d_sign_map);
        return;
    }

    // === Optimized path (edt_method == 3) ===
    if (edt_method == 3) {
        // --- EDT Round 1: fused boundary + PBA (same as edt_method 2) ---
        stage_start = std::chrono::high_resolution_clock::now();
        edt_3d_pba_fused_boundary(d_quant_inds, d_boundary, d_sign_map,
                                   index_edge, distance_edge,
                                   width, height, depth, pba_buf0, pba_buf1);
        cudaDeviceSynchronize();
        stage_edt1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

        // --- Fill sign (propagate to non-boundary voxels) ---
        stage_start = std::chrono::high_resolution_clock::now();
        {
            dim3 block(8, 8, 8);
            dim3 grid((width + 7) / 8, (height + 7) / 8, (depth + 7) / 8);
            fill_sign_and_neutral_boundary_fused<<<grid, block>>>(
                d_sign_map, d_boundary, d_boundary_neutral,
                (const unsigned int*)index_edge, (char)1,
                width, height, depth);
            cudaDeviceSynchronize();
        }
        stage_fill_sign = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

        // --- Neutral boundary detection (must run after sign propagation completes) ---
        stage_start = std::chrono::high_resolution_clock::now();
        {
            dim3 block(8, 8, 8);
            dim3 grid((width + 7) / 8, (height + 7) / 8, (depth + 7) / 8);
            get_filtered_boundary<char, char><<<grid, block>>>(
                d_sign_map, d_boundary, d_boundary_neutral, (char)1, width, height, depth);
            cudaDeviceSynchronize();
        }
        stage_neutral_boundary = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

        // --- EDT Round 2: downsampled PBA ---
        stage_start = std::chrono::high_resolution_clock::now();
        int ds_w, ds_h, ds_d;
        pba_downsample_dims(width, height, depth, ds_w, ds_h, ds_d);
        size_t ds_size = (size_t)ds_w * ds_h * ds_d;
        float* d_ds_distance = nullptr;
        check_cuda(cudaMalloc(&d_ds_distance, ds_size * sizeof(float)), "malloc ds_distance");
        edt_3d_pba_downsampled(d_boundary_neutral, d_ds_distance,
                               width, height, depth, pba_buf0, pba_buf1);
        cudaDeviceSynchronize();
        stage_edt2 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

        // --- Compensation: on-the-fly d_edge + trilinear-interpolated downsampled d_neutral ---
        stage_start = std::chrono::high_resolution_clock::now();
        {
            dim3 block(8, 8, 8);
            dim3 grid((width + 7) / 8, (height + 7) / 8, (depth + 7) / 8);

            compensation_idw_downsample<<<grid, block>>>(
                d_boundary, (const unsigned int*)index_edge,
                d_ds_distance, ds_w, ds_h, ds_d,
                d_sign_map, d_quantized_data, (float)magnitude,
                width, height, depth);
            cudaDeviceSynchronize();
        }
        stage_comp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

        cudaFree(d_ds_distance);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        printf("Elapsed time: %f\n", diff.count());
        printf("StageTime boundary_detect: %f\n", stage_boundary);
        printf("StageTime edt_round1: %f\n", stage_edt1);
        printf("StageTime fill_sign: %f\n", stage_fill_sign);
        printf("StageTime neutral_boundary: %f\n", stage_neutral_boundary);
        printf("StageTime edt_round2: %f\n", stage_edt2);
        printf("StageTime compensation: %f\n", stage_comp);
        printf("StageTime edt_total: %f\n", stage_edt1 + stage_edt2);

        // Cleanup
        if (pba_buf0) cudaFree(pba_buf0);
        if (pba_buf1) cudaFree(pba_buf1);
        cudaMemcpy(quantized_data, d_quantized_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_quant_inds); cudaFree(d_quantized_data);
        cudaFree(d_boundary); cudaFree(d_boundary_neutral);
        cudaFree(distance_edge); cudaFree(index_edge);
        cudaFree(distance_neutral); cudaFree(index_neutral);
        cudaFree(d_sign_map);
        return;
    }

    stage_start = std::chrono::high_resolution_clock::now();
    // boundary detect (skipped for PBA+ — fused into EDT round 1)
    if (edt_method != 2)
    {
        char b_tag = 1;
        dim3 block(8, 8, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);
        printf("DEBUG: grid=(%u,%u,%u) block=(%u,%u,%u)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
        get_boundary_and_sign_map<int><<<grid, block>>>(d_quant_inds, d_boundary, d_sign_map, b_tag, 3, width, height, depth);
        check_cuda(cudaGetLastError(), "kernel launch get_boundary_and_sign_map");
        check_cuda(cudaDeviceSynchronize(), "sync after get_boundary_and_sign_map");
        // use sign map to create neutral boundary 
        // fill boundary and compensation map 
        // dump the boundary to the host 
        if(0)
        {
            std::vector<char> boundary(size, 0);
            cudaMemcpy(boundary.data(), d_boundary, size*sizeof(char), cudaMemcpyDeviceToHost);
            int bcount = 0;
            for (size_t i = 0; i < size; i++) if (boundary[i] != 0) bcount++;
            printf("DEBUG: boundary points = %d / %zu\n", bcount, size);
        }
        if(0)
        {
            std::vector<char> sign_map(size, 0);
            cudaMemcpy(sign_map.data(), d_sign_map, size*sizeof(char), cudaMemcpyDeviceToHost);
            int pos = 0, neg = 0, zero = 0;
            for (size_t i = 0; i < size; i++) {
                if (sign_map[i] > 0) pos++;
                else if (sign_map[i] < 0) neg++;
                else zero++;
            }
            printf("DEBUG: sign_map pos=%d neg=%d zero=%d\n", pos, neg, zero);
        }
    }
    stage_boundary = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

    stage_start = std::chrono::high_resolution_clock::now();
    // edt core 1
    if(1){
        if(edt_method == 2) {
            // Fused: boundary detect + sign map + PBA EDT in one call
            edt_3d_pba_fused_boundary(d_quant_inds, d_boundary, d_sign_map,
                                       index_edge, distance_edge,
                                       width, height, depth, pba_buf0, pba_buf1);
        }
        else if(edt_method == 1) edt_3d_jfa_level(d_boundary, index_edge, distance_edge, width, height, depth, jfa_level);
        else if(!use_chunck) edt_3d(d_boundary, index_edge, distance_edge, width, height, depth);
        else edt_3d_chunck(d_boundary, index_edge, distance_edge, width, height, depth);

        if(0)
        {
            std::vector<float> distance_host(size, 0);  
            cudaMemcpy(distance_host.data(), distance_edge, size*sizeof(float), cudaMemcpyDeviceToHost);
            writefile("distance_edge.dat", distance_host.data(), size);

            std::vector<int> index_host(size*3, 0);
            cudaMemcpy(index_host.data(), index_edge, size*sizeof(int)*3, cudaMemcpyDeviceToHost);
            writefile("index_edge.dat", index_host.data(), size*3);
        }
    }
    if (edt_method == 2) cudaDeviceSynchronize();  // PBA runs fully async
    stage_edt1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

    stage_start = std::chrono::high_resolution_clock::now();

    if(1){
        // fill sign and compensation
        dim3 block(8, 8, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);
        if (edt_method == 1 || edt_method == 2)
            fill_sign_and_compensation_packed<<<grid, block>>>(
                d_sign_map, d_boundary, (const unsigned int*)index_edge,
                magnitude, width, height, depth);
        else
            fill_sign_and_compensation<<<grid, block>>>(
                d_sign_map, d_boundary, index_edge, magnitude, width, height, depth);
        if (edt_method != 2) cudaDeviceSynchronize();  // PBA: skip sync, same stream ordering
        if(0)
        {
            std::vector<char> sign_map(size, 0);
            cudaMemcpy(sign_map.data(), d_sign_map, size*sizeof(char), cudaMemcpyDeviceToHost);
            writefile("sign_map.uint8", sign_map.data(), size);
        }
    }
    stage_fill_sign = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

    stage_start = std::chrono::high_resolution_clock::now();

    if(1){
        char b_tag = 1;
        dim3 block(8, 8, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);
        if (edt_method == 2) {
            // Fused: get_boundary + filter_boundary in one kernel (no intermediate sync)
            get_filtered_boundary<char, char><<<grid, block>>>(
                d_sign_map, d_boundary, d_boundary_neutral, b_tag, width, height, depth);
        } else {
            get_boundary<char><<<grid, block>>>(d_sign_map, d_boundary_neutral, b_tag, 3, width, height, depth);
            cudaDeviceSynchronize();
            if(1)filter_boundary<<<grid, block>>>(d_boundary, d_boundary_neutral, b_tag, width, height, depth);
            cudaDeviceSynchronize();
        }
        if(0)
        {
            std::vector<char> boundary(size, 0);
            cudaMemcpy(boundary.data(), d_boundary_neutral, size*sizeof(char), cudaMemcpyDeviceToHost);
            writefile("boundary_neutral.uint8", boundary.data(), size);
        }
    }
    stage_neutral_boundary = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

    stage_start = std::chrono::high_resolution_clock::now();

    // second round of edts
    if(1){
        if(edt_method == 2) edt_3d_pba(d_boundary_neutral, index_neutral, distance_neutral, width, height, depth, pba_buf0, pba_buf1);
        else if(edt_method == 1) edt_3d_jfa_level(d_boundary_neutral, index_neutral, distance_neutral, width, height, depth, jfa_level);
        else if(!use_chunck) edt_3d(d_boundary_neutral, index_neutral, distance_neutral, width, height, depth);
        else edt_3d_chunck(d_boundary_neutral, index_neutral, distance_neutral, width, height, depth);
    }
    if (edt_method == 2) cudaDeviceSynchronize();  // PBA runs fully async
    stage_edt2 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

    stage_start = std::chrono::high_resolution_clock::now();
    // compensation
    if(1){
        dim3 block(8, 8, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);
        // template <typename T_distance, typename T_data, typename T_index> 
        // __global__ void compensation_idw(char* boundary, T_distance* d_edge, T_index* idx_edge,  T_distance* d_neutral, 
        //                                 char* sign_map, T_data* quantized_data, T_data magnitude, uint width, uint height, uint depth) 
        compensation_idw<float, float, int><<<grid, block>>>(d_boundary, distance_edge, index_edge, distance_neutral,
                                         d_sign_map, d_quantized_data, magnitude, width, height, depth);
        cudaDeviceSynchronize();
    }
    stage_comp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stage_start).count();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    printf("Elapsed time: %f\n", diff.count());
    printf("StageTime boundary_detect: %f\n", stage_boundary);
    printf("StageTime edt_round1: %f\n", stage_edt1);
    printf("StageTime fill_sign: %f\n", stage_fill_sign);
    printf("StageTime neutral_boundary: %f\n", stage_neutral_boundary);
    printf("StageTime edt_round2: %f\n", stage_edt2);
    printf("StageTime compensation: %f\n", stage_comp);
    printf("StageTime edt_total: %f\n", stage_edt1 + stage_edt2);

    // Sync to ensure all async PBA kernels finish before timing
    cudaDeviceSynchronize();

    // Free PBA buffers
    if (pba_buf0) cudaFree(pba_buf0);
    if (pba_buf1) cudaFree(pba_buf1);

    // copy the data back to the host
    // cudaMemcpy(quant_inds, d_quant_inds, size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(quantized_data, d_quantized_data, size*sizeof(float), cudaMemcpyDeviceToHost);
    // free the memory
    cudaFree(d_quant_inds);
    cudaFree(d_quantized_data);
    cudaFree(d_boundary);
    cudaFree(d_boundary_neutral);
    cudaFree(distance_edge);
    cudaFree(index_edge);
    cudaFree(distance_neutral);
    cudaFree(index_neutral);
    cudaFree(d_sign_map);
}




template <typename T>
inline int quantization_(T data, double abs_eb)
{
    double recipPrecision = 1.0/(2.0*abs_eb);
    double dataRecip = data*recipPrecision;
    int s = dataRecip>=-0.5?0:1;
    return (int)(dataRecip+0.5) - s;
}

int main(int argc, char** argv)
{
    if (argc < 7) {
        printf("Usage: %s <input_file> <rel_eb> <use_chunk> <dim0_fast> <dim1> <dim2_slow> [edt_method] [jfa_level]\n", argv[0]);
        printf("Example: %s Uf48.bin.f32 0.01 1 500 500 100 3\n", argv[0]);
        printf("edt_method: 0=chunk(default), 1=JFA, 2=PBA+, 3=PBA+ optimized, 4=both-downsampled\n");
        return 0;
    }
    std::filesystem::path p{argv[1]} ;
    if (!std::filesystem::exists(p)) {
        printf("File %s does not exist\n", argv[1]);
        return 0;
    }
    size_t file_size = std::filesystem::file_size(p)/sizeof(float);
    uint width = atoi(argv[4]);   // fastest dimension
    uint height = atoi(argv[5]);
    uint depth = atoi(argv[6]);   // slowest dimension
    printf("dimensions: width=%u height=%u depth=%u (total=%zu)\n", width, height, depth, (size_t)width*height*depth);
    if ((size_t)width * height * depth != file_size) {
        printf("ERROR: dimensions don't match file size (%zu)\n", file_size);
        return 1;
    }

    std::vector<int> quant_inds(file_size, 0);
    std::vector<float> input_data(file_size, 0);
    readfile(argv[1],  file_size, input_data.data());
    std::vector<float> input_copy(file_size, 0);
    std::copy(input_data.begin(), input_data.end(), input_copy.begin());
    float max = *std::max_element(input_data.begin(), input_data.end());
    float min = *std::min_element(input_data.begin(), input_data.end());
    printf("max: %f, min: %f\n", max, min);
    double eb = atof(argv[2])*(max - min);
    printf("relative eb: %.6f\n", atof(argv[2]));
    printf("absolute eb: %.6f\n", eb);
    bool use_chunck = bool(atoi(argv[3]));
    int edt_method = (argc > 7) ? atoi(argv[7]) : 0;
    int jfa_level = (argc > 8) ? atoi(argv[8]) : 0;
    if (jfa_level < 0) jfa_level = 0;
    if (jfa_level > 2) jfa_level = 2;
    const char* method_names[] = {"chunk", "JFA", "PBA+", "PBA+ optimized"};
    std::cout << "use chunck edt: " << use_chunck
              << ", EDT method: " << method_names[edt_method < 0 ? 0 : (edt_method > 3 ? 3 : edt_method)]
              << ", JFA level: " << jfa_level << std::endl;

    // quantize using same method as CPU code
    double compensation_factor = 0.9;
    for (size_t i = 0; i < file_size; i++) {
        quant_inds[i] = quantization_(input_data[i], eb);
        input_data[i] = static_cast<float>(2.0 * eb * quant_inds[i]);
        if (std::abs(input_data[i] - input_copy[i]) > eb) {
            input_data[i] = input_copy[i];
            quant_inds[i] = 0;
        }
    }

    double psnr, nrmse, max_diff;
    verify(input_copy.data(), input_data.data(), file_size, psnr, nrmse, max_diff);
    printf("max quantization error: %.6E\n", max_diff);

    // --- Reliability guards (mirrors CPU compensation.hpp defaults) ---
    const double sparsity_threshold    = 0.10;
    const double edge_density_threshold = 0.001;
    auto [sparsity, edge_density] = compute_field_stats(
        quant_inds.data(), file_size, width, height, depth);
    printf("Sparsity %.6f\n", sparsity);
    printf("EdgeDensity %.6f\n", edge_density);
    if (sparsity < sparsity_threshold) {
        printf("Sparsity %.6f < %.3f, skipping compensation\n", sparsity, sparsity_threshold);
        goto done;
    }
    if (edge_density < edge_density_threshold) {
        printf("EdgeDensity %.6f < %.4f, skipping compensation\n", edge_density, edge_density_threshold);
        goto done;
    }

    run_cuda(quant_inds.data(), input_data.data(), file_size, width, height, depth,
             max_diff * compensation_factor, use_chunck, edt_method, jfa_level);
    done:

    double psnr2, nrmse2, max_diff2;
    verify(input_copy.data(), input_data.data(), file_size, psnr2, nrmse2, max_diff2);

    // write output
    std::string out_data_file = p.filename().string() + ".cuda_compensated.f32";
    writefile(out_data_file.c_str(), input_data.data(), file_size);
    printf("Wrote compensated data to %s\n", out_data_file.c_str());

    return 0;
}
