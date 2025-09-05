#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
// #include "SZ3/api/sz.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include<algorithm>
#include "utils/timer.hpp"
#include "compensation.hpp"
#include <vector>
#include "utils/qcat_ssim.hpp"
#include <omp.h> 
#include "utils/file_utils.hpp" 
#include "utils/stats.hpp"

using Real = float; 
namespace SZ=SZ3; 

namespace fs = std::filesystem;



int main(int argc, char** argv)
{
    int N = atoi(argv[1]);
    std::vector<int> dims(N);
    size_t data_size = 1;
    for (int i = 0; i < N; i++) {
        dims[i] = atoi(argv[i + 2]);
        data_size *= dims[i];
    }

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
    double eb = rel_eb*(max - min);
    printf("relative eb: %.6f\n",rel_eb);
    printf("absolute eb: %.6f\n", eb);
    // create a linear quantizer
    auto quantizer = SZ::LinearQuantizer<float>();
    quantizer.set_eb(eb);
    // iterate the input data and quantize it
    for (size_t i = 0; i < data_size; i++) {
        quant_inds[i] = quantizer.quantize_and_overwrite(dec_data[i],0)-32768;
    }
    // write quantized data 
    writefile(argv[N+4], dec_data.data(), data_size);

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
    double global_max_error = max_diff;


    


    std::vector<size_t> dim_offsets = {(size_t) dims[1]*dims[2], (size_t)dims[2], (size_t)1};

    // compensation by slices 
    // 1. slicing from the first dimension 
    // dim1 slices 
    std::vector<Real> compensated_data1(data_size, 0);
    std::copy(dec_data.begin(), dec_data.end(), compensated_data1.begin());
    std::vector<int> slice_dims = {dims[1], dims[2]};
    int omp_num_threads = 4 ;
    // get the largest two dims 
    std::sort(dims_.begin(), dims_.end(), std::greater<size_t>()); // sort in descending order 
    std::vector<std::vector<Real>> slice_dec_data(omp_num_threads, std::vector<Real>(dims_[1]*dims_[0], 0));
    std::vector<std::vector<int>> slice_quant_inds(omp_num_threads, std::vector<int>(dims_[1]*dims_[0], 0));

    std::vector<std::vector<double> > distance_map_1_1(dims[0]);
    std::vector<std::vector<double> > distance_map_1_2(dims[0]);
    std::vector<std::vector<int>> sign_map_1(dims[0]);

    for (int i = 0; i < N; i++) {
        dims_[i] = dims[i];
    }

    #pragma omp parallel for num_threads(omp_num_threads)
    for(int i = 0; i < dims[0]; i++)
    {
        // std::vector<Real> slice_dec_data(dims[1]*dims[2], 0);
        // std::vector<int> slice_quant_inds(dims[1]*dims[2], 0);
        int thread_id = omp_get_thread_num(); 
        size_t slice_offset = i * dim_offsets[0];
        Real slice_max_diff = 0; 
        for(int j = 0; j < dims[1]; j++)
        {
            for(int k = 0; k < dims[2]; k++)
            {
                size_t local_index = j*slice_dims[1] + k; 
                size_t global_idx = slice_offset + j*dim_offsets[1]+  k*dim_offsets[2]; 
                slice_dec_data[thread_id][local_index] = dec_data[global_idx];
                slice_quant_inds[thread_id][local_index] = quant_inds[global_idx];
                slice_max_diff = std::max(slice_max_diff, std::abs(original_data[global_idx] - 
                                                            dec_data[global_idx]));
            }
        }
        std::cout<<"failed here \n";
        auto compensator = PM::Compensation<Real, int>(2, slice_dims.data(),
                    slice_dec_data[thread_id].data(), slice_quant_inds[thread_id].data(), 
                    slice_max_diff*compensation_factor);

        std::cout<<"failed here \n";
        auto compensation_map = compensator.get_compensation_map_2d(distance_map_1_1[i], 
                                                                distance_map_1_2[i],
                                                                sign_map_1[i]);
        std::cout<<"compensation map size = "<<compensation_map.size()<<std::endl;
                                                                
        for(int j = 0; j < dims[1]; j++)
        {
            for(int k = 0; k < dims[2]; k++)
            {
                size_t local_index = j*slice_dims[1] + k; 
                size_t global_idx = slice_offset + j*dim_offsets[1]+  k*dim_offsets[2]; 
                compensated_data1[global_idx] += compensation_map[local_index];
            }
        }
    }

    // verify the compensated data
    printf("====================================\n");
    printf("Direction 1\n");
    verify(original_data.data(), compensated_data1.data(), data_size, psnr, nrmse, max_diff);
    ssim = PM::calculateSSIM(original_data.data(), compensated_data1.data(), N, dims_.data());
    printf("SSIM = %f\n", ssim);
    // write the compensation map to file
    std::string outfile = argv[N+5] + std::string(".dir1");
    writefile(outfile.c_str(), compensated_data1.data(), data_size);
    printf("====================================\n");





    // dim2  slices 
    std::vector<Real> compensated_data2(data_size, 0);
    std::copy(dec_data.begin(), dec_data.end(), compensated_data2.begin());
    slice_dims = {dims[0], dims[2]};
    std::vector<std::vector<double> > distance_map_2_1(dims[1]);
    std::vector<std::vector<double> > distance_map_2_2(dims[1]);
    std::vector<std::vector<int>> sign_map_2(dims[1]);

    #pragma omp parallel for num_threads(omp_num_threads)
    for(int i = 0; i < dims[1]; i++)
    {
        // std::vector<Real> slice_dec_data(dims[0]*dims[2], 0);
        // std::vector<int> slice_quant_inds(dims[0]*dims[2], 0);
        int thread_id = omp_get_thread_num(); 
        size_t slice_offset = i * dim_offsets[1];
        Real slice_max_diff = 0; 
        for(int j = 0; j < dims[0]; j++)
        {
            for(int k = 0; k < dims[2]; k++)
            {
                size_t local_index = j*slice_dims[1] + k; 
                size_t global_idx = slice_offset + j*dim_offsets[0]+  k*dim_offsets[2]; 
                slice_dec_data[thread_id][local_index] = dec_data[global_idx];
                slice_quant_inds[thread_id][local_index] = quant_inds[global_idx   ];
                slice_max_diff = std::max(slice_max_diff, std::abs(original_data[global_idx] - 
                                                            dec_data[global_idx]));
            }
        }
        auto compensator = PM::Compensation<Real, int>(2, slice_dims.data(),
                    slice_dec_data[thread_id].data(), slice_quant_inds[thread_id].data(), slice_max_diff*compensation_factor);
        auto compensation_map = compensator.get_compensation_map_2d(distance_map_2_1[i], 
                                                                distance_map_2_2[i],
                                                                sign_map_2[i]);
        for(int j = 0; j < dims[0]; j++)
        {
            for(int k = 0; k < dims[2]; k++)
            {
                size_t local_index = j*slice_dims[1] + k; 
                size_t global_idx = slice_offset + j*dim_offsets[0]+  k*dim_offsets[2]; 
                compensated_data2[global_idx] += compensation_map[local_index];
            }
        }

    }

    // verify the compensated data
    printf("====================================\n");
    printf("Direction 2\n");
    verify(original_data.data(), compensated_data2.data(), data_size, psnr, nrmse, max_diff);
    ssim = PM::calculateSSIM(original_data.data(), compensated_data2.data(), N, dims_.data());
    printf("SSIM = %f\n", ssim);
    printf("====================================\n");
    // write the compensation map to file
    outfile = argv[N+5] + std::string(".dir2");
    writefile(outfile.c_str(), compensated_data2.data(), data_size);

    // dim3  slices
    std::vector<std::vector<double> > distance_map_3_1(dims[2]);
    std::vector<std::vector<double> > distance_map_3_2(dims[2]);
    std::vector<std::vector<int>> sign_map_3(dims[2]);

    std::vector<Real> compensated_data3(data_size, 0);
    std::copy(dec_data.begin(), dec_data.end(), compensated_data3.begin());
    slice_dims = {dims[0], dims[1]};
    #pragma omp parallel for num_threads(omp_num_threads)
    for(int i = 0; i < dims[2]; i++)
    {
        // std::vector<Real> slice_dec_data(dims[0]*dims[1], 0);
        // std::vector<int> slice_quant_inds(dims[0]*dims[1], 0);
        size_t slice_offset = i * dim_offsets[2];
        Real slice_max_diff = 0; 
        int thread_id = omp_get_thread_num(); 

        for(int j = 0; j < dims[0]; j++)
        {
            for(int k = 0; k < dims[1]; k++)
            {
                size_t local_index = j*slice_dims[1] + k; 
                size_t global_idx = slice_offset + j*dim_offsets[0]+  k*dim_offsets[1]; 
                slice_dec_data[thread_id][local_index] = dec_data[global_idx];
                slice_quant_inds[thread_id][local_index] = quant_inds[global_idx   ];
                slice_max_diff = std::max(slice_max_diff, std::abs(original_data[global_idx] - 
                                                            dec_data[global_idx]));
            }
        }
        auto compensator = PM::Compensation<Real, int>(2, slice_dims.data(),
                    slice_dec_data[thread_id].data(), slice_quant_inds[thread_id].data(), slice_max_diff*compensation_factor);
        // auto compensation_map = compensator.get_compensation_map();
        auto compensation_map = compensator.get_compensation_map_2d(distance_map_3_1[i], 
                                                                distance_map_3_2[i],
                                                                sign_map_3[i]);
        for(int j = 0; j < dims[0]; j++)
        {
            for(int k = 0; k < dims[1]; k++)
            {
                size_t local_index = j*slice_dims[1] + k; 
                size_t global_idx = slice_offset + j*dim_offsets[0]+  k*dim_offsets[1]; 
                compensated_data3[global_idx] += compensation_map[local_index];
            }
        }
    }

    // verify the compensated data
    printf("====================================\n");
    printf("Direction 3\n");
    verify(original_data.data(), compensated_data3.data(), data_size, psnr, nrmse, max_diff);
    ssim = PM::calculateSSIM(original_data.data(), compensated_data3.data(), N, dims_.data());
    printf("SSIM = %f\n", ssim);
    printf("====================================\n");
    // write the compensation map to file
    outfile = argv[N+5] + std::string(".dir3");
    writefile(outfile.c_str(), compensated_data3.data(), data_size);


    // now use 3d compensation 
    auto compensator = PM::Compensation<Real, int>(N, dims.data(),
                    dec_data.data(), quant_inds.data(),  (global_max_error)*compensation_factor);
    std::vector<int> sign_map;
    auto compensation_map = compensator.get_compensation_map_3d(sign_map);
    std::vector<Real> compensated_data3d(data_size, 0);
    for(int i = 0; i < data_size; i++)
    {
        compensated_data3d[i] = dec_data[i] + compensation_map[i];
    }
    // verify the compensated data
    printf("====================================\n");
    printf("3D Compensation\n");
    verify(original_data.data(), compensated_data3d.data(), data_size, psnr, nrmse, max_diff);
    ssim = PM::calculateSSIM(original_data.data(), compensated_data3d.data(), N, dims_.data());
    printf("SSIM = %f\n", ssim);
    // write the compensation map to file
    outfile = argv[N+5] + std::string(".dir3d");
    writefile(outfile.c_str(), compensated_data3d.data(), data_size);


    // use 6 distances to compensate 
    printf("====================================\n");
    std::vector<Real> compensated_data_6(data_size, 0);
    for (int i = 0; i < data_size; i++)
    {
        int x = i / (dims[1] * dims[2]); //0
        int y = (i / dims[2]) % dims[1]; //1
        int z = i % dims[2];             //2
        slice_dims = {dims[1], dims[2]};
        int idx = y * dims[1] + z;
        double d_1_1 = distance_map_1_1[x][idx]+0.5;
        double d_1_2 = distance_map_1_2[x][idx]+0.5;
        int sign_1 = sign_map_1[x][idx];
        slice_dims = {dims[0], dims[2]};
        idx = x * dims[1] + z;
        double d_2_1 = distance_map_2_1[y][idx]+0.5;
        double d_2_2 = distance_map_2_2[y][idx]+0.5;
        int sign_2 = sign_map_2[y][idx];
        slice_dims = {dims[0], dims[1]};
        idx = x * dims[1] + y;
        double d_3_1 = distance_map_3_1[z][idx]+0.5;
        double d_3_2 = distance_map_3_2[z][idx]+0.5;
        int sign_3 = sign_map_3[z][idx];

        // double magnitude = (1/d_1_1*sign_1 + 1/d_2_1*sign_2 + 1/d_3_1*sign_3) /( 1/d_1_1 + 1/d_2_1 + 1/d_3_1 + 
        //                                                     1/d_1_2 + 1/d_2_2 + 1/d_3_2) * 
        //                                                     global_max_error * compensation_factor;
        double magnitude = (1/(d_1_1*d_1_1)*sign_1 + 1/(d_2_1*d_2_1)*sign_2 + 1/(d_3_1*d_3_1)*sign_3 ) /
                                    (1/(d_1_1*d_1_1) + 1/(d_2_1*d_2_1) + 1/(d_3_1*d_3_1) + 
                                     1/(d_1_2*d_1_2) + 1/(d_2_2*d_2_2) + 1/(d_3_2*d_3_2)) * 
                                                            global_max_error * compensation_factor;
                                            

        // compensated_data_6[i] = dec_data[i] + magnitude * sign_map[i];
        if(  y == 81 && z == 300)
        {
            if(x <30 && x >20){
                printf("====================================\n");
                printf("x: %d, y: %d, z: %d\n", x, y, z);
                printf("d_1_1: %f, d_1_2: %f, sign_1: %d\n", d_1_1, d_1_2, sign_1);
                printf("d_2_1: %f, d_2_2: %f, sign_2: %d\n", d_2_1, d_2_2, sign_2);
                printf("d_3_1: %f, d_3_2: %f, sign_3: %d\n", d_3_1, d_3_2, sign_3);
                printf("magnitude: %f\n", magnitude);
                printf("====================================\n");
            }

        }
        compensated_data_6[i] = dec_data[i] + magnitude;

    }
    // verify the compensated data
    printf("====================================\n");
    printf("6 distances Compensation\n");
    verify(original_data.data(), compensated_data_6.data(), data_size, psnr, nrmse, max_diff);
    auto timer = Timer(); 
    timer.start();
    ssim = PM::calculateSSIM(original_data.data(), compensated_data_6.data(), N, dims_.data());
    std::cout << "ssim_timing= " << timer.stop() << std::endl; 
    printf("SSIM = %f\n", ssim);
    // write the compensation map to file
    printf("====================================\n");

    outfile = argv[N+5]+std::string("");
    writefile(outfile.c_str(), compensated_data_6.data(), data_size);



    // get the avg of three directions
    std::vector<Real> compensated_data_avg(data_size, 0);
    for(int i = 0; i < data_size; i++)
    {
        compensated_data_avg[i] = (compensated_data1[i] + compensated_data2[i] + compensated_data3[i])/3;
    }
    // verify the compensated data
    printf("====================================\n");
    printf("Average of 3 directions\n");
    verify(original_data.data(), compensated_data_avg.data(), data_size, psnr, nrmse, max_diff);
    ssim = PM::calculateSSIM(original_data.data(), compensated_data_avg.data(), N, dims_.data());
    printf("SSIM = %f\n", ssim);
    // write the compensation map to file
    outfile = argv[N+5] +std::string(".avg_dir");
    writefile(outfile.c_str(), compensated_data_avg.data(), data_size);
    printf("====================================\n");

    return 0;
}
