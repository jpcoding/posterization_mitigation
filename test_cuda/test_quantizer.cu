#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string.h>
#include <iostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_profiler_api.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cmath>
#include "edt.hpp"
#include <algorithm>
#include "utils/file_utils.hpp"
#include "quantizer.hpp"


using namespace std;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;


template <typename T>
void runCUDA(T*orig_data, int* quant_inds, size_t input_size, double abs_eb)
{
    // create device memory
    T* d_orig_data;
    int* d_quant_inds;

    cudaMalloc(&d_orig_data, input_size * sizeof(T));
    cudaMalloc(&d_quant_inds, input_size * sizeof(int));
    // copy data to device
    cudaMemcpy(d_orig_data, orig_data, input_size * sizeof(T), cudaMemcpyHostToDevice);
    // use quantizer
    dim3 block3(1024);
    dim3 grid3((input_size + block3.x - 1) / block3.x);
    // quantize_and_overwrite<T><<<grid3, block3>>>(d_orig_data, 0, d_quant_inds, input_size, abs_eb);
    cudaDeviceSynchronize();
    // copy data
    cudaMemcpy(quant_inds, d_quant_inds, input_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(orig_data, d_orig_data, input_size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_orig_data);
    cudaFree(d_quant_inds);
    cudaProfilerStop();
    cudaDeviceReset();

}








int main(int argc, char** argv)
{
    printf("Starting\n");

    size_t input_size = 0; 

    auto input_data = readfile<float>(argv[1], input_size); 
    double rel_eb = atof(argv[2]); 

    double max_val =*std::max_element(input_data.get(), input_data.get()+input_size);
    double min_val =*std::min_element(input_data.get(), input_data.get()+input_size);
    double range = max_val - min_val; 
    double abs_eb = rel_eb * range; 

    std::vector<int> quant_inds(input_size,0); 

    runCUDA(input_data.get(), quant_inds.data(), input_size, abs_eb);

    // write quantized data to file

    writefile(argv[3], quant_inds.data(), input_size);
    writefile(argv[4], input_data.get(), input_size);

    return 0;
}
