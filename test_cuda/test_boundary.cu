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
#include "compensation_cuda.hpp"


using namespace std;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;


void runCUDA(const int *quant_inds, char* boundary, size_t input_size,  uint width, uint height, uint depth)
{
    // create device memory
    int* d_quant_inds;
    char* d_boundary; 

    cudaMalloc(&d_quant_inds, input_size * sizeof(int));
    cudaMalloc(&d_boundary, input_size * sizeof(char));
    // copy data to device
    cudaMemcpy(d_quant_inds, quant_inds, input_size * sizeof(int), cudaMemcpyHostToDevice);
    // use quantizer
    dim3 block(8, 8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);
    // __global__ void get_boundary(int* input, char* output, char b_tag, int rank, uint width, uint height, uint depth)

    char b_tag = 1; 

    // int trails = = 10; 
    int num_trials = 20;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i <num_trials; i++){
    get_boundary<<<grid, block>>>(d_quant_inds, d_boundary, b_tag, 3, width, height, depth); 
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    printf("runCUDA executed in %lld microseconds\n", ms / num_trials);    
    // double size = sizeof(T) * width * height;
    printf("Throughput = %f GB/s\n",(double) input_size / (ms / num_trials) * 1e-3); 

    cudaDeviceSynchronize();
    // copy data

    cudaMemcpy(boundary, d_boundary, input_size * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(d_quant_inds);
    cudaFree(d_boundary);

    cudaProfilerStop();
    cudaDeviceReset();

}








int main(int argc, char** argv)
{
    printf("Starting\n");

    size_t input_size = 0; 

    auto quant_inds = readfile<int>(argv[1], input_size); 

    std::cout << "quant max " << *std::max_element(quant_inds.get(), quant_inds.get() + input_size) << std::endl; 
    std::cout << "quant min " << *std::min_element(quant_inds.get(), quant_inds.get() + input_size) << std::endl;

    std::vector<char> boundary(input_size, 0);

    int dims[3] = {256,384,384};

    // size_t input_size = dims[0]*dims[1]*dims[2];


    
    runCUDA( quant_inds.get(), boundary.data(),input_size,  dims[2], dims[1], dims[0]);

    // write quantized data to file

    std::cout << "bound_max " << (int) *std::max_element(boundary.begin(), boundary.end()) << std::endl;
    std::cout << "bound_min " << (int) *std::min_element(boundary.begin(), boundary.end()) << std::endl;
    writefile<char>("boundary.uint8", boundary.data(), input_size);


    return 0;
}
