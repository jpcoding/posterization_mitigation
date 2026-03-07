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


using namespace std;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;




void runCUDA(char* boundData, int* outputData, float* distanceData, uint width, uint height, uint depth)
{
        // create device memory
        char* d_input;
        int* d_output;
        float* d_distance; 
        int rank = 3; 
        size_t size = width * height * depth; 
        size_t width_stride = 3; 
        size_t height_stride = width * 3;
        size_t depth_stride = width * height * 3; 
    

        cudaMalloc(&d_input, width * height * depth * sizeof(char));
        cudaMalloc(&d_output, width * height * depth * 3 * sizeof(int));
        cudaMalloc(&d_distance, width * height * depth * sizeof(float)); 
        // copy data to device
        cudaMemcpy(d_input, boundData, width * height * depth * sizeof(char), cudaMemcpyHostToDevice);
        // init the output data
    
        dim3 block(8, 8, 8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);

        // dim3 block1(2,2, 1); 
        // dim3 grid1((height+block1.x - 1)/block1.x, (width+ block1.y - 1) / block1.y, (1 + block1.z - 1) / block1.z);  

        dim3 block1(16,16); 
        dim3 grid1((height + block1.x - 1) / block1.x, (width + block1.y - 1) / block1.y);  


        // dim3 block1(1,1); 
        // dim3 grid1((height), (width));  

        // dim3 block3(1,1);
        // dim3 grid3(width, depth);

        dim3 block3(16,16);
        dim3 grid3((width+block3.x-1)/block3.x  , (depth + block3.y - 1) / block3.y );

        // dim3 block2(1,1);
        // dim3 grid2(depth, height);

        dim3 block2(16,16);
        dim3 grid2((depth +block2.x-1)/block2.x, (height+ block2.y - 1) / block2.y);

        int num_trials = 3;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i <num_trials; i++){
            init_edt_3d<<<grid, block>>>(d_input, d_output, (char) 1, (int) 3, width, height, depth);
            cudaDeviceSynchronize();
            edt_depth<<<grid1, block1>>>(d_output, depth_stride, 3, 0, depth, height, width, height_stride,  width_stride);
            cudaDeviceSynchronize();
            edt_depth<<<grid3, block3>>>(d_output, height_stride, 3, 1, height,  width, depth, width_stride, depth_stride); 
            cudaDeviceSynchronize();
            edt_depth<<<grid2, block2>>>(d_output, width_stride, 3, 2, width, depth, height, depth_stride, height_stride);
            cudaDeviceSynchronize();
            calculate_distance<<<grid, block>>>(d_output, d_distance, 3, width, height, depth);
            cudaDeviceSynchronize();
        }
        auto duration = std::chrono::high_resolution_clock::now() - start;
        long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        printf("runCUDA executed in %lld microseconds\n", ms / num_trials);    
        // double size = sizeof(T) * width * height;
        printf("Throughput = %f GB/s\n",(double) size / (ms / num_trials) * 1e-3); 
        // init_edt_3d<<<grid, block>>>(d_input, d_output, (char) 1, (int) 3, width, height, depth);
        // cudaDeviceSynchronize();
        // cudaError_t err = cudaGetLastError();
        // // now perform the edt on the first dimension (slowest dimension)
        // // each block process one line of data 
        // // depth direction
        // // coordinats order [depth, height, width]
        // // dim3 block1(1,1, 1); 
        // // dim3 grid1(height, width,1 );  
        // edt_depth<<<grid1, block1>>>(d_output, depth_stride, 3, 0, depth, height, width, height_stride,  width_stride);
        // cudaDeviceSynchronize();
        // // height direction
        // // dim3 block3(1,1,1);
        // // dim3 grid3(width, depth, 1);
        // edt_depth<<<grid3, block3>>>(d_output, height_stride, 3, 1, height,  width, depth, width_stride, depth_stride); 
        // cudaDeviceSynchronize();
        // // width direction
        // // dim3 block2(1,1,1);
        // // dim3 grid2(depth, height,1);
        // edt_depth<<<grid2, block2>>>(d_output, width_stride, 3, 2, width, depth, height, depth_stride, height_stride);
        // cudaDeviceSynchronize();

        // calculate_distance<<<grid, block>>>(d_output, d_distance, 3, width, height, depth);
        // cudaDeviceSynchronize();


        // copy data
        cudaMemcpy(outputData, d_output, width * height * depth * 3 * sizeof(int), cudaMemcpyDeviceToHost);
        // calculate the distance
        cudaDeviceSynchronize();
        cudaMemcpy(distanceData, d_distance, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_distance);
        cudaProfilerStop();
        cudaDeviceReset();

}







int main()
{
    printf("Starting\n");
	
	uint width = 384;
	uint height = 384;
	uint depth = 256;

	vector<float> inputData(width * height*depth);
    vector<char> boundData(width * height*depth);
    vector<int> outputData(width * height*depth*3); 
    vector<float> distanceData(width * height*depth); 


	ifstream fin("boundary3d.int8", ios::in | ios::binary);
	fin.read((char*)boundData.data(), boundData.size() * sizeof(boundData[0]));
	fin.close();

    runCUDA(boundData.data(), outputData.data(), distanceData.data(), width, height, depth);

    ofstream fout("output.i32", ios::out | ios::binary);
    fout.write((char*)outputData.data(), outputData.size() * sizeof(outputData[0]));
    fout.close();

    // cudaMemcpy(distanceData.data(), d_distance, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
    ofstream fout1("distance.f32", ios::out | ios::binary);
    fout1.write((char*)distanceData.data(), distanceData.size() * sizeof(distanceData[0]));
    fout1.close();

    double max_distance = *max_element(distanceData.begin(), distanceData.end());
    printf("Max distance = %f\n", max_distance); 


    // cudaFree(d_input);
    // cudaFree(d_output);
    // cudaFree(d_distance);
    // cudaProfilerStop();
    // cudaDeviceReset();


    return 0;
}
