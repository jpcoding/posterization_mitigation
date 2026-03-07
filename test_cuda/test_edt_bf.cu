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
        {int num_trials = 10;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_trials; i++)
        {
            edt_3d_bf(d_input, d_output, d_distance, width, height, depth);
        }
        auto duration = std::chrono::high_resolution_clock::now() - start;
        long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        printf("runCUDA executed in %lld microseconds\n", ms / num_trials);    
        // double size = sizeof(T) * width * height;
        printf("Throughput = %f GB/s\n",(double) size / (ms / num_trials) * 1e-3); 
    }

    {int num_trials = 10;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_trials; i++)
        {
            edt_3d(d_input, d_output, d_distance, width, height, depth);
        }
        auto duration = std::chrono::high_resolution_clock::now() - start;
        long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        printf("runCUDA executed in %lld microseconds\n", ms / num_trials);    
        // double size = sizeof(T) * width * height;
        printf("Throughput = %f GB/s\n",(double) size / (ms / num_trials) * 1e-3); 
    }

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


    ofstream fout("index.i32", ios::out | ios::binary);
    fout.write((char*)outputData.data(), outputData.size() * sizeof(outputData[0]));
    fout.close();

    ofstream fout1("distance.f32", ios::out | ios::binary);
    fout1.write((char*)distanceData.data(), distanceData.size() * sizeof(distanceData[0]));
    fout1.close();

    float max_distance = *std::max_element(distanceData.begin(), distanceData.end());
    printf("Max distance: %f\n", max_distance);



    // cudaFree(d_input);
    // cudaFree(d_output);
    // cudaFree(d_distance);
    // cudaProfilerStop();
    // cudaDeviceReset();


    return 0;
}
