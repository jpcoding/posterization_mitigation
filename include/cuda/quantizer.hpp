#ifndef LinearQuantizer_hpp
#define LinearQuantizer_hpp
#include <cuda_profiler_api.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cmath>


template <typename T> 
__global__ void quantize_and_overwrite(T* d_input_data, const T pred, T* d_quant_inds, double abs_eb, size_t global_szize, const int radius = 32768)
{

    // int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    // if (idx >= global_szize) return; 
    // T* data = d_input_data + idx; 
    // T* quant_inds = d_quant_inds + idx; 
    // double error_bound_reciprocal = 1.0 / abs_eb; 
    // T diff = data - pred;
    // auto quant_index = static_cast<int64_t>(fabs(diff) * error_bound_reciprocal) + 1;
    // if (quant_index < radius * 2) {
    //     quant_index >>= 1;
    //     int half_index = quant_index;
    //     quant_index <<= 1;
    //     int quant_index_shifted;
    //     if (diff < 0) {
    //         quant_index = -quant_index;
    //         quant_index_shifted = radius - half_index;
    //     } else {
    //         quant_index_shifted = radius + half_index;
    //     }
    //     T decompressed_data = pred + quant_index * error_bound;
    //     if (fabs(decompressed_data - data) > error_bound) {
    //         unpred.push_back(data);
    //         quant_inds =  0;
    //     } else {
    //         data = decompressed_data;
    //         quant_inds =  quant_index_shifted;
    //     }
    // } else {
    //     unpred.push_back(data);
    //     quant_inds = 0;
    // }
}









#endif