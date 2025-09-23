#ifndef GET_BOUNDARY_HPP
#define GET_BOUNDARY_HPP
#include <cstddef>
#include <iostream>
#include <vector>
#include <tuple> 
#include "omp.h"    

namespace PM {
template<typename T>
inline std::vector<char> get_boundary_2d(T* quant_index, int N, int* dims) {
    std::vector<char> boundary;

    size_t stride[2];
    stride[0] = dims[1];
    stride[1] = 1;
    size_t n = dims[0] * dims[1];
    boundary.resize(n, 0);


    for (size_t i = 1; i < dims[0] - 1; i++) {
        for (size_t j = 1; j < dims[1] - 1; j++) {
            size_t idx = i * stride[0] + j * stride[1];
            if (quant_index[idx] != quant_index[idx - stride[0]] || quant_index[idx] != quant_index[idx + stride[0]] ||
                quant_index[idx] != quant_index[idx - stride[1]] || quant_index[idx] != quant_index[idx + stride[1]]) {
                boundary[idx] = 1;
            }
        }
    }
    // filter boundary
    // for (size_t i = 1; i < dims[0] - 1; i++) {
    //     for (size_t j = 1; j < dims[1] - 1; j++) {
    //         size_t idx = i * stride[0] + j * stride[1];
    //         if (quant_index[idx] != 0) {
    //             int left = quant_index[idx - stride[0]];
    //             int right = quant_index[idx + stride[0]];
    //             int up = quant_index[idx - stride[1]];
    //             int down = quant_index[idx + stride[1]];
    //             if (left == 0 && right == 0 && up == 0 && down == 0) {
    //                 boundary[idx] = 0;
    //                 boundary[idx - stride[0]] = 0;
    //                 boundary[idx + stride[0]] = 0;
    //                 boundary[idx - stride[1]] = 0;
    //                 boundary[idx + stride[1]] = 0;
    //             }
    //         }
    //     }
    // }
    return boundary;
}

template<typename T> 
inline std::vector<char> get_boundary_3d(T* quant_index, int N, int* dims, int num_threads = 1) { 
    // define stride
    size_t stride[3];
    stride[0] = dims[1] * dims[2];
    stride[1] = dims[2];
    stride[2] = 1;

    std::vector<char> boundary;
    size_t n = dims[0] * dims[1] * dims[2];
    boundary.resize(n, 0);
    #pragma omp parallel for collapse(1) num_threads(num_threads)
    for (size_t i = 1; i < dims[0] - 1; i++) {
        for (size_t j = 1; j < dims[1] - 1; j++) {
            for (size_t k = 1; k < dims[2] - 1; k++) {
                size_t idx = i * stride[0] + j * stride[1] + k * stride[2];
                if (quant_index[idx] != quant_index[idx - stride[0]] ||
                    quant_index[idx] != quant_index[idx + stride[0]] ||
                    quant_index[idx] != quant_index[idx - stride[1]] ||
                    quant_index[idx] != quant_index[idx + stride[1]] ||
                    quant_index[idx] != quant_index[idx - stride[2]] ||
                    quant_index[idx] != quant_index[idx + stride[2]]) {
                    boundary[idx] = 1;
                }
            }
        }
    }
    // filter boundary
    // for (size_t i = 1; i < dims[0] - 1; i++) {
    //     for (size_t j = 1; j < dims[1] - 1; j++) {
    //         for (size_t k = 1; k < dims[2] - 1; k++) {
    //             size_t idx = i * stride[0] + j * stride[1] + k * stride[2];
    //             if (quant_index[idx] != 0) {
    //                 int left = quant_index[idx - stride[0]];
    //                 int right = quant_index[idx + stride[0]];
    //                 int up = quant_index[idx - stride[1]];
    //                 int down = quant_index[idx + stride[1]];
    //                 int front = quant_index[idx - stride[2]];
    //                 int back = quant_index[idx + stride[2]];
    //                 if (left == 0 && right == 0 && up == 0 && down == 0 && front == 0 && back == 0) {
    //                     boundary[idx] = 0;
    //                     boundary[idx - stride[0]] = 0;
    //                     boundary[idx + stride[0]] = 0;
    //                     boundary[idx - stride[1]] = 0;
    //                     boundary[idx + stride[1]] = 0;
    //                     boundary[idx - stride[2]] = 0;
    //                     boundary[idx + stride[2]] = 0;
    //                 }
    //             }
    //         }
    //     }
    // }
    return boundary;
}


template <typename T_data_sign>
char get_sign(T_data_sign data) {
    char sign = (char)(((double)data > 0.0) - ((double)data < 0.0));
    return sign;
}



template<typename T> 
std::vector<char> get_boundary(T* quant_index, int N, int* dims, int thread_num = 1 ) {
    if (N == 2) {
        return get_boundary_2d(quant_index, N, dims);
    } else if (N == 3) {
        return get_boundary_3d(quant_index, N, dims,thread_num);
    } else {
        std::cout << "Error: N should be 2 or 3" << std::endl;
        exit(1);
    }
}

inline std::tuple<std::vector<char>, std::vector<char>> get_boundary_and_sign_map_2d(int* quant_index, int N, int* dims,int num_threads)
{
    std::vector<char> boundary;
    std::vector<char> sign_map;
    size_t stride[2];
    stride[0] = dims[1];
    stride[1] = 1;
    size_t n = dims[0] * dims[1];
    boundary.resize(n, 0);
    sign_map.resize(n, 0);

    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (size_t i = 1; i < dims[0] - 1; i++) {
        for (size_t j = 1; j < dims[1] - 1; j++) {
            size_t idx = i * stride[0] + j * stride[1];

            int left = quant_index[idx - stride[0]];
            int right = quant_index[idx + stride[0]];
            int up = quant_index[idx - stride[1]];
            int down = quant_index[idx + stride[1]];

            if(left != quant_index[idx] || right != quant_index[idx] || up != quant_index[idx] || down != quant_index[idx]) {
                boundary[idx] = 1;
                char sign = 0; 
                if (left != quant_index[idx]) {
                    sign = get_sign( left -quant_index[idx]); 
                } else if (right != quant_index[idx]) {
                    sign = get_sign( right -quant_index[idx]); 
                } else if (up != quant_index[idx]) {
                    sign = get_sign( up -quant_index[idx]); 
                } else if (down != quant_index[idx]) {
                    sign = get_sign( down -quant_index[idx]); 
                }
                sign_map[idx] = sign;
                // check the gradient
                double grad_x = std::abs((right - left) / 2.0);
                double grad_y = std::abs((down - up) / 2.0);
                double max_grad = std::max(grad_x, grad_y);
                if(max_grad >=1.0) {
                    sign_map[idx] = 0;
                }
            }
        }
    }
    return {std::move(boundary), std::move(sign_map)};
}

inline std::tuple<std::vector<char>, std::vector<char>> get_boundary_and_sign_map_3d(int* quant_index, int N, int* dims,int num_threads) {
    // define stride
    size_t stride[3];
    stride[0] = dims[1] * dims[2];
    stride[1] = dims[2];
    stride[2] = 1;
    std::vector<char> boundary;
    std::vector<char> sign_map;
    size_t n = dims[0] * dims[1] * dims[2];
    boundary.resize(n, 0);
    sign_map.resize(n, 0);

    // int neighbor_quant[6];
    // char signs[6]; 

    #pragma omp parallel for collapse(1) num_threads(num_threads)
    for (size_t i = 1; i < dims[0] - 1; i++) {
        for (size_t j = 1; j < dims[1] - 1; j++) {
            for (size_t k = 1; k < dims[2] - 1; k++) {
                size_t idx = i * stride[0] + j * stride[1] + k * stride[2];
                int neighbor_quant[6];
                char signs[6]; 
                int left = quant_index[idx - stride[0]];
                int right = quant_index[idx + stride[0]];
                int up = quant_index[idx - stride[1]];
                int down = quant_index[idx + stride[1]];
                int front = quant_index[idx - stride[2]];
                int back = quant_index[idx + stride[2]];

                neighbor_quant[0] = up;
                neighbor_quant[1] = down;
                neighbor_quant[2] = front;
                neighbor_quant[3] = back;
                neighbor_quant[4] = left;
                neighbor_quant[5] = right;

                if(left != quant_index[idx] || right != quant_index[idx] || up != quant_index[idx] || down != quant_index[idx] || front != quant_index[idx] || back != quant_index[idx]) {
                    boundary[idx] = 1;
                    char sign = 0; 
                    for (int i = 0; i < 6; i++) {
                        if(neighbor_quant[i] != quant_index[idx]) {
                            sign = get_sign( neighbor_quant[i] -quant_index[idx]); 
                            break;
                        }
                    }
                    sign_map[idx] = sign;
                    double grad_x = std::abs((right - left) / 2.0);
                    double grad_y = std::abs((down - up) / 2.0);
                    double grad_z = std::abs((back - front) / 2.0);
                    double max_grad = std::max(std::max(grad_x, grad_y), grad_z);   
                    if(max_grad >=1.0) {
                        sign_map[idx] = 0;
                    } 
                }
                else {
                    boundary[idx] = 0;
                }
            }
        }
    }
    return {std::move(boundary), std::move(sign_map)};
}

inline std::tuple<std::vector<char>, std::vector<char>>  get_boundary_and_sign_map(int* quant_index, int N, int* dims, int num_threads = 1) {
    if (N == 2) {
        return get_boundary_and_sign_map_2d(quant_index, N, dims,num_threads);
    } else if (N == 3) {
        return get_boundary_and_sign_map_3d(quant_index, N, dims,num_threads );
    } else {
        std::cout << "Error: N should be 2 or 3" << std::endl;
        exit(1);
    }
  }

}  // namespace PM
#endif // GET_BOUNDARY_HPP
// int main(int argc, char** argv)
// {

//     return 0;
// }
