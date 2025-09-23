#ifndef EDT_MPI
#define EDT_MPI

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "utils/file_utils.hpp"
// this function does not require communication between processes
void edt_init_mpi(char *input, int *output, char b_tag, uint width, uint height, uint depth, int *mpi_coords) {
    size_t block_size = width * height * depth;
    int dim = 3;
    int global_z = mpi_coords[0] * depth;
    int global_y = mpi_coords[1] * height;
    int global_x = mpi_coords[2] * width;
    for (size_t i = 0; i < block_size; i++) {
        int z = i / (width * height);
        int y = (i % (width * height)) / width;
        int x = (i % (width * height)) % width;
        if (input[i] == b_tag) {
            output[i * dim] = z + global_z;
            output[i * dim + 1] = y + global_y;
            output[i * dim + 2] = x + global_x;
        } else {
            output[i * dim] = -1;
            //   // output[i*dim + 1] = -1;
            //   // output[i*dim + 2] = -1;
            //   output[i*dim] = z + global_z;
            //   output[i*dim + 1] = y + global_y;
            //   output[i*dim + 2] = x + global_x;
        }
    }
}

template <typename T>
void calculate_distance(int *output, T *distance, uint width, uint height, uint depth, int *mpi_coords) {
    size_t block_size = width * height * depth;
    int dim = 3;
    int global_z = mpi_coords[0] * depth;
    int global_y = mpi_coords[1] * height;
    int global_x = mpi_coords[2] * width;
    for (size_t i = 0; i < block_size; i++) {
        int z = i / (width * height);
        int y = (i % (width * height)) / width;
        int x = (i % (width * height)) % width;
        int dz = output[i * dim] - z - global_z;
        int dy = output[i * dim + 1] - y - global_y;
        int dx = output[i * dim + 2] - x - global_x;
        distance[i] = dz * dz + dy * dy + dx * dx;
        distance[i] = std::sqrt(distance[i]);
    }
}

template <typename T, typename T_index>
void calculate_distance_and_index(int *output, T *distance, T_index *indexes, int width, int height, int depth,
                                  int global_width, int global_height, int global_depth, int *mpi_coords) {
    size_t block_size = width * height * depth;
    int dim = 3;
    int global_z = mpi_coords[0] * depth;
    int global_y = mpi_coords[1] * height;
    int global_x = mpi_coords[2] * width;
    size_t  global_strides[3] = {(size_t)global_width * global_height, (size_t)global_width, 1};
    for (size_t i = 0; i < block_size; i++) {
        int z = i / (width * height);
        int y = (i % (width * height)) / width;
        int x = (i % (width * height)) % width;
        int dz = output[i * dim] - z - global_z;
        int dy = output[i * dim + 1] - y - global_y;
        int dx = output[i * dim + 2] - x - global_x;
        distance[i] = dz * dz + dy * dy + dx * dx;
        distance[i] = std::sqrt(distance[i]);
        // indexes should be global index
        indexes[i] =
            (output[i * dim]) * global_strides[0] + (output[i * dim + 1]) * global_strides[1] + (output[i * dim + 2]);
    }
}

inline void edt_core_mpi(int *d_output,  size_t stride, uint rank, uint d, uint len, uint width, uint height,
                  size_t width_stride, size_t height_stride, int local_x, int local_y, int **f, int *f_send_buffer,
                  int mpi_rank, int mpi_size, int *mpi_coords, int mpi_depth, int mpi_height, int mpi_width,
                  int mpi_direction, int *mpi_dims, MPI_Comm &cart_comm) {
    int l = -1, ii, maxl, idx1, idx2, jj;

    int dim = 3;
    int global_z = mpi_coords[0] * mpi_depth;
    int global_y = mpi_coords[1] * mpi_height;
    int global_x = mpi_coords[2] * mpi_width;

    int global_coord[3];
    global_coord[0] = global_z;
    global_coord[1] = global_y;
    global_coord[2] = global_x;
    int coor[3];
    coor[d] = 0;
    coor[(d + 1) % 3] = local_x + global_coord[(d + 1) % 3];
    coor[(d + 2) % 3] = local_y + global_coord[(d + 2) % 3];

    int global_line_size = len * mpi_dims[d];
    //   global_line_size = 384;

    //   MPI_Comm_rank(cart_comm, &mpi_rank);

    int *d_output_start = d_output + local_x * width_stride + local_y * height_stride;

    // int chunck_start = global_coord[d];
    int chunck_start = 0;

    // std::vector<char> local_sign_buffer_vector(global_line_size, 0);

    for (ii = 0; ii < len; ii++) {
        for (jj = 0; jj < rank; jj++) {
            f[ii + chunck_start][jj] = d_output_start[ii * stride + jj];
        }
    }

    MPI_Status MPI_status;

    int *fptr = f[0];
    int f_buffer_length = 3 * len * mpi_dims[d];
    // changing size of each time how much to pass

    if (mpi_coords[d] != 0) {
        int sender_coords[3] = {0, 0, 0};
        int sender_rank;
        MPI_Cart_coords(cart_comm, mpi_rank, 3, sender_coords);
        sender_coords[d] = mpi_coords[d] - 1;
        MPI_Cart_rank(cart_comm, sender_coords, &sender_rank);
        MPI_Recv(&l, 1, MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
        if (1) {
            MPI_Recv(f_send_buffer, 3 * (l + 1), MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
        }
    }

    for (ii = 0; ii < len; ii++) {
        if (f[ii][0] >= 0) {
            double fd = f[ii][d];
            double wR = 0.0;
            for (jj = 0; jj < rank; jj++) {
                if (jj != d) {
                    int tw = (f[ii][jj] - coor[jj]);
                    wR += tw * tw;
                }
            }
            while (l >= 1) {
                double a, b, c, uR = 0.0, vR = 0.0, f1;
                idx1 = l;
                idx2 = l - 1;
                f1 = f_send_buffer[idx1 * 3 + d];
                a = f1 - f_send_buffer[idx2 * 3 + d];
                b = fd - f1;
                c = a + b;
                for (jj = 0; jj < rank; jj++) {
                    if (jj != d) {
                        double cc = coor[jj];
                        double tu = f_send_buffer[idx2 * 3 + jj] - cc;
                        double tv = f_send_buffer[idx1 * 3 + jj] - cc;
                        uR += tu * tu;
                        vR += tv * tv;
                    }
                }
                if (c * vR - b * uR - a * wR - a * b * c <= 0.0) {
                    break;
                }
                --l;
            }
            ++l;
            // g[l] = ii;
            for (int jj = 0; jj < 3; jj++) {
                f_send_buffer[l * 3 + jj] = f[ii][jj];
            }
        }
    }

    if (mpi_coords[d] < mpi_dims[d] - 1) {
        // copy the g to the global g
        int recr_coords[3] = {0, 0, 0};
        int receiver_rank;
        MPI_Cart_coords(cart_comm, mpi_rank, 3, recr_coords);
        recr_coords[d] = mpi_coords[d] + 1;
        MPI_Cart_rank(cart_comm, recr_coords, &receiver_rank);
        MPI_Send(&l, 1, MPI_INT, receiver_rank, 0, cart_comm);
        if (1) {
            MPI_Send(f_send_buffer, 3 * (l + 1), MPI_INT, receiver_rank, 0, cart_comm);  // adjust
        }
    }
    // barrier

    MPI_Barrier(cart_comm);
    // now we have to update the g and f for all the previous blocks

    fptr = f[0];

    // TODO
    //  1. optimize the communication by only pass necessary length of f;

    if (mpi_coords[d] == mpi_dims[d] - 1) {
        int aux_coord[3] = {0, 0, 0};
        int cur_rank;
        MPI_Comm_rank(cart_comm, &cur_rank);
        MPI_Cart_coords(cart_comm, cur_rank, 3, aux_coord);
        for (int i = 0; i < mpi_dims[d] - 1; i++) {
            aux_coord[d] = i;
            int receiver_id;
            MPI_Cart_rank(cart_comm, aux_coord, &receiver_id);
            MPI_Send(&l, 1, MPI_INT, receiver_id, 0, cart_comm);
            MPI_Send(f_send_buffer, 3 * (l + 1), MPI_INT, receiver_id, 0, cart_comm);  // adjust
        }
    } else {
        // receive informaton from the last block
        int aux_coord[3] = {0, 0, 0};
        int cur_rank;
        MPI_Comm_rank(cart_comm, &cur_rank);
        MPI_Cart_coords(cart_comm, cur_rank, 3, aux_coord);
        aux_coord[d] = mpi_dims[d] - 1;
        int sender_id;
        MPI_Cart_rank(cart_comm, aux_coord, &sender_id);
        MPI_Recv(&l, 1, MPI_INT, sender_id, 0, cart_comm, &MPI_status);
        MPI_Recv(f_send_buffer, 3 * (l + 1), MPI_INT, sender_id, 0, cart_comm, &MPI_status);
    }

    maxl = l;
    chunck_start = global_coord[d];
    // no need for communication since we have the g and f for all the blocks
    if (maxl >= 0) {
        l = 0;
        for (ii = chunck_start; ii < len + chunck_start; ii++) {
            double delta1 = 0.0, t;
            for (jj = 0; jj < rank; jj++) {
                t = jj == d ? f_send_buffer[l * 3 + jj] - ii : f_send_buffer[l * 3 + jj] - coor[jj];
                delta1 += t * t;
            }
            while (l < maxl) {
                double delta2 = 0.0;
                for (jj = 0; jj < rank; jj++) {
                    t = jj == d ? f_send_buffer[(l + 1) * 3 + jj] - ii : f_send_buffer[(l + 1) * 3 + jj] - coor[jj];
                    delta2 += t * t;
                }
                if (delta1 <= delta2) break;
                delta1 = delta2;
                ++l;
            }
            idx1 = l;
            for (jj = 0; jj < rank; jj++) {
                d_output_start[(ii - chunck_start) * stride + jj] = f_send_buffer[idx1 * 3 + jj];
            }
        }
    }
}

inline void edt_and_sign_core_mpi(int *d_output, char *sign_map, size_t stride, uint rank, uint d, uint len, uint width,
                           uint height, size_t width_stride, size_t height_stride, int local_x, int local_y, int **f,
                           char *local_sign_buffer, int *f_send_buffer, char *sign_send_buffer, int mpi_rank,
                           int mpi_size, int *mpi_coords, int mpi_depth, int mpi_height, int mpi_width,
                           int mpi_direction, int *mpi_dims, MPI_Comm &cart_comm) {
    int l = -1, ii, maxl, idx1, idx2, jj;

    int dim = 3;
    int global_z = mpi_coords[0] * mpi_depth;
    int global_y = mpi_coords[1] * mpi_height;
    int global_x = mpi_coords[2] * mpi_width;

    int global_coord[3];
    global_coord[0] = global_z;
    global_coord[1] = global_y;
    global_coord[2] = global_x;
    int coor[3];
    coor[d] = 0;
    coor[(d + 1) % 3] = local_x + global_coord[(d + 1) % 3];
    coor[(d + 2) % 3] = local_y + global_coord[(d + 2) % 3];

    int global_line_size = len * mpi_dims[d];
    //   global_line_size = 384;

    //   MPI_Comm_rank(cart_comm, &mpi_rank);

    int *d_output_start = d_output + local_x * width_stride + local_y * height_stride;
    char *sign_start = sign_map + local_x * width_stride / 3 + local_y * height_stride / 3;

    // int chunck_start = global_coord[d];
    int chunck_start = 0;

    // std::vector<char> local_sign_buffer_vector(global_line_size, 0);

    for (ii = 0; ii < len; ii++) {
        for (jj = 0; jj < rank; jj++) {
            f[ii + chunck_start][jj] = d_output_start[ii * stride + jj];
        }
        local_sign_buffer[ii + chunck_start] = sign_start[ii * stride / 3];
    }

    MPI_Status MPI_status;

    int *fptr = f[0];
    int f_buffer_length = 3 * len * mpi_dims[d];
    // changing size of each time how much to pass

    if (mpi_coords[d] != 0) {
        int sender_coords[3] = {0, 0, 0};
        int sender_rank;
        MPI_Cart_coords(cart_comm, mpi_rank, 3, sender_coords);
        sender_coords[d] = mpi_coords[d] - 1;
        MPI_Cart_rank(cart_comm, sender_coords, &sender_rank);
        MPI_Recv(&l, 1, MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
        if (1) {
            MPI_Recv(f_send_buffer, 3 * (l + 1), MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
            MPI_Recv(sign_send_buffer, l + 1, MPI_CHAR, sender_rank, 0, cart_comm, &MPI_status);
        }
    }

    for (ii = 0; ii < len; ii++) {
        if (f[ii][0] >= 0) {
            double fd = f[ii][d];
            int wR = 0.0;
            for (jj = 0; jj < rank; jj++) {
                if (jj != d) {
                    int tw = (f[ii][jj] - coor[jj]);
                    wR += tw * tw;
                }
            }
            while (l >= 1) {
                double a, b, c, uR = 0.0, vR = 0.0, f1;
                idx1 = l;
                idx2 = l - 1;
                f1 = f_send_buffer[idx1 * 3 + d];
                a = f1 - f_send_buffer[idx2 * 3 + d];
                b = fd - f1;
                c = a + b;
                for (jj = 0; jj < rank; jj++) {
                    if (jj != d) {
                        double cc = coor[jj];
                        double tu = f_send_buffer[idx2 * 3 + jj] - cc;
                        double tv = f_send_buffer[idx1 * 3 + jj] - cc;
                        uR += tu * tu;
                        vR += tv * tv;
                    }
                }
                if (c * vR - b * uR - a * wR - a * b * c <= 0.0) {
                    break;
                }
                --l;
            }
            ++l;
            for (int jj = 0; jj < 3; jj++) {
                f_send_buffer[l * 3 + jj] = f[ii][jj];
            }
            sign_send_buffer[l] = local_sign_buffer[ii];
        }
    }

    if (mpi_coords[d] < mpi_dims[d] - 1) {
        // copy the g to the global g
        int recr_coords[3] = {0, 0, 0};
        int receiver_rank;
        MPI_Cart_coords(cart_comm, mpi_rank, 3, recr_coords);
        recr_coords[d] = mpi_coords[d] + 1;
        MPI_Cart_rank(cart_comm, recr_coords, &receiver_rank);
        MPI_Send(&l, 1, MPI_INT, receiver_rank, 0, cart_comm);
        if (1) {
            MPI_Send(f_send_buffer, 3 * (l + 1), MPI_INT, receiver_rank, 0, cart_comm);  // adjust
            MPI_Send(sign_send_buffer, l + 1, MPI_CHAR, receiver_rank, 0, cart_comm);    // adjust
        }
    }
    // barrier

    MPI_Barrier(cart_comm);
    // now we have to update the g and f for all the previous blocks

    fptr = f[0];
    if (mpi_coords[d] == mpi_dims[d] - 1) {
        int aux_coord[3] = {0, 0, 0};
        int cur_rank;
        MPI_Comm_rank(cart_comm, &cur_rank);
        MPI_Cart_coords(cart_comm, cur_rank, 3, aux_coord);
        for (int i = 0; i < mpi_dims[d] - 1; i++) {
            aux_coord[d] = i;
            int receiver_id;
            MPI_Cart_rank(cart_comm, aux_coord, &receiver_id);
            MPI_Send(&l, 1, MPI_INT, receiver_id, 0, cart_comm);
            MPI_Send(f_send_buffer, 3 * (l + 1), MPI_INT, receiver_id, 0, cart_comm);  // adjust
            MPI_Send(sign_send_buffer, l + 1, MPI_CHAR, receiver_id, 0, cart_comm);    // adjust
        }
    } else {
        // receive informaton from the last block
        int aux_coord[3] = {0, 0, 0};
        int cur_rank;
        MPI_Comm_rank(cart_comm, &cur_rank);
        MPI_Cart_coords(cart_comm, cur_rank, 3, aux_coord);
        aux_coord[d] = mpi_dims[d] - 1;
        int sender_id;
        MPI_Cart_rank(cart_comm, aux_coord, &sender_id);
        MPI_Recv(&l, 1, MPI_INT, sender_id, 0, cart_comm, &MPI_status);
        MPI_Recv(f_send_buffer, 3 * (l + 1), MPI_INT, sender_id, 0, cart_comm, &MPI_status);
        MPI_Recv(sign_send_buffer, l + 1, MPI_CHAR, sender_id, 0, cart_comm, &MPI_status);
    }

    maxl = l;
    chunck_start = global_coord[d];
    // no need for communication since we have the g and f for all the blocks
    if (maxl >= 0) {
        l = 0;
        for (ii = chunck_start; ii < len + chunck_start; ii++) {
            double delta1 = 0.0, t;
            for (jj = 0; jj < rank; jj++) {
                t = jj == d ? f_send_buffer[l * 3 + jj] - ii : f_send_buffer[l * 3 + jj] - coor[jj];
                delta1 += t * t;
            }
            while (l < maxl) {
                double delta2 = 0.0;
                for (jj = 0; jj < rank; jj++) {
                    t = jj == d ? f_send_buffer[(l + 1) * 3 + jj] - ii : f_send_buffer[(l + 1) * 3 + jj] - coor[jj];
                    delta2 += t * t;
                }
                if (delta1 <= delta2) break;
                delta1 = delta2;
                ++l;
            }
            idx1 = l;
            for (jj = 0; jj < rank; jj++) {
                d_output_start[(ii - chunck_start) * stride + jj] = f_send_buffer[idx1 * 3 + jj];
            }
            sign_start[(ii - chunck_start) * stride / 3] = sign_send_buffer[idx1];
        }
    }
}

void edt_and_sign_core_mpi_partial(int *d_output, char *sign_map, size_t stride, uint rank, uint d, uint len,
                                   uint width, uint height, size_t width_stride, size_t height_stride, int local_x,
                                   int local_y, int **f, int *g, char *sign_buffer, int mpi_rank, int mpi_size,
                                   int *mpi_coords, int mpi_depth, int mpi_height, int mpi_width, int mpi_direction,
                                   int *mpi_dims, MPI_Comm &cart_comm) {
    int l = -1, ii, maxl, idx1, idx2, jj;

    int dim = 3;
    int global_z = mpi_coords[0] * mpi_depth;
    int global_y = mpi_coords[1] * mpi_height;
    int global_x = mpi_coords[2] * mpi_width;

    int global_coord[3];
    global_coord[0] = global_z;
    global_coord[1] = global_y;
    global_coord[2] = global_x;
    int coor[3];
    coor[d] = 0;
    coor[(d + 1) % 3] = local_x + global_coord[(d + 1) % 3];
    coor[(d + 2) % 3] = local_y + global_coord[(d + 2) % 3];
    int global_line_size = len * mpi_dims[d];
    int *d_output_start = d_output + local_x * width_stride + local_y * height_stride;
    char *sign_start = sign_map + local_x * width_stride / 3 + local_y * height_stride / 3;
    int chunck_start = global_coord[d];
    char *local_sign_buffer = sign_buffer;
    for (ii = 0; ii < len; ii++) {
        for (jj = 0; jj < rank; jj++) {
            f[ii + chunck_start][jj] = d_output_start[ii * stride + jj];
        }
        local_sign_buffer[ii + chunck_start] = sign_start[ii * stride / 3];
    }

    MPI_Status MPI_status;

    int *fptr = f[0];
    int f_buffer_length = 3 * len * mpi_dims[d];
    // changing size of each time how much to pass
    std::vector<int> f_send_buffer(f_buffer_length, 0);
    std::vector<int> f_recv_buffer(f_buffer_length, 0);
    std::vector<char> sign_send_buffer(len * mpi_dims[d], 0);
    std::vector<char> sign_recv_buffer(len * mpi_dims[d], 0);

    int max_pass = 5;
    // pass up to 5 elements to the next block

    if (mpi_coords[d] != 0) {
        int sender_coords[3] = {0, 0, 0};
        int sender_rank;
        MPI_Cart_coords(cart_comm, mpi_rank, 3, sender_coords);
        sender_coords[d] = mpi_coords[d] - 1;
        MPI_Cart_rank(cart_comm, sender_coords, &sender_rank);
        MPI_Recv(&l, 1, MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
        // MPI_Recv(g, global_line_size, MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
        // MPI_Recv(fptr, len * rank * (mpi_coords[d]), MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
        // MPI_Recv(local_sign_buffer, len * (mpi_coords[d]), MPI_CHAR, sender_rank, 0, cart_comm, &MPI_status);
        // printf("Rank %d received l from %d, l = %d, status =%d \n", mpi_rank,
        //    neighbor_up, l, MPI_status.MPI_ERROR);
        // wait for the top block to finish

        if (1) {
            int pass_elements = l > max_pass ? max_pass : l;
            int g_start = l - pass_elements + 1;
            MPI_Recv(&g[g_start], pass_elements + 1, MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
            MPI_Recv(f_recv_buffer.data(), 3 * (pass_elements + 1), MPI_INT, sender_rank, 0, cart_comm, &MPI_status);
            MPI_Recv(sign_recv_buffer.data(), pass_elements + 1, MPI_CHAR, sender_rank, 0, cart_comm, &MPI_status);
            for (int i = 0; i < l + 1; i++) {
                for (int j = 0; j < 3; j++) {
                    f[g[i]][j] = f_recv_buffer[i * 3 + j];
                }
                local_sign_buffer[g[i]] = sign_recv_buffer[i];
            }
        }
    }

    for (ii = chunck_start; ii < len + chunck_start; ii++) {
        if (f[ii][0] >= 0) {
            int fd = f[ii][d];
            int wR = 0.0;
            for (jj = 0; jj < rank; jj++) {
                if (jj != d) {
                    int tw = (f[ii][jj] - coor[jj]);
                    wR += tw * tw;
                }
            }
            while (l >= 1) {
                int a, b, c, uR = 0.0, vR = 0.0, f1;
                idx1 = g[l];
                idx2 = g[l - 1];
                f1 = f[idx1][d];
                a = f1 - f[idx2][d];
                b = fd - f1;
                c = a + b;
                for (jj = 0; jj < rank; jj++) {
                    if (jj != d) {
                        int cc = coor[jj];
                        int tu = f[idx2][jj] - cc;
                        int tv = f[idx1][jj] - cc;
                        uR += tu * tu;
                        vR += tv * tv;
                    }
                }
                if (c * vR - b * uR - a * wR - a * b * c <= 0.0) {
                    break;
                }
                --l;
            }
            ++l;
            g[l] = ii;
        }
    }

    if (mpi_coords[d] < mpi_dims[d] - 1) {
        // copy the g to the global g
        int recr_coords[3] = {0, 0, 0};
        int receiver_rank;
        MPI_Cart_coords(cart_comm, mpi_rank, 3, recr_coords);
        recr_coords[d] = mpi_coords[d] + 1;
        MPI_Cart_rank(cart_comm, recr_coords, &receiver_rank);
        MPI_Send(&l, 1, MPI_INT, receiver_rank, 0, cart_comm);
        // MPI_Send(g, global_line_size, MPI_INT, receiver_rank, 0, cart_comm);
        // MPI_Send(fptr, len * rank * (mpi_coords[d] + 1), MPI_INT, receiver_rank, 0, cart_comm);
        // MPI_Send(local_sign_buffer, len * (mpi_coords[d] + 1), MPI_CHAR, receiver_rank, 0, cart_comm);
        if (1) {
            int pass_elements = l > max_pass ? max_pass : l;
            int g_start = l - pass_elements + 1;
            MPI_Send(&g[g_start], pass_elements + 1, MPI_INT, receiver_rank, 0,
                     cart_comm);  // limit the number of g indexes.
            for (int i = 0; i < pass_elements + 1; i++) {
                for (int j = 0; j < 3; j++) {
                    f_send_buffer[i * 3 + j] = f[g[i]][j];
                }
                sign_send_buffer[i] = local_sign_buffer[g[i]];
            }
            MPI_Send(f_send_buffer.data(), 3 * (pass_elements + 1), MPI_INT, receiver_rank, 0, cart_comm);  // adjust
            MPI_Send(sign_send_buffer.data(), pass_elements + 1, MPI_CHAR, receiver_rank, 0, cart_comm);    // adjust
        }
    }
    // barrier

    MPI_Barrier(cart_comm);
    // now we have to update the g and f for all the previous blocks

    fptr = f[0];

    // TODO
    //  1. optimize the communication by only pass necessary length of f;

    if (mpi_coords[d] == mpi_dims[d] - 1) {
        int aux_coord[3] = {0, 0, 0};
        int cur_rank;
        MPI_Comm_rank(cart_comm, &cur_rank);
        MPI_Cart_coords(cart_comm, cur_rank, 3, aux_coord);
        for (int i = 0; i < mpi_dims[d] - 1; i++) {
            aux_coord[d] = i;
            int receiver_id;
            MPI_Cart_rank(cart_comm, aux_coord, &receiver_id);
            MPI_Send(&l, 1, MPI_INT, receiver_id, 0, cart_comm);
            // MPI_Send(g, global_line_size, MPI_INT, receiver_id, 0, cart_comm);
            // MPI_Send(fptr, global_line_size * 3, MPI_INT, receiver_id, 0, cart_comm);  // send the whole array
            // MPI_Send(local_sign_buffer, global_line_size, MPI_CHAR, receiver_id, 0, cart_comm);  // send the whole
            // array

            MPI_Send(g, l + 1, MPI_INT, receiver_id, 0, cart_comm);  // limit the number of g indexes.
            for (int i = 0; i < l + 1; i++) {
                for (int j = 0; j < 3; j++) {
                    f_send_buffer[i * 3 + j] = f[g[i]][j];
                }
                sign_send_buffer[i] = local_sign_buffer[g[i]];
            }
            MPI_Send(f_send_buffer.data(), 3 * (l + 1), MPI_INT, receiver_id, 0, cart_comm);  // adjust
            MPI_Send(sign_send_buffer.data(), l + 1, MPI_CHAR, receiver_id, 0, cart_comm);    // adjust
        }
    } else {
        // receive informaton from the last block
        int aux_coord[3] = {0, 0, 0};
        int cur_rank;
        MPI_Comm_rank(cart_comm, &cur_rank);
        MPI_Cart_coords(cart_comm, cur_rank, 3, aux_coord);
        aux_coord[d] = mpi_dims[d] - 1;
        int sender_id;
        MPI_Cart_rank(cart_comm, aux_coord, &sender_id);
        MPI_Recv(&l, 1, MPI_INT, sender_id, 0, cart_comm, &MPI_status);
        // MPI_Recv(g, global_line_size, MPI_INT, sender_id, 0, cart_comm, &MPI_status);
        // MPI_Recv(fptr, global_line_size * 3, MPI_INT, sender_id, 0, cart_comm, &MPI_status);
        // MPI_Recv(local_sign_buffer, global_line_size, MPI_CHAR, sender_id, 0, cart_comm, &MPI_status);
        MPI_Recv(g, l + 1, MPI_INT, sender_id, 0, cart_comm, &MPI_status);
        MPI_Recv(f_recv_buffer.data(), 3 * (l + 1), MPI_INT, sender_id, 0, cart_comm, &MPI_status);
        MPI_Recv(sign_recv_buffer.data(), l + 1, MPI_CHAR, sender_id, 0, cart_comm, &MPI_status);
        for (int i = 0; i < l + 1; i++) {
            for (int j = 0; j < 3; j++) {
                f[g[i]][j] = f_recv_buffer[i * 3 + j];
            }
            local_sign_buffer[g[i]] = sign_recv_buffer[i];
        }
        // printf("Rank %d, sender_rank: %d,  status %d \n", cur_rank, sender_id,
        // MPI_status.MPI_ERROR);
    }

    maxl = l;

    // no need for communication since we have the g and f for all the blocks
    if (maxl >= 0) {
        l = 0;
        for (ii = chunck_start; ii < len + chunck_start; ii++) {
            int delta1 = 0.0, t;
            for (jj = 0; jj < rank; jj++) {
                t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];
                delta1 += t * t;
            }
            while (l < maxl) {
                int delta2 = 0.0;
                for (jj = 0; jj < rank; jj++) {
                    t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];
                    delta2 += t * t;
                }
                if (delta1 <= delta2) break;
                delta1 = delta2;
                ++l;
            }
            idx1 = g[l];
            for (jj = 0; jj < rank; jj++) {
                d_output_start[(ii - chunck_start) * stride + jj] = f[idx1][jj];
            }
            sign_start[(ii - chunck_start) * stride / 3] = local_sign_buffer[idx1];
        }
    }
}

template <typename T_boundary, typename T_distance, typename T_index>
void edt_3d(T_boundary *boundary, T_distance *distance, T_index *index, int *data_block_dims, int *mpi_dims,
            int *mpi_coords, int mpi_rank, int size, MPI_Comm cart_comm) {
    size_t block_size = (size_t) data_block_dims[0] * (size_t) data_block_dims[1] * (size_t) data_block_dims[2];
    std::vector<int> output_data =
        std::vector<int>(block_size * 3, 0);

    // make output data as a smart pointer

    int *output = output_data.data();
    // edt_init_mpi(boundary, output, 1, data_block_dims[2],
    //     data_block_dims[1], data_block_dims[0], mpi_coords);
    int max_dim = *std::max_element(data_block_dims, data_block_dims + 3);
    int max_mpi_dim = *std::max_element(mpi_dims, mpi_dims + 3);


    // local buffer 
    int *f = (int *)malloc(sizeof(int) * max_dim * 3 );
    int **ff = (int **)malloc(sizeof(int *) * max_dim);
    for (int i = 0; i < max_dim ; i++) {
        ff[i] = f + i * 3;
    }

    int* f_send_buffer = (int *)malloc(sizeof(int) * max_dim * 3 * max_mpi_dim);


    size_t input_stride[3] = {(size_t)data_block_dims[2] * data_block_dims[1], (size_t)data_block_dims[2], (size_t)1};
    size_t output_stride[3] = {(size_t)3 * input_stride[0], (size_t)3 * input_stride[1], (size_t)3 * input_stride[2]};

    int global_dims[3] = {data_block_dims[0] * mpi_dims[0], data_block_dims[1] * mpi_dims[1],
                          data_block_dims[2] * mpi_dims[2]};

    int direction;
    // dim 0
    direction = 0;
    int x_dir = (direction + 1) % 3;
    int y_dir = (direction + 2) % 3;

    if (0) {
        int i = 10;
        int j = 10;
        edt_core_mpi(output, output_stride[direction], 3, direction, data_block_dims[direction], data_block_dims[x_dir],
                     data_block_dims[y_dir], output_stride[x_dir], output_stride[y_dir], i, j, ff, f_send_buffer, mpi_rank, size,
                     mpi_coords, data_block_dims[0], data_block_dims[1], data_block_dims[2], direction, mpi_dims,
                     cart_comm);
    }

    // speed test
    // barrier
    MPI_Barrier(cart_comm);
    double time = MPI_Wtime();
    edt_init_mpi(boundary, output, 1, data_block_dims[2], data_block_dims[1], data_block_dims[0], mpi_coords);


    // void edt_core_mpi(int *d_output, char *sign_map, size_t stride, uint rank, uint d, uint len, uint width, uint height,
    //     size_t width_stride, size_t height_stride, int local_x, int local_y, int **f, int *f_send_buffer,
    //     int mpi_rank, int mpi_size, int *mpi_coords, int mpi_depth, int mpi_height, int mpi_width,
    //     int mpi_direction, int *mpi_dims, MPI_Comm &cart_comm) 
    if (1) {
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_core_mpi(output, output_stride[direction], 3, direction, data_block_dims[direction],
                             data_block_dims[x_dir], data_block_dims[y_dir], output_stride[x_dir], output_stride[y_dir],
                             i, j, ff, f_send_buffer, mpi_rank, size, mpi_coords, data_block_dims[0], data_block_dims[1],
                             data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        MPI_Barrier(cart_comm);
        // printf("rank %d finished dim 0\n", mpi_rank);
    }

    // dim1
    direction = 1;
    if (1) {
        x_dir = (direction + 1) % 3;
        y_dir = (direction + 2) % 3;
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_core_mpi(output, output_stride[direction], 3, direction, data_block_dims[direction],
                             data_block_dims[x_dir], data_block_dims[y_dir], output_stride[x_dir], output_stride[y_dir],
                             i, j, ff, f_send_buffer, mpi_rank, size, mpi_coords, data_block_dims[0], data_block_dims[1],
                             data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        MPI_Barrier(cart_comm);
        // printf("rank %d finished dim 1\n", mpi_rank);
    }

    // dim 2
    if (1) {
        direction = 2;
        x_dir = (direction + 1) % 3;
        y_dir = (direction + 2) % 3;
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_core_mpi(output, output_stride[direction], 3, direction, data_block_dims[direction],
                             data_block_dims[x_dir], data_block_dims[y_dir], output_stride[x_dir], output_stride[y_dir],
                             i, j, ff, f_send_buffer, mpi_rank, size, mpi_coords, data_block_dims[0], data_block_dims[1],
                             data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        MPI_Barrier(cart_comm);
        // printf("rank %d finished dim 2\n", mpi_rank);
    }

    // calculate distance
    // calculate_distance(output, distance, data_block_dims[2], data_block_dims[1], data_block_dims[0], mpi_coords);
    calculate_distance_and_index(output, distance, index, data_block_dims[2], data_block_dims[1], data_block_dims[0],
                                 global_dims[2], global_dims[1], global_dims[0], mpi_coords);

    // free(g);
    free(f);
    free(ff);
    free(f_send_buffer);
}

template <typename T_boundary, typename T_distance, typename T_index>
void edt_3d_and_sign_map(T_boundary *boundary, T_distance *distance, T_index *indexes, char *sign_map,
                         int *data_block_dims, int *mpi_dims, int *mpi_coords, int mpi_rank, int size,
                         MPI_Comm cart_comm) {
    size_t block_size = (size_t) data_block_dims[0] * (size_t) data_block_dims[1] * (size_t) data_block_dims[2];

    std::vector<int> output_data =
        std::vector<int>(block_size * 3, 0);

    int *output = output_data.data();
    // edt_init_mpi(boundary, output, 1, data_block_dims[2],
    //     data_block_dims[1], data_block_dims[0], mpi_coords);
    int max_dim = *std::max_element(data_block_dims, data_block_dims + 3);
    int max_mpi_dim = *std::max_element(mpi_dims, mpi_dims + 3);

    int global_dims[3] = {data_block_dims[0] * mpi_dims[0], data_block_dims[1] * mpi_dims[1],
                          data_block_dims[2] * mpi_dims[2]};

    // send and receive buffer
    // int *g = (int *)malloc(sizeof(int) * max_dim * max_mpi_dim);
    int *f_send_buffer = (int *)malloc(sizeof(int) * max_dim * 3 * max_mpi_dim);
    char *sign_send_buffer = (char *)malloc(sizeof(char) * max_dim * max_mpi_dim);

    // local data buffer to store the line
    char *local_sign_buffer = (char *)malloc(sizeof(char) * max_dim);
    int *f = (int *)malloc(sizeof(int) * max_dim * 3);
    int **ff = (int **)malloc(sizeof(int *) * max_dim);
    for (int i = 0; i < max_dim; i++) {
        ff[i] = f + i * 3;
    }

    size_t input_stride[3] = {(size_t)data_block_dims[2] * data_block_dims[1], (size_t)data_block_dims[2], 1};
    size_t output_stride[3] = {(size_t)3 * input_stride[0], (size_t)3 * input_stride[1], (size_t)3 * input_stride[2]};

    int direction;
    // dim 0
    direction = 0;
    int x_dir = (direction + 1) % 3;
    int y_dir = (direction + 2) % 3;

    if (0) {
        int i = 10;
        int j = 10;
        edt_and_sign_core_mpi(output, sign_map, output_stride[direction], 3, direction, data_block_dims[direction],
                              data_block_dims[x_dir], data_block_dims[y_dir], output_stride[x_dir],
                              output_stride[y_dir], i, j, ff, local_sign_buffer, f_send_buffer, sign_send_buffer,
                              mpi_rank, size, mpi_coords, data_block_dims[0], data_block_dims[1], data_block_dims[2],
                              direction, mpi_dims, cart_comm);
    }

    // speed test
    // barrier
    MPI_Barrier(cart_comm);
    double time = MPI_Wtime();
    edt_init_mpi(boundary, output, 1, data_block_dims[2], data_block_dims[1], data_block_dims[0], mpi_coords);

    if (1) {
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_and_sign_core_mpi(output, sign_map, output_stride[direction], 3, direction,
                                      data_block_dims[direction], data_block_dims[x_dir], data_block_dims[y_dir],
                                      output_stride[x_dir], output_stride[y_dir], i, j, ff, local_sign_buffer,
                                      f_send_buffer, sign_send_buffer, mpi_rank, size, mpi_coords, data_block_dims[0],
                                      data_block_dims[1], data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        MPI_Barrier(cart_comm);
        // printf("rank %d finished dim 0\n", mpi_rank);
    }

    // dim1
    direction = 1;
    if (1) {
        x_dir = (direction + 1) % 3;
        y_dir = (direction + 2) % 3;
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_and_sign_core_mpi(output, sign_map, output_stride[direction], 3, direction,
                                      data_block_dims[direction], data_block_dims[x_dir], data_block_dims[y_dir],
                                      output_stride[x_dir], output_stride[y_dir], i, j, ff, local_sign_buffer,
                                      f_send_buffer, sign_send_buffer, mpi_rank, size, mpi_coords, data_block_dims[0],
                                      data_block_dims[1], data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        MPI_Barrier(cart_comm);
        // printf("rank %d finished dim 1\n", mpi_rank);
    }

    // dim 2
    if (1) {
        direction = 2;
        x_dir = (direction + 1) % 3;
        y_dir = (direction + 2) % 3;
        for (int i = 0; i < data_block_dims[x_dir]; i++)  // y
        {
            for (int j = 0; j < data_block_dims[y_dir]; j++)  // x
            {
                edt_and_sign_core_mpi(output, sign_map, output_stride[direction], 3, direction,
                                      data_block_dims[direction], data_block_dims[x_dir], data_block_dims[y_dir],
                                      output_stride[x_dir], output_stride[y_dir], i, j, ff, local_sign_buffer,
                                      f_send_buffer, sign_send_buffer, mpi_rank, size, mpi_coords, data_block_dims[0],
                                      data_block_dims[1], data_block_dims[2], direction, mpi_dims, cart_comm);
            }
        }
        MPI_Barrier(cart_comm);
        // printf("rank %d finished dim 2\n", mpi_rank);
    }

    // calculate distance
    // calculate_distance(output, distance, data_block_dims[2], data_block_dims[1], data_block_dims[0], mpi_coords);
    calculate_distance_and_index(output, distance, indexes, data_block_dims[2], data_block_dims[1], data_block_dims[0],
                                 global_dims[2], global_dims[1], global_dims[0], mpi_coords);

    // std::string dir_prefix = "/scratch/pji228/useful/direct_quantize/mpi/blocks/";
    // char distance_filename[100];
    // sprintf(distance_filename, "%s/index_%d_%d_%d.i32", dir_prefix.c_str(), mpi_coords[0],
    // mpi_coords[1], mpi_coords[2]);
    // writefile<int>(distance_filename, output_data.data(), output_data.size());

    // free(g);
    free(f);
    free(ff);
    free(local_sign_buffer);
    free(f_send_buffer);
    free(sign_send_buffer);
}

#endif  // EDT_MPI