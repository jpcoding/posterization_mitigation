#ifndef MPI_DATA_EXCHANGE_HPP
#define MPI_DATA_EXCHANGE_HPP
#include <cstddef>
#include <vector>

#include "mpi.h"
#include "mpi/mpi_datatype.hpp"

template <typename T>
void data_exchange3d(T* src, int* src_dims, size_t* src_strides, T* dest, int* dest_dims, size_t* dest_strides,
                     int* mpi_coords, int* mpi_dims, MPI_Comm& cart_comm) {
    // use mpi_win to createa global memory to get the boundayer points

    // use ghost elements for the boundary points
    // need to pass the boundary points to the neighboring blocks
    // check if the block is at the boundary of the global domain to decide which direction to expand the dimension
    // copy the data to the new array
    size_t w_offset;
    size_t orig_idx;
    int mpi_rank;
    MPI_Datatype mpi_type = mpi_get_type<T>();
    // MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(T), &mpi_type);
    MPI_Comm_rank(cart_comm, &mpi_rank);
    // if(mpi_rank == 21){
    //     printf("MPI type size: %d, mpitye %d , mpi_int %d \n", sizeof(T), mpi_type, MPI_INT);
    // }

    for (int i = 0; i < src_dims[0]; i++) {
        int w_i = mpi_coords[0] == 0 ? i : i + 1;
        for (int j = 0; j < src_dims[1]; j++) {
            int w_j = mpi_coords[1] == 0 ? j : j + 1;
            for (int k = 0; k < src_dims[2]; k++) {
                int w_k = mpi_coords[2] == 0 ? k : k + 1;
                w_offset = w_i * dest_strides[0] + w_j * dest_strides[1] + w_k * dest_strides[2];
                orig_idx = i * src_strides[0] + j * src_strides[1] + k * src_strides[2];
                dest[w_offset] = src[orig_idx];
            }
        }
    }
    if (1) {
        // 1. deal with the 6 faces
        for (int i = 0; i < 3; i++) {
            int face_idx1 = (i + 1) % 3;
            int face_idx2 = (i + 2) % 3;
            int send_buffer_size = src_dims[face_idx1] * src_dims[face_idx2];
            std::vector<T> send_buffer_vector = std::vector<T>(send_buffer_size, 0);  // send buffer
            std::vector<T> recv_buffer_vector = std::vector<T>(send_buffer_size, 0);  // receive buffer
            T* send_buffer = send_buffer_vector.data();
            T* recv_buffer = recv_buffer_vector.data();
            int recv_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
            int receiver_rank;
            int sender_rank;
            MPI_Request req;
            MPI_Request request_send, request_recv;
            MPI_Status status;
            // printf("send buffer \n");
            //  if(mpi_rank ==0 ) printf("coords %d, %d %d \n ",mpi_coords[0], mpi_coords[1], mpi_coords[2]);
            if (1) {
                if (mpi_coords[i] != mpi_dims[i] - 1) {
                    // pass the block_dim[i] - 1 face to mpi_coords[i] + 1
                    T* quant_ints_start = src + (src_dims[i] - 1) * src_strides[i];
                    // printf("start  copy to buffer\n");

                    for (int j = 0; j < src_dims[face_idx1]; j++) {
                        for (int k = 0; k < src_dims[face_idx2]; k++) {
                            send_buffer[j * src_dims[face_idx2] + k] =
                                quant_ints_start[j * src_strides[face_idx1] + k * src_strides[face_idx2]];
                            // send_buffer[j * src_dims[face_idx2] + k] = -1;
                        }
                    }
                    // printf("complete copy to buffer\n");
                    recv_coords[i] = mpi_coords[i] + 1;
                    MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                    // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
                    // MPI_Send(send_buffer, send_buffer_size, mpi_type, receiver_rank, 0, cart_comm);
                    MPI_Sendrecv(send_buffer, send_buffer_size, mpi_type, receiver_rank, 0, recv_buffer,
                                 send_buffer_size, mpi_type, receiver_rank, 0, cart_comm, &status);
                    // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
                    {
                        int i_start = mpi_coords[i] == 0 ? src_dims[i] : src_dims[i] + 1;
                        int j_start = mpi_coords[face_idx1] == 0 ? 0 : 1;
                        int k_start = mpi_coords[face_idx2] == 0 ? 0 : 1;
                        T* w_quant_inds_start = dest + (i_start)*dest_strides[i] + (j_start)*dest_strides[face_idx1] +
                                                (k_start)*dest_strides[face_idx2];
                        // if (mpi_rank == 21) {
                        //     printf("i_start: %d, j_start: %d, k_start: %d\n", i_start, j_start, k_start);
                        // }
                        for (int j = 0; j < src_dims[face_idx1]; j++) {
                            for (int k = 0; k < src_dims[face_idx2]; k++) {
                                w_quant_inds_start[j * dest_strides[face_idx1] + k * dest_strides[face_idx2]] =
                                    recv_buffer[j * src_dims[face_idx2] + k];
                            }
                        }
                    }
                }
                if (mpi_coords[i] != 0) {
                    // pass the 0 face to mpi_coords[i] - 1
                    for (int j = 0; j < src_dims[face_idx1]; j++) {
                        for (int k = 0; k < src_dims[face_idx2]; k++) {
                            send_buffer[j * src_dims[face_idx2] + k] =
                                src[j * src_strides[face_idx1] + k * src_strides[face_idx2]];
                            // send_buffer[j * src_dims[face_idx2] + k] = -1;
                        }
                    }
                    recv_coords[i] = mpi_coords[i] - 1;
                    MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                    // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
                    // MPI_Send(send_buffer, send_buffer_size, mpi_type, receiver_rank, 0, cart_comm);
                    MPI_Sendrecv(send_buffer, send_buffer_size, mpi_type, receiver_rank, 0, recv_buffer,
                                 send_buffer_size, mpi_type, receiver_rank, 0, cart_comm, &status);
                    // printf("Rank %d, send to %d\n", mpi_rank, receiver_rank);
                    {
                        int i_start = 0;
                        int j_start = mpi_coords[face_idx1] == 0 ? 0 : 1;
                        int k_start = mpi_coords[face_idx2] == 0 ? 0 : 1;
                        T* w_quant_inds_start = dest + (i_start)*dest_strides[i] + (j_start)*dest_strides[face_idx1] +
                                                (k_start)*dest_strides[face_idx2];
                        for (int j = 0; j < src_dims[face_idx1]; j++) {
                            for (int k = 0; k < src_dims[face_idx2]; k++) {
                                w_quant_inds_start[j * dest_strides[face_idx1] + k * dest_strides[face_idx2]] =
                                    recv_buffer[j * src_dims[face_idx2] + k];
                            }
                        }
                    }
                }
            }
        }
        // 2. deal with the 12 edges
        // edges are send from mpi_block {i,j,k} to {i-1, j-1,k}
        if (1) {
            // this edge is aling dim_idx1
            // thie edge is normal to plane dim_idx2 and dim_idx3
            for (int dim_idx1 = 0; dim_idx1 < 3; dim_idx1++) {
                // send
                int dim_idx2 = (dim_idx1 + 1) % 3;
                int dim_idx3 = (dim_idx1 + 2) % 3;
                int buffer_size = src_dims[dim_idx1];  // the size of the buffer to be sent
                std::vector<T> buffer(buffer_size, 0);
                std::vector<T> recv_buffer(buffer_size, 0);
                int recv_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
                int receiver_rank;
                MPI_Status status;
                {
                    // depends on the current block's position, we need to send the edge to the corresponding block
                    // top left
                    if (mpi_coords[dim_idx2] != 0 && mpi_coords[dim_idx3] != 0) {
                        // send the edge to the block with mpi_coords[dim_idx2] - 1, mpi_coords[dim_idx3] - 1
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            buffer[i] = src[i * src_strides[dim_idx1]];
                        }
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] - 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] - 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        // MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);

                        // MPI_Recv(recv_buffer.data(), recv_buffer.size(), mpi_type, source_rank, 0, cart_comm,
                        // &status);

                        MPI_Sendrecv(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, recv_buffer.data(),
                                     recv_buffer.size(), mpi_type, receiver_rank, 0, cart_comm, &status);
                        // update the working block
                        {
                            int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                            int j_start = 0;
                            int k_start = 0;
                            T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                    (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                            for (int i = 0; i < src_dims[dim_idx1]; i++) {
                                w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                            }
                        }
                    }
                    // top right
                    if (mpi_coords[dim_idx2] != 0 && mpi_coords[dim_idx3] != mpi_dims[dim_idx3] - 1) {
                        // send the edge to the block with mpi_coords[dim_idx2] - 1, mpi_coords[dim_idx3] + 1
                        T* quant_ints_start = src + (src_dims[dim_idx3] - 1) * src_strides[dim_idx3];
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            buffer[i] = quant_ints_start[i * src_strides[dim_idx1]];
                        }
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] - 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] + 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        // MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                        MPI_Sendrecv(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, recv_buffer.data(),
                                     recv_buffer.size(), mpi_type, receiver_rank, 0, cart_comm, &status);
                        {
                            int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                            int j_start = 0;
                            int k_start = dest_dims[dim_idx3] - 1;
                            T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                    (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                            for (int i = 0; i < src_dims[dim_idx1]; i++) {
                                w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                            }
                        }
                    }
                    // bottom left
                    if (mpi_coords[dim_idx2] != mpi_dims[dim_idx2] - 1 && mpi_coords[dim_idx3] != 0) {
                        // send the edge to the block with mpi_coords[dim_idx2] + 1, mpi_coords[dim_idx3] - 1
                        T* quant_ints_start = src + (src_dims[dim_idx2] - 1) * src_strides[dim_idx2];
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            buffer[i] = quant_ints_start[i * src_strides[dim_idx1]];
                        }
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] + 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] - 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        // MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                        MPI_Sendrecv(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, recv_buffer.data(),
                                     recv_buffer.size(), mpi_type, receiver_rank, 0, cart_comm, &status);
                        {
                            int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                            int j_start = dest_dims[dim_idx2] - 1;
                            int k_start = 0;
                            T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                    (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                            for (int i = 0; i < src_dims[dim_idx1]; i++) {
                                w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                            }
                        }
                    }
                    // bottom right
                    if (mpi_coords[dim_idx2] != mpi_dims[dim_idx2] - 1 &&
                        mpi_coords[dim_idx3] != mpi_dims[dim_idx3] - 1) {
                        // send the edge to the block with mpi_coords[dim_idx2] + 1, mpi_coords[dim_idx3] + 1
                        T* quant_ints_start = src + (src_dims[dim_idx2] - 1) * src_strides[dim_idx2] +
                                              (src_dims[dim_idx3] - 1) * src_strides[dim_idx3];
                        for (int i = 0; i < src_dims[dim_idx1]; i++) {
                            buffer[i] = quant_ints_start[i * src_strides[dim_idx1]];
                        }
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] + 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] + 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        // MPI_Send(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, cart_comm);
                        MPI_Sendrecv(buffer.data(), buffer.size(), mpi_type, receiver_rank, 0, recv_buffer.data(),
                                     recv_buffer.size(), mpi_type, receiver_rank, 0, cart_comm, &status);
                        {
                            int i_start = mpi_coords[dim_idx1] == 0 ? 0 : 1;
                            int j_start = dest_dims[dim_idx2] - 1;
                            int k_start = dest_dims[dim_idx3] - 1;
                            T* w_quant_inds_start = dest + (i_start)*dest_strides[dim_idx1] +
                                                    (j_start)*dest_strides[dim_idx2] + (k_start)*dest_strides[dim_idx3];
                            for (int i = 0; i < src_dims[dim_idx1]; i++) {
                                w_quant_inds_start[i * dest_strides[dim_idx1]] = recv_buffer[i];
                            }
                        }
                    }
                }
            }
        }
        // 3. deal with the 8 corners
        if (1) {
            // 8 corners
            // top left fron
            // send
            {
                MPI_Status status;
                int target_mpi_coords[8][3] = {{mpi_coords[0] - 1, mpi_coords[1] - 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] - 1, mpi_coords[1] - 1, mpi_coords[2] + 1},
                                               {mpi_coords[0] - 1, mpi_coords[1] + 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] - 1, mpi_coords[1] + 1, mpi_coords[2] + 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] - 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] - 1, mpi_coords[2] + 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] + 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] + 1, mpi_coords[2] + 1}};
                int send_index[8][3] = {{0, 0, 0},
                                        {0, 0, src_dims[2] - 1},
                                        {0, src_dims[1] - 1, 0},
                                        {0, src_dims[1] - 1, src_dims[2] - 1},
                                        {src_dims[0] - 1, 0, 0},
                                        {src_dims[0] - 1, 0, src_dims[2] - 1},
                                        {src_dims[0] - 1, src_dims[1] - 1, 0},
                                        {src_dims[0] - 1, src_dims[1] - 1, src_dims[2] - 1}};

                int receive_index[8][3] = {{0, 0, 0},
                                           {0, 0, dest_dims[2] - 1},
                                           {0, dest_dims[1] - 1, 0},
                                           {0, dest_dims[1] - 1, dest_dims[2] - 1},
                                           {dest_dims[0] - 1, 0, 0},
                                           {dest_dims[0] - 1, 0, dest_dims[2] - 1},
                                           {dest_dims[0] - 1, dest_dims[1] - 1, 0},
                                           {dest_dims[0] - 1, dest_dims[1] - 1, dest_dims[2] - 1}};
                for (int i = 0; i < 8; i++) {
                    int target_rank;
                    int cur_coords[3] = {target_mpi_coords[i][0], target_mpi_coords[i][1], target_mpi_coords[i][2]};
                    bool valid_coords = true;
                    for (int j = 0; j < 3; j++) {
                        if (cur_coords[j] < 0 || cur_coords[j] >= mpi_dims[j]) {
                            valid_coords = false;
                            break;
                        }
                    }
                    if (valid_coords) {
                        int res = MPI_Cart_rank(cart_comm, cur_coords, &target_rank);
                        size_t idx = send_index[i][0] * src_strides[0] + send_index[i][1] * src_strides[1] +
                                     send_index[i][2] * src_strides[2];
                        // MPI_Send(&src[idx], 1, mpi_type, target_rank, 0, cart_comm);
                        size_t idx2 = receive_index[i][0] * dest_strides[0] + receive_index[i][1] * dest_strides[1] +
                                      receive_index[i][2] * dest_strides[2];
                        MPI_Sendrecv(&src[idx], 1, mpi_type, target_rank, 0, &dest[idx2], 1, mpi_type, target_rank, 0,
                                     cart_comm, &status);
                    }
                }
            }
        }
    }
}

template <typename T>
void data_exchange3d_extended(T* src, int* src_dims, size_t* src_strides, T* dest, int* dest_dims, size_t* dest_strides,
                              int extend_size, int* mpi_coords, int* mpi_dims, MPI_Comm& cart_comm) {
    // use mpi_win to createa global memory to get the boundayer points

    // use ghost elements for the boundary points
    // need to pass the boundary points to the neighboring blocks
    // check if the block is at the boundary of the global domain to decide which direction to expand the dimension
    // copy the data to the new array
    size_t w_offset;
    size_t orig_idx;
    int mpi_rank;
    MPI_Datatype mpi_type = mpi_get_type<T>();
    // MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(T), &mpi_type);
    MPI_Comm_rank(cart_comm, &mpi_rank);
    int ndims = 3;

    for (int i = 0; i < src_dims[0]; i++) {
        int w_i = mpi_coords[0] == 0 ? i : i + extend_size;
        for (int j = 0; j < src_dims[1]; j++) {
            int w_j = mpi_coords[1] == 0 ? j : j + extend_size;
            for (int k = 0; k < src_dims[2]; k++) {
                int w_k = mpi_coords[2] == 0 ? k : k + extend_size;
                w_offset = w_i * dest_strides[0] + w_j * dest_strides[1] + w_k * dest_strides[2];
                orig_idx = i * src_strides[0] + j * src_strides[1] + k * src_strides[2];
                dest[w_offset] = src[orig_idx];
            }
        }
    }
    if (1) {
        // 1. deal with the 6 faces i
        for (int i = 0; i < 3; i++) {  // i is the direction normal to the face
            int face_idx1 = (i + 1) % 3;
            int face_idx2 = (i + 2) % 3;
            int send_buffer_size = src_dims[face_idx1] * src_dims[face_idx2] * extend_size;
            int subsizes[3] = {src_dims[0], src_dims[1], src_dims[2]};
            subsizes[i] = extend_size;
            size_t subarray_size = (size_t)src_dims[face_idx1] * src_dims[face_idx2] * extend_size;
            int recv_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
            int receiver_rank;
            int sender_rank;
            MPI_Status status;
            MPI_Datatype subarray_send;
            MPI_Datatype subarray_recv;
            if (1) {
                if (mpi_coords[i] != mpi_dims[i] - 1) {
                    // create a view on the src array
                    recv_coords[i] = mpi_coords[i] + 1;
                    MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                    int src_starts[3] = {0, 0, 0};
                    src_starts[i] = src_dims[i] - extend_size;
                    MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                             &subarray_send);
                    MPI_Type_commit(&subarray_send);
                    int i_start = mpi_coords[i] == 0 ? src_dims[i] : src_dims[i] + extend_size;
                    int j_start = mpi_coords[face_idx1] == 0 ? 0 : extend_size;
                    int k_start = mpi_coords[face_idx2] == 0 ? 0 : extend_size;
                    int dest_starts[3] = {0, 0, 0};
                    dest_starts[i] = i_start;
                    dest_starts[face_idx1] = j_start;
                    dest_starts[face_idx2] = k_start;
                    MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                             &subarray_recv);
                    MPI_Type_commit(&subarray_recv);
                    MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, 
                                        dest, 1, subarray_recv, receiver_rank, 0,
                                 cart_comm, &status);
                    MPI_Type_free(&subarray_send);
                    MPI_Type_free(&subarray_recv);
                }
                if (mpi_coords[i] != 0) {
                    // pass the 0 face to mpi_coords[i] - 1
                    recv_coords[i] = mpi_coords[i] - 1;
                    MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                    int src_starts[3] = {0, 0, 0};
                    src_starts[i] = 0;
                    MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                             &subarray_send);
                    MPI_Type_commit(&subarray_send);
                    int i_start = 0;
                    int j_start = mpi_coords[face_idx1] == 0 ? 0 : extend_size;
                    int k_start = mpi_coords[face_idx2] == 0 ? 0 : extend_size;
                    int dest_starts[3] = {0, 0, 0};
                    dest_starts[i] = i_start;
                    dest_starts[face_idx1] = j_start;
                    dest_starts[face_idx2] = k_start;
                    MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                             &subarray_recv);
                    MPI_Type_commit(&subarray_recv);
                    MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, dest, 1, subarray_recv, receiver_rank, 0,
                                 cart_comm, &status);
                    MPI_Type_free(&subarray_send);
                    MPI_Type_free(&subarray_recv);
                }
            }
        }
        // 2. deal with the 12 edges
        // edges are send from mpi_block {i,j,k} to {i-1, j-1,k}
        if (1) {
            // this edge is aling dim_idx1
            // thie edge is normal to plane dim_idx2 and dim_idx3
            for (int dim_idx1 = 0; dim_idx1 < 3; dim_idx1++) {
                // send
                int dim_idx2 = (dim_idx1 + 1) % 3;
                int dim_idx3 = (dim_idx1 + 2) % 3;
                int buffer_size = src_dims[dim_idx1] * extend_size * extend_size;  // the size of the buffer to be sent
                int subsizes[3] = {src_dims[0], src_dims[1], src_dims[2]};
                subsizes[dim_idx2] = extend_size;
                subsizes[dim_idx3] = extend_size;
                int recv_coords[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
                int receiver_rank;
                MPI_Status status;
                MPI_Datatype subarray_send;
                MPI_Datatype subarray_recv;
                {
                    // depends on the current block's position, we need to send the edge to the corresponding block
                    // top left
                    if (mpi_coords[dim_idx2] != 0 && mpi_coords[dim_idx3] != 0) {
                        // send the edge to the block with mpi_coords[dim_idx2] - 1, mpi_coords[dim_idx3] - 1
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] - 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] - 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        int src_starts[3] = {0, 0, 0};
                        src_starts[dim_idx1] = 0;
                        MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_send);
                        MPI_Type_commit(&subarray_send);
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : extend_size;
                        int j_start = 0;
                        int k_start = 0;
                        int dest_starts[3] = {0, 0, 0};
                        dest_starts[dim_idx1] = i_start;
                        dest_starts[dim_idx2] = j_start;
                        dest_starts[dim_idx3] = k_start;
                        MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_recv);
                        MPI_Type_commit(&subarray_recv);
                        MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, dest, 1, subarray_recv, receiver_rank, 0,
                                     cart_comm, &status);
                        MPI_Type_free(&subarray_send);
                        MPI_Type_free(&subarray_recv);
                    }
                    // top right
                    if (mpi_coords[dim_idx2] != 0 && mpi_coords[dim_idx3] != mpi_dims[dim_idx3] - 1) {
                        // send the edge to the block with mpi_coords[dim_idx2] - 1, mpi_coords[dim_idx3] + 1
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] - 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] + 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        int src_starts[3] = {0, 0, 0};
                        src_starts[dim_idx3] = src_dims[dim_idx3] - extend_size;
                        MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_send);
                        MPI_Type_commit(&subarray_send);
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : extend_size;
                        int j_start = 0;
                        int k_start = dest_dims[dim_idx3] - extend_size;
                        int dest_starts[3] = {0, 0, 0};
                        dest_starts[dim_idx1] = i_start;
                        dest_starts[dim_idx2] = j_start;
                        dest_starts[dim_idx3] = k_start;
                        MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_recv);
                        MPI_Type_commit(&subarray_recv);
                        MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, dest, 1, subarray_recv, receiver_rank, 0,
                                     cart_comm, &status);
                        MPI_Type_free(&subarray_send);
                        MPI_Type_free(&subarray_recv);
                    }
                    // bottom left
                    if (mpi_coords[dim_idx2] != mpi_dims[dim_idx2] - 1 && mpi_coords[dim_idx3] != 0) {
                        // send the edge to the block with mpi_coords[dim_idx2] + 1, mpi_coords[dim_idx3] - 1
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] + 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] - 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        int src_starts[3] = {0, 0, 0};
                        src_starts[dim_idx2] = src_dims[dim_idx2] - extend_size;
                        MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_send);
                        MPI_Type_commit(&subarray_send);
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : extend_size;
                        int j_start = dest_dims[dim_idx2] - extend_size;
                        int k_start = 0;
                        int dest_starts[3] = {0, 0, 0};
                        dest_starts[dim_idx1] = i_start;
                        dest_starts[dim_idx2] = j_start;
                        dest_starts[dim_idx3] = k_start;
                        MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_recv);
                        MPI_Type_commit(&subarray_recv);
                        MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, dest, 1, subarray_recv, receiver_rank, 0,
                                     cart_comm, &status);
                        MPI_Type_free(&subarray_send);
                        MPI_Type_free(&subarray_recv);
                    }
                    // bottom right
                    if (mpi_coords[dim_idx2] != mpi_dims[dim_idx2] - 1 &&
                        mpi_coords[dim_idx3] != mpi_dims[dim_idx3] - 1) {
                        // send the edge to the block with mpi_coords[dim_idx2] + 1, mpi_coords[dim_idx3] + 1
                        recv_coords[dim_idx2] = mpi_coords[dim_idx2] + 1;
                        recv_coords[dim_idx3] = mpi_coords[dim_idx3] + 1;
                        MPI_Cart_rank(cart_comm, recv_coords, &receiver_rank);
                        int src_starts[3] = {0, 0, 0};
                        src_starts[dim_idx2] = src_dims[dim_idx2] - extend_size;
                        src_starts[dim_idx3] = src_dims[dim_idx3] - extend_size;
                        MPI_Type_create_subarray(ndims, src_dims, subsizes, src_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_send);

                        MPI_Type_commit(&subarray_send);
                        int i_start = mpi_coords[dim_idx1] == 0 ? 0 : extend_size;
                        int j_start = dest_dims[dim_idx2] - extend_size;
                        int k_start = dest_dims[dim_idx3] - extend_size;
                        int dest_starts[3] = {0, 0, 0};
                        dest_starts[dim_idx1] = i_start;
                        dest_starts[dim_idx2] = j_start;
                        dest_starts[dim_idx3] = k_start;
                        MPI_Type_create_subarray(ndims, dest_dims, subsizes, dest_starts, MPI_ORDER_C, mpi_type,
                                                 &subarray_recv);
                        MPI_Type_commit(&subarray_recv);
                        MPI_Sendrecv(src, 1, subarray_send, receiver_rank, 0, dest, 1, subarray_recv, receiver_rank, 0,
                                     cart_comm, &status);
                        MPI_Type_free(&subarray_send);
                        MPI_Type_free(&subarray_recv);
                    }
                }
            }
        }
        // 3. deal with the 8 corners
        if (1) {
            // 8 corners
            // top left fron
            // send
            {
                int target_mpi_coords[8][3] = {{mpi_coords[0] - 1, mpi_coords[1] - 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] - 1, mpi_coords[1] - 1, mpi_coords[2] + 1},
                                               {mpi_coords[0] - 1, mpi_coords[1] + 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] - 1, mpi_coords[1] + 1, mpi_coords[2] + 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] - 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] - 1, mpi_coords[2] + 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] + 1, mpi_coords[2] - 1},
                                               {mpi_coords[0] + 1, mpi_coords[1] + 1, mpi_coords[2] + 1}};
                int send_index[8][3] = {
                    {0, 0, 0},
                    {0, 0, src_dims[2] - extend_size},
                    {0, src_dims[1] - extend_size, 0},
                    {0, src_dims[1] - extend_size, src_dims[2] - extend_size},
                    {src_dims[0] - extend_size, 0, 0},
                    {src_dims[0] - extend_size, 0, src_dims[2] - extend_size},
                    {src_dims[0] - extend_size, src_dims[1] - extend_size, 0},
                    {src_dims[0] - extend_size, src_dims[1] - extend_size, src_dims[2] - extend_size}};
                int receive_index[8][3] = {
                    {0, 0, 0},
                    {0, 0, dest_dims[2] - extend_size},
                    {0, dest_dims[1] - extend_size, 0},
                    {0, dest_dims[1] - extend_size, dest_dims[2] - extend_size},
                    {dest_dims[0] - extend_size, 0, 0},
                    {dest_dims[0] - extend_size, 0, dest_dims[2] - extend_size},
                    {dest_dims[0] - extend_size, dest_dims[1] - extend_size, 0},
                    {dest_dims[0] - extend_size, dest_dims[1] - extend_size, dest_dims[2] - extend_size}};
                int subsizes[3] = {extend_size, extend_size, extend_size};
                MPI_Datatype subarray_send;
                MPI_Datatype subarray_recv;
                MPI_Status status;
                for (int i = 0; i < 8; i++) {
                    int target_rank;
                    int cur_coords[3] = {target_mpi_coords[i][0], target_mpi_coords[i][1], target_mpi_coords[i][2]};
                    bool valid_coords = true;
                    for (int j = 0; j < 3; j++) {
                        if (cur_coords[j] < 0 || cur_coords[j] >= mpi_dims[j]) {
                            valid_coords = false;
                            break;
                        }
                    }
                    if (valid_coords) {
                        int res = MPI_Cart_rank(cart_comm, cur_coords, &target_rank);
                        MPI_Type_create_subarray(ndims, src_dims, subsizes, send_index[i], MPI_ORDER_C, mpi_type,
                                                 &subarray_send);
                        MPI_Type_commit(&subarray_send);
                        MPI_Type_create_subarray(ndims, dest_dims, subsizes, receive_index[i], MPI_ORDER_C, mpi_type,
                                                 &subarray_recv);
                        MPI_Type_commit(&subarray_recv);
                        MPI_Sendrecv(src, 1, subarray_send, target_rank, 0, dest, 1, subarray_recv, target_rank,
                                     0, cart_comm, &status);
                        MPI_Type_free(&subarray_send);
                        MPI_Type_free(&subarray_recv);
                    }
                }
            }
        }
    }
}

// Non-blocking version of data_exchange3d_extended.
// Posts all 26 Isend/Irecv pairs simultaneously (faces, edges, corners), then
// waits with a single MPI_Waitall.  The destination sub-regions written by face,
// edge, and corner receives are disjoint, so concurrent posting is safe.
template <typename T>
void data_exchange3d_extended_nb(T* src, int* src_dims, size_t* src_strides, T* dest, int* dest_dims,
                                  size_t* dest_strides, int extend_size, int* mpi_coords, int* mpi_dims,
                                  MPI_Comm& cart_comm) {
    MPI_Datatype mpi_type = mpi_get_type<T>();
    int ndims = 3;

    // Copy local data into dest at the appropriate ghost offset.
    for (int i = 0; i < src_dims[0]; i++) {
        int w_i = mpi_coords[0] == 0 ? i : i + extend_size;
        for (int j = 0; j < src_dims[1]; j++) {
            int w_j = mpi_coords[1] == 0 ? j : j + extend_size;
            for (int k = 0; k < src_dims[2]; k++) {
                int w_k = mpi_coords[2] == 0 ? k : k + extend_size;
                dest[w_i * dest_strides[0] + w_j * dest_strides[1] + w_k * dest_strides[2]] =
                    src[i * src_strides[0] + j * src_strides[1] + k * src_strides[2]];
            }
        }
    }

    std::vector<MPI_Request> requests;
    std::vector<MPI_Datatype> types;

    // Post one Isend + Irecv pair using subarray derived types.
    // Derived types are stored in `types` and freed after MPI_Waitall.
    auto post = [&](int target_rank, int* ss, int* ds, int* subsizes) {
        MPI_Datatype st, rt;
        MPI_Type_create_subarray(ndims, src_dims, subsizes, ss, MPI_ORDER_C, mpi_type, &st);
        MPI_Type_commit(&st);
        MPI_Type_create_subarray(ndims, dest_dims, subsizes, ds, MPI_ORDER_C, mpi_type, &rt);
        MPI_Type_commit(&rt);
        types.push_back(st);
        types.push_back(rt);
        MPI_Request rs, rr;
        MPI_Isend(src,  1, st, target_rank, 0, cart_comm, &rs);
        MPI_Irecv(dest, 1, rt, target_rank, 0, cart_comm, &rr);
        requests.push_back(rs);
        requests.push_back(rr);
    };

    // --- 6 faces ---
    for (int i = 0; i < 3; i++) {
        int fi1 = (i + 1) % 3, fi2 = (i + 2) % 3;
        int subsizes[3] = {src_dims[0], src_dims[1], src_dims[2]};
        subsizes[i] = extend_size;

        if (mpi_coords[i] != mpi_dims[i] - 1) {
            int rc[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
            rc[i] = mpi_coords[i] + 1;
            int target_rank; MPI_Cart_rank(cart_comm, rc, &target_rank);
            int ss[3] = {0, 0, 0}; ss[i] = src_dims[i] - extend_size;
            int ds[3] = {0, 0, 0};
            ds[i]   = mpi_coords[i]   == 0 ? src_dims[i]   : src_dims[i]   + extend_size;
            ds[fi1] = mpi_coords[fi1] == 0 ? 0             : extend_size;
            ds[fi2] = mpi_coords[fi2] == 0 ? 0             : extend_size;
            post(target_rank, ss, ds, subsizes);
        }
        if (mpi_coords[i] != 0) {
            int rc[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
            rc[i] = mpi_coords[i] - 1;
            int target_rank; MPI_Cart_rank(cart_comm, rc, &target_rank);
            int ss[3] = {0, 0, 0};
            int ds[3] = {0, 0, 0};
            ds[fi1] = mpi_coords[fi1] == 0 ? 0 : extend_size;
            ds[fi2] = mpi_coords[fi2] == 0 ? 0 : extend_size;
            post(target_rank, ss, ds, subsizes);
        }
    }

    // --- 12 edges ---
    for (int d1 = 0; d1 < 3; d1++) {
        int d2 = (d1 + 1) % 3, d3 = (d1 + 2) % 3;
        int subsizes[3] = {src_dims[0], src_dims[1], src_dims[2]};
        subsizes[d2] = extend_size;
        subsizes[d3] = extend_size;

        auto post_edge = [&](int dv2, int dv3, int ss2, int ss3, int ds2, int ds3) {
            int rc[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
            rc[d2] = mpi_coords[d2] + dv2;
            rc[d3] = mpi_coords[d3] + dv3;
            int target_rank; MPI_Cart_rank(cart_comm, rc, &target_rank);
            int ss[3] = {0, 0, 0}; ss[d2] = ss2; ss[d3] = ss3;
            int ds[3] = {0, 0, 0};
            ds[d1] = mpi_coords[d1] == 0 ? 0 : extend_size;
            ds[d2] = ds2; ds[d3] = ds3;
            post(target_rank, ss, ds, subsizes);
        };

        if (mpi_coords[d2] != 0              && mpi_coords[d3] != 0)
            post_edge(-1, -1, 0, 0, 0, 0);
        if (mpi_coords[d2] != 0              && mpi_coords[d3] != mpi_dims[d3] - 1)
            post_edge(-1, +1, 0, src_dims[d3] - extend_size, 0, dest_dims[d3] - extend_size);
        if (mpi_coords[d2] != mpi_dims[d2]-1 && mpi_coords[d3] != 0)
            post_edge(+1, -1, src_dims[d2] - extend_size, 0, dest_dims[d2] - extend_size, 0);
        if (mpi_coords[d2] != mpi_dims[d2]-1 && mpi_coords[d3] != mpi_dims[d3] - 1)
            post_edge(+1, +1, src_dims[d2] - extend_size, src_dims[d3] - extend_size,
                               dest_dims[d2] - extend_size, dest_dims[d3] - extend_size);
    }

    // --- 8 corners ---
    {
        int target_coords[8][3] = {
            {mpi_coords[0]-1, mpi_coords[1]-1, mpi_coords[2]-1},
            {mpi_coords[0]-1, mpi_coords[1]-1, mpi_coords[2]+1},
            {mpi_coords[0]-1, mpi_coords[1]+1, mpi_coords[2]-1},
            {mpi_coords[0]-1, mpi_coords[1]+1, mpi_coords[2]+1},
            {mpi_coords[0]+1, mpi_coords[1]-1, mpi_coords[2]-1},
            {mpi_coords[0]+1, mpi_coords[1]-1, mpi_coords[2]+1},
            {mpi_coords[0]+1, mpi_coords[1]+1, mpi_coords[2]-1},
            {mpi_coords[0]+1, mpi_coords[1]+1, mpi_coords[2]+1},
        };
        int send_idx[8][3] = {
            {0,                          0,                          0},
            {0,                          0,                          src_dims[2] - extend_size},
            {0,                          src_dims[1] - extend_size,  0},
            {0,                          src_dims[1] - extend_size,  src_dims[2] - extend_size},
            {src_dims[0] - extend_size,  0,                          0},
            {src_dims[0] - extend_size,  0,                          src_dims[2] - extend_size},
            {src_dims[0] - extend_size,  src_dims[1] - extend_size,  0},
            {src_dims[0] - extend_size,  src_dims[1] - extend_size,  src_dims[2] - extend_size},
        };
        int recv_idx[8][3] = {
            {0,                           0,                           0},
            {0,                           0,                           dest_dims[2] - extend_size},
            {0,                           dest_dims[1] - extend_size,  0},
            {0,                           dest_dims[1] - extend_size,  dest_dims[2] - extend_size},
            {dest_dims[0] - extend_size,  0,                           0},
            {dest_dims[0] - extend_size,  0,                           dest_dims[2] - extend_size},
            {dest_dims[0] - extend_size,  dest_dims[1] - extend_size,  0},
            {dest_dims[0] - extend_size,  dest_dims[1] - extend_size,  dest_dims[2] - extend_size},
        };
        int subsizes[3] = {extend_size, extend_size, extend_size};
        for (int ci = 0; ci < 8; ci++) {
            bool valid = true;
            for (int j = 0; j < 3; j++)
                if (target_coords[ci][j] < 0 || target_coords[ci][j] >= mpi_dims[j]) { valid = false; break; }
            if (!valid) continue;
            int target_rank; MPI_Cart_rank(cart_comm, target_coords[ci], &target_rank);
            post(target_rank, send_idx[ci], recv_idx[ci], subsizes);
        }
    }

    MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    for (auto& t : types) MPI_Type_free(&t);
}

// Non-blocking version of data_exchange3d (extend_size == 1 specialisation).
template <typename T>
void data_exchange3d_nb(T* src, int* src_dims, size_t* src_strides, T* dest, int* dest_dims, size_t* dest_strides,
                        int* mpi_coords, int* mpi_dims, MPI_Comm& cart_comm) {
    data_exchange3d_extended_nb(src, src_dims, src_strides, dest, dest_dims, dest_strides,
                                1, mpi_coords, mpi_dims, cart_comm);
}

// ---------------------------------------------------------------------------
// Persistent-request context: types and requests are created once at setup
// and reused across calls.  Each exchange call costs only local-copy +
// MPI_Startall + MPI_Waitall — no type creation, no vector allocation.
//
// Usage:
//   auto ctx = make_exchange_context(src, ..., dest, ..., extend, coords, dims, comm);
//   for (each timestep) exchange(ctx);   // cheap hot path
//   free_exchange_context(ctx);
// ---------------------------------------------------------------------------

// Internal helper: call callback(target_rank, ss[3], ds[3], subsizes[3]) for
// every valid neighbor (up to 26) in the 3-D Cartesian decomposition.
template <typename Fn>
void _for_each_neighbor_3d(int* src_dims, int* dest_dims, int extend_size,
                            int* mpi_coords, int* mpi_dims, MPI_Comm& cart_comm,
                            Fn callback) {
    // --- 6 faces ---
    for (int i = 0; i < 3; i++) {
        int fi1 = (i+1)%3, fi2 = (i+2)%3;
        int subsizes[3] = {src_dims[0], src_dims[1], src_dims[2]};
        subsizes[i] = extend_size;
        if (mpi_coords[i] != mpi_dims[i]-1) {
            int rc[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]}; rc[i] = mpi_coords[i]+1;
            int tr; MPI_Cart_rank(cart_comm, rc, &tr);
            int ss[3]={0,0,0}; ss[i] = src_dims[i]-extend_size;
            int ds[3]={0,0,0};
            ds[i]   = mpi_coords[i]   == 0 ? src_dims[i]   : src_dims[i]  +extend_size;
            ds[fi1] = mpi_coords[fi1] == 0 ? 0             : extend_size;
            ds[fi2] = mpi_coords[fi2] == 0 ? 0             : extend_size;
            callback(tr, ss, ds, subsizes);
        }
        if (mpi_coords[i] != 0) {
            int rc[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]}; rc[i] = mpi_coords[i]-1;
            int tr; MPI_Cart_rank(cart_comm, rc, &tr);
            int ss[3]={0,0,0};
            int ds[3]={0,0,0};
            ds[fi1] = mpi_coords[fi1] == 0 ? 0 : extend_size;
            ds[fi2] = mpi_coords[fi2] == 0 ? 0 : extend_size;
            callback(tr, ss, ds, subsizes);
        }
    }
    // --- 12 edges ---
    for (int d1 = 0; d1 < 3; d1++) {
        int d2 = (d1+1)%3, d3 = (d1+2)%3;
        int subsizes[3] = {src_dims[0], src_dims[1], src_dims[2]};
        subsizes[d2] = extend_size; subsizes[d3] = extend_size;
        auto edge = [&](int dv2, int dv3, int ss2, int ss3, int ds2, int ds3) {
            int rc[3] = {mpi_coords[0], mpi_coords[1], mpi_coords[2]};
            rc[d2] = mpi_coords[d2]+dv2; rc[d3] = mpi_coords[d3]+dv3;
            int tr; MPI_Cart_rank(cart_comm, rc, &tr);
            int ss[3]={0,0,0}; ss[d2]=ss2; ss[d3]=ss3;
            int ds[3]={0,0,0};
            ds[d1] = mpi_coords[d1]==0 ? 0 : extend_size;
            ds[d2]=ds2; ds[d3]=ds3;
            callback(tr, ss, ds, subsizes);
        };
        if (mpi_coords[d2]!=0              && mpi_coords[d3]!=0)              edge(-1,-1,0,0,0,0);
        if (mpi_coords[d2]!=0              && mpi_coords[d3]!=mpi_dims[d3]-1) edge(-1,+1,0,src_dims[d3]-extend_size,0,dest_dims[d3]-extend_size);
        if (mpi_coords[d2]!=mpi_dims[d2]-1 && mpi_coords[d3]!=0)              edge(+1,-1,src_dims[d2]-extend_size,0,dest_dims[d2]-extend_size,0);
        if (mpi_coords[d2]!=mpi_dims[d2]-1 && mpi_coords[d3]!=mpi_dims[d3]-1) edge(+1,+1,src_dims[d2]-extend_size,src_dims[d3]-extend_size,dest_dims[d2]-extend_size,dest_dims[d3]-extend_size);
    }
    // --- 8 corners ---
    {
        int tc[8][3]={{mpi_coords[0]-1,mpi_coords[1]-1,mpi_coords[2]-1},{mpi_coords[0]-1,mpi_coords[1]-1,mpi_coords[2]+1},
                      {mpi_coords[0]-1,mpi_coords[1]+1,mpi_coords[2]-1},{mpi_coords[0]-1,mpi_coords[1]+1,mpi_coords[2]+1},
                      {mpi_coords[0]+1,mpi_coords[1]-1,mpi_coords[2]-1},{mpi_coords[0]+1,mpi_coords[1]-1,mpi_coords[2]+1},
                      {mpi_coords[0]+1,mpi_coords[1]+1,mpi_coords[2]-1},{mpi_coords[0]+1,mpi_coords[1]+1,mpi_coords[2]+1}};
        int si[8][3]={{0,0,0},{0,0,src_dims[2]-extend_size},{0,src_dims[1]-extend_size,0},{0,src_dims[1]-extend_size,src_dims[2]-extend_size},
                      {src_dims[0]-extend_size,0,0},{src_dims[0]-extend_size,0,src_dims[2]-extend_size},
                      {src_dims[0]-extend_size,src_dims[1]-extend_size,0},{src_dims[0]-extend_size,src_dims[1]-extend_size,src_dims[2]-extend_size}};
        int ri[8][3]={{0,0,0},{0,0,dest_dims[2]-extend_size},{0,dest_dims[1]-extend_size,0},{0,dest_dims[1]-extend_size,dest_dims[2]-extend_size},
                      {dest_dims[0]-extend_size,0,0},{dest_dims[0]-extend_size,0,dest_dims[2]-extend_size},
                      {dest_dims[0]-extend_size,dest_dims[1]-extend_size,0},{dest_dims[0]-extend_size,dest_dims[1]-extend_size,dest_dims[2]-extend_size}};
        int subsizes[3]={extend_size,extend_size,extend_size};
        for (int ci=0; ci<8; ci++) {
            bool valid=true;
            for (int j=0; j<3; j++) if(tc[ci][j]<0||tc[ci][j]>=mpi_dims[j]){valid=false;break;}
            if (!valid) continue;
            int tr; MPI_Cart_rank(cart_comm, tc[ci], &tr);
            callback(tr, si[ci], ri[ci], subsizes);
        }
    }
}

template <typename T>
struct ExchangeContext3D {
    T*                        src;
    T*                        dest;
    int                       src_dims[3];
    size_t                    src_strides[3];
    size_t                    dest_strides[3];
    int                       extend_size;
    int                       mpi_coords[3];
    std::vector<MPI_Datatype> types;  // committed send+recv types
    std::vector<MPI_Request>  reqs;   // persistent send+recv requests

    // Non-copyable: MPI handles must not be aliased.
    ExchangeContext3D() = default;
    ExchangeContext3D(const ExchangeContext3D&) = delete;
    ExchangeContext3D& operator=(const ExchangeContext3D&) = delete;
    ExchangeContext3D(ExchangeContext3D&&) = default;
    ExchangeContext3D& operator=(ExchangeContext3D&&) = default;
};

template <typename T>
ExchangeContext3D<T> make_exchange_context(
        T* src, int* src_dims, size_t* src_strides,
        T* dest, int* dest_dims, size_t* dest_strides,
        int extend_size, int* mpi_coords, int* mpi_dims, MPI_Comm& cart_comm) {
    ExchangeContext3D<T> ctx;
    ctx.src = src; ctx.dest = dest; ctx.extend_size = extend_size;
    for (int d = 0; d < 3; d++) {
        ctx.src_dims[d]     = src_dims[d];
        ctx.src_strides[d]  = src_strides[d];
        ctx.dest_strides[d] = dest_strides[d];
        ctx.mpi_coords[d]   = mpi_coords[d];
    }

    MPI_Datatype mpi_type = mpi_get_type<T>();
    int ndims = 3;

    _for_each_neighbor_3d(src_dims, dest_dims, extend_size, mpi_coords, mpi_dims, cart_comm,
        [&](int target_rank, int* ss, int* ds, int* subsizes) {
            MPI_Datatype st, rt;
            MPI_Type_create_subarray(ndims, src_dims,  subsizes, ss, MPI_ORDER_C, mpi_type, &st);
            MPI_Type_commit(&st);
            MPI_Type_create_subarray(ndims, dest_dims, subsizes, ds, MPI_ORDER_C, mpi_type, &rt);
            MPI_Type_commit(&rt);
            ctx.types.push_back(st);
            ctx.types.push_back(rt);
            MPI_Request rs, rr;
            MPI_Send_init(src,  1, st, target_rank, 0, cart_comm, &rs);
            MPI_Recv_init(dest, 1, rt, target_rank, 0, cart_comm, &rr);
            ctx.reqs.push_back(rs);
            ctx.reqs.push_back(rr);
        });

    return ctx;
}

// Hot path: local copy + Startall + Waitall.  No allocation, no type creation.
template <typename T>
void exchange(ExchangeContext3D<T>& ctx) {
    int* sd = ctx.src_dims;
    int  es = ctx.extend_size;
    int* mc = ctx.mpi_coords;
    for (int i = 0; i < sd[0]; i++) {
        int w_i = mc[0] == 0 ? i : i + es;
        for (int j = 0; j < sd[1]; j++) {
            int w_j = mc[1] == 0 ? j : j + es;
            for (int k = 0; k < sd[2]; k++) {
                int w_k = mc[2] == 0 ? k : k + es;
                ctx.dest[w_i*ctx.dest_strides[0] + w_j*ctx.dest_strides[1] + w_k*ctx.dest_strides[2]] =
                    ctx.src[i*ctx.src_strides[0]  + j*ctx.src_strides[1]  + k*ctx.src_strides[2]];
            }
        }
    }
    if (!ctx.reqs.empty()) {
        MPI_Startall((int)ctx.reqs.size(), ctx.reqs.data());
        MPI_Waitall((int)ctx.reqs.size(), ctx.reqs.data(), MPI_STATUSES_IGNORE);
    }
}

template <typename T>
void free_exchange_context(ExchangeContext3D<T>& ctx) {
    for (auto& r : ctx.reqs)  MPI_Request_free(&r);
    for (auto& t : ctx.types) MPI_Type_free(&t);
    ctx.reqs.clear();
    ctx.types.clear();
}

#endif  // MPI_DATA_EXCHANGE_HPP
