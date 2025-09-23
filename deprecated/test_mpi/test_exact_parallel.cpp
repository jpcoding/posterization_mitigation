#include <mpi.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include "mpi/boundary.hpp"
#include "mpi/compensation.hpp"
#include "mpi/data_exchange.hpp"
#include "mpi/edt.hpp"
#include "mpi/mpi_datatype.hpp"
#include "mpi/stats.hpp"
#include "utils/file_utils.hpp"
#include "CLI/CLI.hpp"

namespace SZ = SZ3;

int main(int argc, char** argv) {
    int mpi_rank, size;
    MPI_Comm cart_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::array<int, 3> dims_array = {0, 0, 0};  // Let MPI decide the best grid dimensions
    std::array<int, 3> orig_dims_array = {0, 0, 0};

    double eb = 0.0; 
    std::string eb_mode = "rel";  // Relative error bound mode 
    std::string dir_prefix;   // Directory prefix for the blocks
    std::string name_prefix;  // Name prefix for the blocks
    std::string out_dir; 
    bool use_rbf = false; 

    CLI::App app{"Merge files using MPI - 3D"};
    argv = app.ensure_utf8(argv);
    app.add_option("-e", eb, "Relative error bound")->required();
    app.add_option("-m", eb_mode, "Relative error bound mode")->required();
    app.add_option("--mpidims", dims_array, "mpi dimensions")->required();
    app.add_option("--dir", dir_prefix, "input file")->required();
    app.add_option("--prefix", name_prefix, "output file")->required();
    app.add_option("--origdims", orig_dims_array, "dimensions")->required();
    app.add_option("--outdir", out_dir, "output file")->required();
    app.add_option("--use_rbf", use_rbf, "use rbf")->required(); 
    CLI11_PARSE(app, argc, argv);

    int* orig_dims = orig_dims_array.data();
    int* dims = dims_array.data(); 

    
    // dims[0] = atoi(argv[1]);
    // dims[1] = atoi(argv[2]);
    // dims[2] = atoi(argv[3]);
    // rel_eb = atof(argv[4]);      // Relative error bound
    // dir_prefix = argv[5];   // Directory prefix for the blocks
    // name_prefix = argv[6];  // Name prefix for the blocks
    // orig_dims[0] = atoi(argv[7]);
    // orig_dims[1] = atoi(argv[8]);
    // orig_dims[2] = atoi(argv[9]);



    int periods[3] = {0, 0, 0};  // No periodicity in any dimension
    int coords[3] = {0, 0, 0};   // Coords of this process in the grid

    // MPI_Dims_create(size, 3, dims);
    if (mpi_rank == 0) {
        printf("Number of processes: %d\n", size);
        printf("Grid dimensions: (%d, %d, %d)\n", dims[0], dims[1], dims[2]);
        printf("use rbf: %d\n", (int) use_rbf);
    }
    // Create a 3D Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);
    // Get coordinates of this process
    MPI_Cart_coords(cart_comm, mpi_rank, 3, coords);

    // read data for each rank
    char filename[100];
    sprintf(filename, "%s/%s_%d_%d_%d.f32", dir_prefix.c_str(), name_prefix.c_str(), coords[0], coords[1], coords[2]);
    size_t num_elements = 0;
    auto data = readfile<float>(filename, num_elements);

    std::vector<float> orig_copy(data.get(), data.get() + num_elements);

    if (data == nullptr) {
        fprintf(stderr, "Error reading file %s\n", filename);
        printf("Rank %d, num_elements: %ld\n", mpi_rank, num_elements);
        MPI_Finalize();
        return 1;
    }
    int block_dims[3] = {orig_dims[0] / dims[0], orig_dims[1] / dims[1], orig_dims[2] / dims[2]};
    int block_size = block_dims[0] * block_dims[1] * block_dims[2];
    size_t block_strides[3] = {(size_t)block_dims[1] * block_dims[2], (size_t)block_dims[2], 1};
    // assert(num_elements == block_size);
    // printf("Rank %d, num_elements: %ld\n", mpi_rank, num_elements);

    // get the global max and min to get the gloibal value range
    float local_max = *std::max_element(data.get(), data.get() + num_elements);
    float local_min = *std::min_element(data.get(), data.get() + num_elements);
    float global_max, global_min;

    bool operation = true;
    double abs_eb;
    if (eb_mode == "rel") {
        float local_max = *std::max_element(data.get(), data.get() + num_elements);
        float local_min = *std::min_element(data.get(), data.get() + num_elements);
        float global_max, global_min;

        double local_range = local_max - local_min;
        if (local_range < 1e-10) {
            operation = false;
            printf("Rank %d, local range is too small: %f\n", mpi_rank, local_range); 
        }
        // barrier
        if (1) {
            MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, cart_comm);
            MPI_Reduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, 0, cart_comm);
            MPI_Bcast(&global_max, 1, MPI_FLOAT, 0, cart_comm);
            MPI_Bcast(&global_min, 1, MPI_FLOAT, 0, cart_comm);
        }
        abs_eb = eb * (global_max - global_min);
    } else {
        abs_eb = eb;
    }
    double compensation_magnitude = abs_eb * 0.9;
    // printf("Rank %d, block_size: %d\n", mpi_rank, block_size);
    auto quantizer = SZ::LinearQuantizer<float>();
    quantizer.set_eb(abs_eb);
    std::vector<int> quant_inds(block_size, 0);
    size_t local_zero_count = 0;
    MPI_Barrier(cart_comm);
    double time = MPI_Wtime();
    if (1) {
        for (int i = 0; i < block_size; i++) {
            quant_inds[i] = quantizer.quantize_and_overwrite(data[i], 0) - 32768;
            if (quant_inds[i] == 0) local_zero_count++;
        }
    }
    MPI_Barrier(cart_comm);
    time = MPI_Wtime() - time;
    size_t global_zero_count = 0;
    size_t global_size = 1;
    for (int i = 0; i < 3; i++) {
        global_size *= orig_dims[i];
    }

    MPI_Reduce(&local_zero_count, &global_zero_count, 1, mpi_get_type<size_t>(), MPI_SUM, 0, cart_comm);
    if (mpi_rank == 0) {
        printf("quantization time = %f \n", time);
        printf("zero count = %ld \n", local_zero_count);
        printf("global size = %ld \n", global_size);
        printf("global zero ratio = %f \n", static_cast<double>(global_zero_count) / static_cast<double>(global_size));
        printf("eb mode = %s \n", eb_mode.c_str());
        printf("input eb = %f \n", eb);
        printf("abs eb = %f \n", abs_eb);

    }
    double orig_psnr = -1;
    if (1) orig_psnr = get_psnr_mpi(orig_copy.data(), data.get(), num_elements, cart_comm);
    if (mpi_rank == 0) {
        printf("Original PSNR: %f\n", orig_psnr);
    }
    {
        char out_filename[100];
        sprintf(out_filename, "%s/%s_%d_%d_%d.decomp.f32", out_dir.c_str(), name_prefix.c_str(), coords[0],
                coords[1], coords[2]);
        writefile<float>(out_filename, data.get(), block_size);
    }

    int w_block_dims[3] = {0, 0, 0};
    size_t w_block_size = 1;
    for (int i = 0; i < 3; i++) {
        if (coords[i] == dims[i] - 1 || coords[i] == 0) {
            w_block_dims[i] = block_dims[i] + 1;
        } else {
            w_block_dims[i] = block_dims[i] + 2;
        }
        w_block_size *= w_block_dims[i];
    }
    size_t w_block_strides[3] = {(size_t)w_block_dims[1] * w_block_dims[2], (size_t)w_block_dims[2], 1};
    std::vector<int> w_quant_inds(w_block_size, 0);

    // quantization index exchange
    MPI_Barrier(cart_comm);
    time = MPI_Wtime();
    double time_exchnage1 = MPI_Wtime();
    if (1)
        data_exhange3d(quant_inds.data(), block_dims, block_strides, w_quant_inds.data(), w_block_dims, w_block_strides,
                       coords, dims, cart_comm);
    if(0)
        data_exhange3d_extended(quant_inds.data(), block_dims, block_strides, w_quant_inds.data(), w_block_dims,
                                w_block_strides, 1, coords, dims, cart_comm);
    // barrier
    // clear the memory used by local quant 

    MPI_Barrier(cart_comm);
    time_exchnage1 = MPI_Wtime() - time_exchnage1;
    // if (mpi_rank == 0) {
    //     printf("data exchange time = %f \n", time);
    // }
    std::vector<int>().swap(quant_inds);  // Forces reallocation and frees memory

    // boundary detection and sign map generation
    std::vector<char> boundary(block_size, 0);
    std::vector<char> sign_map(block_size, 0);
    std::vector<float> compensation_map(block_size, 0.0);
    if (operation)
        get_boundary_and_sign_map3d<int, char>(w_quant_inds.data(), boundary.data(), sign_map.data(), w_block_dims,
                                               w_block_strides, block_dims, block_strides, coords, dims, cart_comm);

    MPI_Barrier(cart_comm);
    if (mpi_rank == 0) {
        printf("boundary and sign map done \n");
    }

    // edt to get the distance map and the indexes

    int depth_dim = orig_dims[0] / dims[0];
    int height_dim = orig_dims[1] / dims[1];
    int width_dim = orig_dims[2] / dims[2];
    std::array<int, 3> data_block_dims = {0, 0, 0};
    data_block_dims[0] = depth_dim;
    data_block_dims[1] = height_dim;
    data_block_dims[2] = width_dim;
    std::vector<size_t> index(num_elements, 0);
    std::vector<float> distance(num_elements, 0.0);

    if (1) {
        edt_3d_and_sign_map(boundary.data(), distance.data(), index.data(), sign_map.data(), data_block_dims.data(),
                            dims, coords, mpi_rank, size, cart_comm);
        MPI_Barrier(cart_comm);
    }
    if (mpi_rank == 0) {
        printf("first edt done  \n");
    }

    // complete the sign map for non-edge voxels
    // fill the compensation map for the edge voxels
    char b_tag = 1;
    // if (1) {
    //     fill_sign_map3d<char, float>(sign_map.data(), index.data(), compensation_map.data(), boundary.data(), b_tag,
    //                                  block_size, compensation_magnitude);
    // }
    std::vector<char> w_sign_map(w_block_size, 0);
    double exchange_time2 = MPI_Wtime();
    if (1) {
        data_exhange3d(sign_map.data(), block_dims, block_strides, w_sign_map.data(), w_block_dims, w_block_strides,
                       coords, dims, cart_comm);
    }
    MPI_Barrier(cart_comm);
    exchange_time2 = MPI_Wtime() - exchange_time2;
    if (mpi_rank == 0) {
        printf("second data exchange done  \n");
    }
    // get the second boundary map
    std::vector<char> boundary_neutral(block_size, 0);
    if (operation) {
        get_boundary3d<char, char>(w_sign_map.data(), boundary_neutral.data(), w_block_dims, w_block_strides,
                                   block_dims, block_strides, coords, dims, cart_comm);
    }
    if (operation) filter_neutral_boundary3d(boundary.data(), boundary_neutral.data(), b_tag, block_size);

    MPI_Barrier(cart_comm);

    if (mpi_rank == 0) {
        printf("new boundary completed  \n");
    }
    std::vector<size_t> index_neutral(num_elements, 0);
    std::vector<float> distance_neutral(num_elements, 0.0);
    if (1) {
        edt_3d(boundary_neutral.data(), distance_neutral.data(), index_neutral.data(), data_block_dims.data(), dims,
               coords, mpi_rank, size, cart_comm);

        // edt_3d_and_sign_map(boundary_neutral.data(), distance_neutral.data(), sign_map.data(),
        // data_block_dims.data(),
        //                     dims, coords, mpi_rank, size, cart_comm);
        MPI_Barrier(cart_comm);
    }

    if (mpi_rank == 0) {
        printf("sedond edt done  \n");
    }

    // compensation
    if (operation && !use_rbf) {
        compensation_idw(compensation_map.data(), data.get(), distance.data(), distance_neutral.data(), sign_map.data(),
                         block_size, compensation_magnitude);
    }

    if (operation && use_rbf) {
        compensation_rbf(compensation_map.data(), data.get(), distance.data(), index.data(), distance_neutral.data(),
                         index_neutral.data(), orig_dims, sign_map.data(), block_size, compensation_magnitude);
    }
    MPI_Barrier(cart_comm);
    time = MPI_Wtime() - time;
    if (mpi_rank == 0) {
        printf("idw done \n");
    }
    // calculate the psnr
    double psnr = -1;
    if (1) psnr = get_psnr_mpi(orig_copy.data(), data.get(), num_elements, cart_comm);
    if (mpi_rank == 0) {
        printf("PSNR: %f\n", psnr);
    }

    // write the quantized data to a file
    char out_filename[100];
    // sprintf(out_filename, "%s/vx_%d_%d_%d.quant.i32", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<int>(out_filename, quant_inds.data(), block_size);
    // // printf("Rank %d, wrote quantized data to %s\n", mpi_rank, out_filename);
    {
        char out_filename[100];
        sprintf(out_filename, "%s/%s_%d_%d_%d.post3d.f32", out_dir.c_str(), name_prefix.c_str(), coords[0],
                coords[1], coords[2]);
        writefile<float>(out_filename, data.get(), block_size);
    }

    // sprintf(out_filename, "%s/vx_%d_%d_%d.wquant.i32", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<int>(out_filename, w_quant_inds.data(), w_block_size);

    // sprintf(out_filename, "%s/vx_%d_%d_%d.boundary.i8", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<char>(out_filename, boundary.data(), block_size);

    // sprintf(out_filename, "%s/vx_%d_%d_%d.sign.i8", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<char>(out_filename, sign_map.data(), block_size);

    // char distance_filename[100];
    // sprintf(distance_filename, "%s/distance_%d_%d_%d.f32", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<float>(distance_filename, distance.data(), num_elements);

    // char distance_filename[100];
    // sprintf(distance_filename, "%s/vx_%d_%d_%d.d2.f32", dir_prefix.c_str(), coords[0], coords[1], coords[2]);
    // writefile<float>(distance_filename, distance_neutral.data(), num_elements);

    // printf("Rank %d, wrote decomp data to %s\n", mpi_rank, out_filename);
    if (mpi_rank == 0) {
        printf("Global max: %f, Global min: %f, abs_eb = %f \n", global_max, global_min, abs_eb);
        printf("Rank %d, time: %f\n", mpi_rank, time);
        printf("Rank %d, exchange time1: %f\n", mpi_rank, time_exchnage1);
        printf("Rank %d, exchange time2: %f\n", mpi_rank, exchange_time2);
    }

    // terminate MPI
    MPI_Finalize();
    return 0;
}
