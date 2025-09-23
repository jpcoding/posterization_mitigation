#include <mpi.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "CLI/CLI.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include "edt_transform_omp.hpp"
#include "mpi/boundary.hpp"
#include "mpi/compensation.hpp"
#include "mpi/data_exchange.hpp"
#include "mpi/edt.hpp"
#include "mpi/edt_opt.hpp"
#include "mpi/mpi_datatype.hpp"
#include "mpi/stats.hpp"
#include "utils/file_utils.hpp"

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
    std::string dir_prefix;   // Directory prefix for the blocks
    std::string name_prefix;  // Name prefix for the blocks
    std::string out_dir;
    bool use_rbf = false;
    bool local_edt = false;
    bool local_quant = false;
    std::string eb_mode = "rel";  // Relative error bound mode
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
    app.add_option("--local_edt", local_edt, "use local edt")->required();
    app.add_option("--local_quant", local_quant, "use local quant")->required();
    CLI11_PARSE(app, argc, argv);

    int* orig_dims = orig_dims_array.data();
    int* dims = dims_array.data();

    int periods[3] = {0, 0, 0};  // No periodicity in any dimension
    int coords[3] = {0, 0, 0};   // Coords of this process in the grid

    // MPI_Dims_create(size, 3, dims);
    if (mpi_rank == 0) {
        printf("Number of processes: %d\n", size);
        printf("Grid dimensions: (%d, %d, %d)\n", dims[0], dims[1], dims[2]);
        printf("use rbf: %d\n", (int)use_rbf);
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
    size_t block_size = block_dims[0] * block_dims[1] * block_dims[2];
    size_t block_strides[3] = {(size_t)block_dims[1] * block_dims[2], (size_t)block_dims[2], 1};
    // assert(num_elements == block_size);
    // printf("Rank %d, num_elements: %ld\n", mpi_rank, num_elements);

    double abs_eb;
    bool operation = true;
    if (eb_mode == "rel") {
        float local_max = *std::max_element(data.get(), data.get() + num_elements);
        float local_min = *std::min_element(data.get(), data.get() + num_elements);
        float global_max, global_min;

        double local_range = local_max - local_min;
        if (local_range < 1e-10) {
            operation = true;
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
        sprintf(out_filename, "%s/%s_%d_%d_%d.decomp.f32", out_dir.c_str(), name_prefix.c_str(), coords[0], coords[1],
                coords[2]);
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
    std::vector<int> w_quant_inds;

    // quantization index exchange
    MPI_Barrier(cart_comm);
    double total_run_time = MPI_Wtime();
    double time_exchnage1 = 0;
    {
        time_exchnage1 = MPI_Wtime();
        w_quant_inds.resize(w_block_size, 0);
        data_exhange3d(quant_inds.data(), block_dims, block_strides, w_quant_inds.data(), w_block_dims,
                           w_block_strides, coords, dims, cart_comm);
        // MPI_Barrier(cart_comm);
        time_exchnage1 = MPI_Wtime() - time_exchnage1;
    }

    // boundary detection and sign map generation
    std::vector<char> boundary(block_size, 0);
    std::vector<char> sign_map(block_size, 0);
    std::vector<float> compensation_map;

    bool use_local_boundary = local_quant;
    {
        
        get_boundary_and_sign_map3d<int, char>(w_quant_inds.data(), boundary.data(), sign_map.data(), w_block_dims,
                                               w_block_strides, block_dims, block_strides, coords, dims, cart_comm);
    }

    std::vector<int>().swap(quant_inds);    // Forces reallocation and frees memory
    std::vector<int>().swap(w_quant_inds);  // Forces reallocation and frees memory

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

    std::unique_ptr<float[]> distance;
    std::unique_ptr<size_t[]> index;

    auto edt_omp = PM2::EDT_OMP<float, int>();
    edt_omp.set_num_threads(1);
    {
        auto edt_result = edt_omp.NI_EuclideanFeatureTransform(boundary.data(), 3, data_block_dims.data(), 1);
        distance = std::move(edt_result.distance);
        index = std::move(edt_result.indexes);
        if (mpi_rank == 0) printf("edt time = %.10f \n", edt_omp.get_edt_time());
        for (size_t i = 0; i < block_size; i++) {
            if (boundary[i] != 1)  // non-boundary points ·
            {
                sign_map[i] = sign_map[index[i]];
            }
        }
    }

    char b_tag = 1;
    std::vector<char> w_sign_map;
    double  exchange_time2 = 0;   
    {
        // MPI_Barrier(cart_comm);
        exchange_time2 = MPI_Wtime();
        w_sign_map.resize(w_block_size, 0);
        {
            data_exhange3d(sign_map.data(), block_dims, block_strides, w_sign_map.data(), w_block_dims, w_block_strides,
                           coords, dims, cart_comm);
        }
        // MPI_Barrier(cart_comm);
        exchange_time2 = MPI_Wtime() - exchange_time2; 
        if (mpi_rank == 0) {
            printf("second data exchange done  \n");
        }
    }
    // get the second boundary map
    std::vector<char> boundary_neutral(block_size, 0);
    if (use_local_boundary == false) {
        get_boundary3d<char, char>(w_sign_map.data(), boundary_neutral.data(), w_block_dims, w_block_strides,
                                   block_dims, block_strides, coords, dims, cart_comm);
    }

    filter_neutral_boundary3d(boundary.data(), boundary_neutral.data(), b_tag, block_size);

    if (mpi_rank == 0) {
        printf("new boundary completed  \n");
    }

    std::unique_ptr<float[]> distance_neutral;
    std::unique_ptr<size_t[]> index_neutral;
    // timer.start();
    {
        auto edt_result = edt_omp.NI_EuclideanFeatureTransform(boundary_neutral.data(), 3, data_block_dims.data(), 1);
        distance_neutral = std::move(edt_result.distance);
        index_neutral = std::move(edt_result.indexes);
    }

    if (mpi_rank == 1) {
        printf("sedond edt done  \n");
    }

    // compensation
    if (operation && !use_rbf) {
        compensation_idw(compensation_map.data(), data.get(), distance.get(), distance_neutral.get(), sign_map.data(),
                         block_size, compensation_magnitude);
    }

    if (operation && use_rbf) {
        compensation_rbf(compensation_map.data(), data.get(), distance.get(), index.get(), distance_neutral.get(),
                         index_neutral.get(), orig_dims, sign_map.data(), block_size, compensation_magnitude);
    }
    MPI_Barrier(cart_comm);
    total_run_time = MPI_Wtime() - total_run_time;
    if (mpi_rank == 0) {
        printf("idw done \n");
    }
    // calculate the psnr
    double psnr = -1;
    if (1) psnr = get_psnr_mpi(orig_copy.data(), data.get(), num_elements, cart_comm);
    if (mpi_rank == 0) {
        printf("PSNR: %f\n", psnr);
    }

    {
        char out_filename[100];
        sprintf(out_filename, "%s/%s_%d_%d_%d.post3d.f32", out_dir.c_str(), name_prefix.c_str(), coords[0], coords[1],
                coords[2]);
        writefile<float>(out_filename, data.get(), block_size);
    }

    if (mpi_rank == 0) {
        printf("Rank %d, time: %f\n", mpi_rank, total_run_time);
        printf("Rank %d, data exchange time: %f\n", mpi_rank, time_exchnage1);
        printf("Rank %d, second data exchange time: %f\n", mpi_rank, exchange_time2);
    }

    // terminate MPI
    MPI_Finalize();
    return 0;
}