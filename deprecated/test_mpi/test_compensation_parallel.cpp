#include <mpi.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "CLI/CLI.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include "compensation.hpp"
#include "mpi/boundary.hpp"
#include "mpi/compensation.hpp"
#include "mpi/data_exchange.hpp"
#include "mpi/edt.hpp"
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

    std::string dir_prefix;   // Directory prefix for the blocks
    std::string name_prefix;  // Name prefix for the blocks
    std::string out_dir;
    bool use_rbf = false;
    std::string quantized_file_sufix;
    std::string compensated_file_sufix;
    std::string eb_mode;
    double eb;
    CLI::App app{"Merge files using MPI - 3D"};
    argv = app.ensure_utf8(argv);
    app.add_option("-e", eb, "Error bound")->required();
    app.add_option("-m", eb_mode, "Error bound mode")->required();
    app.add_option("--mpidims", dims_array, "mpi dimensions")->required();
    app.add_option("--dir", dir_prefix, "input file")->required();
    app.add_option("--prefix", name_prefix, "output file")->required();
    app.add_option("--q_suffix", quantized_file_sufix, "quantized file sufix")->required();
    app.add_option("--c_suffix", compensated_file_sufix, "compensated file sufix")->required();

    app.add_option("--origdims", orig_dims_array, "dimensions")->required();
    app.add_option("--outdir", out_dir, "output file")->required();
    app.add_option("--use_rbf", use_rbf, "use rbf")->required();
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
    size_t block_size = (size_t)block_dims[0] * block_dims[1] * block_dims[2];
    size_t block_strides[3] = {(size_t)block_dims[1] * block_dims[2], (size_t)block_dims[2], 1};
    // assert(num_elements == block_size);
    // printf("Rank %d, num_elements: %ld\n", mpi_rank, num_elements);

    // get the global max and min to get the gloibal value range
    double abs_eb;
    bool operation = true;
    if (eb_mode == "rel") {
        float local_max = *std::max_element(data.get(), data.get() + num_elements);
        float local_min = *std::min_element(data.get(), data.get() + num_elements);
        float global_max, global_min;

        double local_range = local_max - local_min;
        if (local_range < 1e-10) {
            operation = false;
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
    double time = MPI_Wtime();  // Start the timer
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
    }
    double orig_psnr = -1;
    if (1) orig_psnr = get_psnr_mpi(orig_copy.data(), data.get(), num_elements, cart_comm);
    if (mpi_rank == 0) {
        printf("Original PSNR: %f\n", orig_psnr);
    }
    {
        char out_filename[100];
        sprintf(out_filename, "%s/%s_%d_%d_%d%s", out_dir.c_str(), name_prefix.c_str(), coords[0], coords[1], coords[2],
                quantized_file_sufix.c_str());
        writefile<float>(out_filename, data.get(), block_size);
    }

    // embrassinly parallel compensation
    // barrier
    MPI_Barrier(cart_comm);
    double c_time = MPI_Wtime();  // Start the timer
    if (operation) {
        auto compensator =
            PM::Compensation<float, int>(3, block_dims, data.get(), quant_inds.data(), compensation_magnitude);
        compensator.set_edt_thread_num(1);
        compensator.set_use_rbf(use_rbf);
        auto compensation_map = compensator.get_compensation_map();
        // writefile("compensation_map.f32", compensation_map.data(), data_size);
        // add the compensation map to the dec_data
        for (int i = 0; i < block_size; i++) {
            data[i] += compensation_map[i];
        }
    }

    // barrier
    MPI_Barrier(cart_comm);
    c_time = MPI_Wtime() - c_time;

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
        sprintf(out_filename, "%s/%s_%d_%d_%d%s", out_dir.c_str(), name_prefix.c_str(), coords[0], coords[1], coords[2],
                compensated_file_sufix.c_str());
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
        printf("abs_eb = %f \n", abs_eb);
        printf("Rank %d, time: %f\n", mpi_rank, c_time);
    }

    // terminate MPI
    MPI_Finalize();
    return 0;
}
