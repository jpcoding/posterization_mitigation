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
#include "mpi/boundary.hpp"
#include "mpi/compensation.hpp"
#include "mpi/data_exchange.hpp"
#include "mpi/edt.hpp"
#include "mpi/edt_opt.hpp"
#include "mpi/mpi_datatype.hpp"
#include "mpi/stats.hpp"
#include "utils/file_utils.hpp"
#include "utils/qcat_ssim.hpp"
#include "utils/stats.hpp"
#include "utils/timer.hpp"

namespace SZ = SZ3;

int main(int argc, char** argv) {
    int mpi_rank, size;
    MPI_Comm cart_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::array<int, 3> dims_array = {1, 1, 1};  // Let MPI decide the best grid dimensions
    std::array<int, 3> orig_dims_array = {0, 0, 0};

    double rel_eb = 0.0;
    std::string input;  // Directory prefix for the blocks
    std::string qfile;
    std::string cfile;
    bool use_rbf = false;
    bool local_edt = false;
    bool local_quant = false;
    bool debug = false;

    CLI::App app{"Merge files using MPI - 3D"};
    argv = app.ensure_utf8(argv);
    app.add_option("--rel_eb", rel_eb, "Relative error bound")->required();
    app.add_option("--input", input, "input file")->required();
    app.add_option("--origdims", orig_dims_array, "dimensions")->required();
    app.add_option("--qfile", qfile, "quantized file")->required();
    app.add_option("--cfile", cfile, "compensation file")->required();
    app.add_option("--use_rbf", use_rbf, "use rbf")->required();
    app.add_option("--local_edt", local_edt, "use local edt")->required();
    app.add_option("--local_quant", local_quant, "use local quant")->required();
    app.add_option("--debug", debug, "debug")->default_val(false);
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
    MPI_Cart_coords(cart_comm, mpi_rank, 3, coords);
    size_t num_elements = 0;
    auto data = readfile<float>(input.c_str(), num_elements);
    std::vector<float> orig_copy(data.get(), data.get() + num_elements);

    if (data == nullptr) {
        fprintf(stderr, "Error reading file %s\n", input.c_str());
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
    float global_max = local_max;
    float global_min = local_min;

    bool operation = true;
    double local_range = local_max - local_min;
    if (local_range < 1e-10) {
        operation = false;
    }
    // barrier
    auto timer = Timer();
    double abs_eb = rel_eb * (global_max - global_min);
    double compensation_magnitude = abs_eb * 0.9;
    // printf("Rank %d, block_size: %d\n", mpi_rank, block_size);
    auto quantizer = SZ::LinearQuantizer<float>();
    quantizer.set_eb(abs_eb);
    std::vector<int> quant_inds(block_size, 0);
    size_t local_zero_count = 0;
    timer.start();
    {
        for (int i = 0; i < block_size; i++) {
            quant_inds[i] = quantizer.quantize_and_overwrite(data[i], 0) - 32768;
            if (quant_inds[i] == 0) local_zero_count++;
        }
    }
    {
        printf("quantization time = %f \n", timer.stop());
        printf("zero count = %ld \n", local_zero_count);
        printf("zero ratio = %f\n", (double)local_zero_count / block_size);
    }

    {
        double psnr, nrmse, max_diff;
        verify(orig_copy.data(), data.get(), block_size, psnr, nrmse, max_diff);
        std::vector<size_t> dims_(3);
        for (int i = 0; i < 3; i++) {
            dims_[i] = block_dims[i];
        }
        auto ssim = PM::calculateSSIM(orig_copy.data(), data.get(), 3, dims_.data());
        printf("SSIM = %f\n", ssim);
    }

    size_t global_zero_count = 0;
    size_t global_size = 1;
    for (int i = 0; i < 3; i++) {
        global_size *= orig_dims[i];
    }

    { writefile<float>(qfile.c_str(), data.get(), block_size); }

    std::vector<char> boundary(block_size, 0);
    std::vector<char> sign_map(block_size, 0);
    std::vector<float> compensation_map(block_size, 0.0);
    bool use_local_boundary = local_quant;

    timer.start();
    if (operation) {
        get_boundary_and_sign_map3d_local<int, char>(quant_inds.data(), boundary.data(), sign_map.data(), block_dims,
                                                     block_strides, block_dims, block_strides, coords, dims, cart_comm);
    }
    // std::vector<int>().swap(quant_inds);  // Forces reallocation and frees memory

    // printf("boundary and sign map done \n");
    if (debug) {
        writefile("boundary_mpi.bin", boundary.data(), block_size);
    }

    int depth_dim = orig_dims[0] / dims[0];
    int height_dim = orig_dims[1] / dims[0];
    int width_dim = orig_dims[2] / dims[0];
    std::array<int, 3> data_block_dims = {0, 0, 0};
    data_block_dims[0] = depth_dim;
    data_block_dims[1] = height_dim;
    data_block_dims[2] = width_dim;
    std::vector<size_t> index(num_elements, 0);
    std::vector<float> distance(num_elements, 0.0);

    auto timer2 = Timer();
    timer2.start();

    
    if (1) {
        edt_3d_and_sign_map_opt(boundary.data(), distance.data(), index.data(), sign_map.data(), data_block_dims.data(),
                                dims, coords, mpi_rank, size, cart_comm, local_edt);
    }
    std::cout << "edt time = " << timer2.stop() << std::endl; 
    
    // printf("first edt done  \n");
    if (debug) {
        writefile("distance_mpi.bin", distance.data(), block_size);
        writefile("sign_map_mpi.bin", sign_map.data(), block_size);
    }

    char b_tag = 1;

    std::vector<char> boundary_neutral(block_size, 0);

    if (operation) {
        get_boundary3d_local<char, char>(sign_map.data(), boundary_neutral.data(), block_dims, block_strides,
                                         block_dims, block_strides, coords, dims, cart_comm);
    }

    if (operation) filter_neutral_boundary3d(boundary.data(), boundary_neutral.data(), b_tag, block_size);

    std::vector<size_t> index_neutral(num_elements, 0);
    std::vector<float> distance_neutral(num_elements, 0.0);
    if (1) {
        edt_3d_opt(boundary_neutral.data(), distance_neutral.data(), index_neutral.data(), data_block_dims.data(), dims,
                   coords, mpi_rank, size, cart_comm, local_edt);
    }

    printf("sedond edt done  \n");

    // compensation
    if (operation && !use_rbf) {
        compensation_idw(compensation_map.data(), data.get(), distance.data(), distance_neutral.data(), sign_map.data(),
                         block_size, compensation_magnitude);
    }

    if (operation && use_rbf) {
        compensation_rbf(compensation_map.data(), data.get(), distance.data(), index.data(), distance_neutral.data(),
                         index_neutral.data(), orig_dims, sign_map.data(), block_size, compensation_magnitude);
    }
    printf("compensation time = %f\n", timer.stop());
    {
        double psnr, nrmse, max_diff;
        verify(orig_copy.data(), data.get(), block_size, psnr, nrmse, max_diff);
        std::vector<size_t> dims_(3);
        for (int i = 0; i < 3; i++) {
            dims_[i] = block_dims[i];
        }
        auto ssim = PM::calculateSSIM(orig_copy.data(), data.get(), 3, dims_.data());
        printf("SSIM = %f\n", ssim);
    }

    { writefile<float>(cfile.c_str(), data.get(), block_size); }

    
    MPI_Finalize();
    return 0;
}
