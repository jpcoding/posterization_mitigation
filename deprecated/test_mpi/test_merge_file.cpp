#include <mpi.h>
#include <stdio.h>
#include <cstddef>
#include <cstdlib>
#include <string>
#include "CLI/CLI.hpp"
#include "utils/file_utils.hpp"

int main(int argc, char** argv) {
    int mpi_rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CLI::App app{"Merge files using MPI - 3D"};
    argv = app.ensure_utf8(argv);
    std::array<int, 3> dims = {0, 0, 0};  // Let MPI decide the best grid dimensions
    std::array<int, 3> orig_dims = {0, 0, 0};
    std::string dir_prefix;   // Directory prefix for the blocks
    std::string orig_prefix;  // Name prefix for the blocks
    std::string orig_sufix;   // Directory prefix for the output files
    std::string out_filename;
    app.add_option("--mpidims", dims, "mpi dimensions")->required();
    app.add_option("--dir", dir_prefix, "input file")->required();
    app.add_option("--prefix", orig_prefix, "output file")->required();
    app.add_option("--sufix", orig_sufix, "output file")->required();
    app.add_option("--orig_dims", orig_dims, "dimensions")->required();
    app.add_option("--out", out_filename, "output file")->required();
    CLI11_PARSE(app, argc, argv);

    int block_dims[3] = {0, 0, 0};
    block_dims[0] = orig_dims[0] / dims[0];
    block_dims[1] = orig_dims[1] / dims[1];
    block_dims[2] = orig_dims[2] / dims[2];
    size_t block_size = block_dims[0] * block_dims[1] * block_dims[2];

    size_t global_strides[3] = {(size_t)orig_dims[2] * orig_dims[1], (size_t)orig_dims[2], 1};
    size_t local_strides[3] = {(size_t)block_dims[2] * block_dims[1], (size_t)block_dims[2], 1};

    int periods[3] = {0, 0, 0};  // No periodicity in any dimension
    int coords[3] = {0, 0, 0};   // Coords of this process in the grid

    // MPI_Dims_create(size, 3, dims);
    if (mpi_rank == 0) {
        printf("Number of processes: %d\n", size);
        printf("Grid dimensions: (%d, %d, %d)\n", dims[0], dims[1], dims[2]);
        printf("Block dimensions: (%d, %d, %d)\n", block_dims[0], block_dims[1], block_dims[2]);
    }
    // Create a 3D Cartesian communicator
    // MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods, 1, &cart_comm);
    // MPI_Cart_coords(cart_comm, mpi_rank, 3, coords);

    int coord_nums = dims[0] * dims[1] * dims[2];
    std::vector<std::array<int, 3>> coords_list(coord_nums);

    
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                coords_list[i * dims[1] * dims[2] + j * dims[2] + k] = {i, j, k};
            }
        }
    }

    // for each rank, get coord_nums/size coords
    int coords_per_rank = coord_nums / size;
    int start = coords_per_rank * mpi_rank;
    int end = coords_per_rank * (mpi_rank + 1);
    if (mpi_rank == size - 1) {
        end = coord_nums;
    }

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, out_filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_Barrier(MPI_COMM_WORLD);    
    double time = MPI_Wtime();
    for (int i = start; i < end; i++) {
        coords[0] = coords_list[i][0];
        coords[1] = coords_list[i][1];
        coords[2] = coords_list[i][2];
        // read data for each rank
        char filename[100];
        sprintf(filename, "%s/%s_%d_%d_%d%s", dir_prefix.c_str(), orig_prefix.c_str(), coords[0], coords[1], coords[2],
                orig_sufix.c_str());
        size_t num_elements = 0;
        auto data = readfile<float>(filename, num_elements);

        // merge and write file to one file
        {
            int local_starts[3] = {0, 0, 0};
            for (int i = 0; i < 3; i++) {
                local_starts[i] = coords[i] * block_dims[i];
            }
            MPI_Datatype subarray;
            MPI_Type_create_subarray(3, orig_dims.data(), block_dims, local_starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
            MPI_Type_commit(&subarray);
            MPI_File_set_view(fh, 0, MPI_FLOAT, subarray, "native", MPI_INFO_NULL);
            MPI_File_write(fh, data.get(), block_size, MPI_FLOAT, MPI_STATUS_IGNORE);
            MPI_Type_free(&subarray);
        }
    }
    MPI_File_close(&fh);
    MPI_Barrier(MPI_COMM_WORLD);    
    double time_end = MPI_Wtime() - time; 
    if (mpi_rank == 0) {
        printf("Time taken to merge files: %f seconds\n", time_end);
    }
    


    MPI_Finalize();
    return 0;
    return 0;
}