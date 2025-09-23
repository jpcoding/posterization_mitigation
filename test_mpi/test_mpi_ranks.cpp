#include <mpi.h>
#include <stdio.h>
#include <array>
#include <string>
#include "CLI/CLI.hpp"

int main(int argc, char** argv) {
    int mpi_rank, size;
    MPI_Comm cart_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::array<int, 3> dims_array = {0, 0, 0};  // Let MPI decide the best grid dimensions
    // CLI::App app{"Test ranks"};
    // argv = app.ensure_utf8(argv);
    // app.add_option("--mpidims", dims_array, "mpi dimensions")->required();
    // CLI11_PARSE(app, argc, argv);
    dims_array[0] = atoi(argv[1]);
    dims_array[1] = atoi(argv[2]);
    dims_array[2] = atoi(argv[3]);
    int* dims = dims_array.data(); 
    int periods[3] = {0, 0, 0};  // No periodicity in any dimension
    int coords[3] = {0, 0, 0};   // Coords of this process in the grid
    // Create a 3D Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);
    // Get coordinates of this process
    MPI_Cart_coords(cart_comm, mpi_rank, 3, coords);
    printf("Rank %d, coords: (%d, %d, %d)\n", mpi_rank, coords[0], coords[1], coords[2]);
    MPI_Finalize();
    return 0;
}
