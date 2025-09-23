#include "utils/file_utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <mpi.h>
#include <stdio.h>
#include <vector>
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include <string>

namespace SZ=SZ3;
int main(int argc, char *argv[]) {
    int mpi_rank, size;
  
    int dims[3] = {2, 2, 2}; // Let MPI decide the best grid dimensions
    dims[0] = atoi(argv[1]);
    dims[1] = atoi(argv[2]);
    dims[2] = atoi(argv[3]);
    double rel_eb = atof(argv[4]); // Relative error bound
    std::string dir_prefix = argv[5]; // Directory prefix for the blocks 
    
    int orig_dims[3] = {256, 384, 384};
    int periods[3] = {0, 0, 0}; // No periodicity in any dimension
    int coords[3] = {0, 0, 0};  // Coords of this process in the grid
    MPI_Comm cart_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // MPI_Dims_create(size, 3, dims);
    if (mpi_rank == 0) {
      printf("Number of processes: %d\n", size);
      printf("Grid dimensions: (%d, %d, %d)\n", dims[0], dims[1], dims[2]);
    }
    // Create a 3D Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);
    // Get coordinates of this process
    MPI_Cart_coords(cart_comm, mpi_rank, 3, coords);

    // read data for each rank 
    char filename[100];
    sprintf(filename, "%s/vx_%d_%d_%d.f32", dir_prefix.c_str(), coords[0], coords[1],
            coords[2]);
    size_t num_elements = 0;
    auto data = readfile<float>(filename, num_elements);
    if (data == nullptr) {
      fprintf(stderr, "Error reading file %s\n", filename);
      MPI_Finalize();
      return 1;
    }
    int block_dims[3] = {orig_dims[0] / dims[0], orig_dims[1] / dims[1],
        orig_dims[2] / dims[2]};
    int block_size = block_dims[0] * block_dims[1] * block_dims[2]; 
    assert(num_elements == block_size);
    // printf("Rank %d, num_elements: %ld\n", mpi_rank, num_elements);

    // get the global max and min to get the gloibal value range 
    float local_max = *std::max_element(data.get(), data.get() + num_elements);
    float local_min = *std::min_element(data.get(), data.get() + num_elements);
    float global_max, global_min;
    // just reduce the max and min to get the global max and min to the root

    // barrier
    MPI_Barrier(cart_comm);

    double time = MPI_Wtime();  // Start the timer
    MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, 0, cart_comm);
    double abs_eb = rel_eb * (global_max - global_min); 

    // broadcast the global max and min to all ranks
    MPI_Bcast(&global_max, 1, MPI_FLOAT, 0, cart_comm);
    MPI_Bcast(&global_min, 1, MPI_FLOAT, 0, cart_comm);
    MPI_Bcast(&abs_eb, 1, MPI_DOUBLE, 0, cart_comm);

    // printf("Rank %d, block_size: %d\n", mpi_rank, block_size);
    auto quantizer = SZ::LinearQuantizer<float>();
    quantizer.set_eb(abs_eb);
    std::vector<int> quant_inds(block_size, 0);
    for (int i = 0; i < block_size; i++) {
        quant_inds[i] = quantizer.quantize_and_overwrite(data[i],0);
    }
    MPI_Barrier(cart_comm);
    time = MPI_Wtime() - time;

    // write the quantized data to a file
    char out_filename[100];
    sprintf(out_filename, "%s/vx_%d_%d_%d.quant.i32", dir_prefix.c_str(), coords[0], coords[1],
            coords[2]);
    writefile<int>(out_filename, quant_inds.data(), block_size);
    // printf("Rank %d, wrote quantized data to %s\n", mpi_rank, out_filename);
    sprintf(out_filename, "%s/vx_%d_%d_%d.decomp.f32", dir_prefix.c_str(), coords[0], coords[1],
        coords[2]);
    writefile<float>(out_filename, data.get(), block_size);

    if (mpi_rank == 0) {
        printf("Global max: %f, Global min: %f, abs_eb = %f \n", global_max, global_min, abs_eb);
        printf("Rank %d, time: %f\n", mpi_rank, time);
      }
      

    // terminate MPI
    MPI_Finalize();
    return 0;


}

