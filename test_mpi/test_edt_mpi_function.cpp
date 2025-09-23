#include "mpi/edt.hpp"
#include "utils/file_utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <array>


int main(int argc, char *argv[]) {
  int mpi_rank, size;

  int dims[3] = {2, 2, 2}; // Let MPI decide the best grid dimensions
  dims[0] = atoi(argv[1]);
  dims[1] = atoi(argv[2]);
  dims[2] = atoi(argv[3]);
  std::string dir_prefix = argv[4]; // Directory prefix for the blocks 


  int orig_dims[3] = {256, 384, 384};
  int periods[3] = {0, 0, 0}; // No periodicity in any dimension
  int coords[3] = {0, 0, 0};  // Coords of this process in the grid
  MPI_Comm cart_comm;

  // MPI_Init(&argc, &argv);
  MPI_Init(nullptr, nullptr);
  MPI_Finalize();

  return 0; 
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Let MPI determine the best grid shape given size
  // MPI_Dims_create(size, 3, dims);

  if (mpi_rank == 0) {
    printf("Number of processes: %d\n", size);
    printf("Grid dimensions: (%d, %d, %d)\n", dims[0], dims[1], dims[2]);
  }

  if(0){

    char hostname[256];  // Buffer to store the hostname
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        std::cout << "Hostname: " << hostname << std::endl;
    } else {
        std::cerr << "Error retrieving hostname" << std::endl;
    }
  }


  // Create a 3D Cartesian communicator
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

  // Get coordinates of this process
  MPI_Cart_coords(cart_comm, mpi_rank, 3, coords);
  // printf("Rank %d -> Coords (%d, %d, %d)\n", mpi_rank, coords[0], coords[1],
  // coords[2]);




  // // Get neighbors in each direction
  // int neighbor_left, neighbor_right;
  // MPI_Cart_shift(cart_comm, 0, 1, &neighbor_left, &neighbor_right);

  // int neighbor_down, neighbor_up;
  // MPI_Cart_shift(cart_comm, 1, 1, &neighbor_down, &neighbor_up);

  // int neighbor_back, neighbor_front;
  // MPI_Cart_shift(cart_comm, 2, 1, &neighbor_back, &neighbor_front);

  // printf("Rank %d -> Coords (%d, %d, %d)\n", rank, coords[0], coords[1],
  // coords[2]);

  // read files by coords
  char filename[100];
  sprintf(filename, "%s/data_%d_%d_%d.bin", dir_prefix.c_str(), coords[0], coords[1],
          coords[2]);
  // printf("  Reading file: %s\n", filename);

  size_t num_elements = 0;
  auto data = readfile<char>(filename, num_elements);
  if (data == nullptr) {
    fprintf(stderr, "Error reading file %s\n", filename);
    MPI_Finalize();
    return 1;
  }

  // prepare the buffers for edt calculation
  // init edt

  MPI_Barrier(cart_comm);

  // void edt_core_mpi(int* d_output, size_t stride, uint rank, uint d,  uint
  // len, uint width, uint height, size_t width_stride, size_t height_stride,
  // int local_x, int local_y,
  // int mpi_rank, int mpi_size, int* mpi_choors, int mpi_depth, int mpi_height,
  // int mpi_width)

  // depth direction
  int depth_dim = orig_dims[0] / dims[0];
  int height_dim = orig_dims[1] / dims[0];
  int width_dim = orig_dims[2] / dims[0];
  std::array<int, 3> data_block_dims = {depth_dim, height_dim, width_dim};

  std::vector<int> index(num_elements, 0);
  std::vector<float> distance(num_elements, 0.0); 

  edt_3d(data.get(), distance.data(), index.data(), data_block_dims.data(), dims, coords, mpi_rank, size, cart_comm);


    // write the distance to a file
  char distance_filename[100];
  sprintf(distance_filename, "%s/distance_%d_%d_%d.f32", dir_prefix.c_str(), coords[0],
          coords[1], coords[2]);
  writefile<float>(distance_filename, distance.data(), num_elements);
  // calculate distance  
  MPI_Finalize();
  return 0;
}
