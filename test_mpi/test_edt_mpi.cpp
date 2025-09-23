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
  orig_dims[0] = atoi(argv[5]);
  orig_dims[1] = atoi(argv[6]);
  orig_dims[2] = atoi(argv[7]);

  int periods[3] = {0, 0, 0}; // No periodicity in any dimension
  int coords[3] = {0, 0, 0};  // Coords of this process in the grid
  MPI_Comm cart_comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Let MPI determine the best grid shape given size
  // MPI_Dims_create(size, 3, dims);

  if (mpi_rank == 0) {
    printf("Number of processes: %d\n", size);
    printf("Grid dimensions: (%d, %d, %d)\n", dims[0], dims[1], dims[2]);
    printf("orig_dims = %d %d %d\n", orig_dims[0], orig_dims[1], orig_dims[2]);

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
  int *output = (int *)malloc(num_elements * sizeof(int) * 3);

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

  if(mpi_rank == 0)
  {
    printf("data_block_dims = %d %d %d\n", data_block_dims[0], data_block_dims[1], data_block_dims[2]); 
  }

  int max_dim =
      *std::max_element(data_block_dims.begin(), data_block_dims.end());

  int input_stride[3] = {height_dim * width_dim, width_dim, 1};
  int output_stride[3] = {3 * input_stride[0], 3 * input_stride[1],
                          3 * input_stride[2]};

  std::vector<float> distance(num_elements, 0.0);

  int max_mpi_dim = *std::max_element(dims, dims + 3);

  int *g = (int *)malloc(sizeof(int) * max_dim * max_mpi_dim);
  int *f = (int *)malloc(sizeof(int) * max_dim * 3 * max_mpi_dim);
  int **ff = (int **)malloc(sizeof(int *) * max_dim * max_mpi_dim);
  for (int i = 0; i < max_dim * max_mpi_dim; i++) {
    ff[i] = f + i * 3;
  }

  int direction;

  // dim 0
  direction = 0;
  int x_dir = (direction + 1) % 3;
  int y_dir = (direction + 2) % 3;

  if (0) {
    int i = 10;
    int j = 10;
    edt_core_mpi(output, output_stride[direction], 3, direction,
                 data_block_dims[direction], data_block_dims[x_dir],
                 data_block_dims[y_dir], output_stride[x_dir],
                 output_stride[y_dir], i, j, ff, g, mpi_rank, size, coords,
                 data_block_dims[0], data_block_dims[1], data_block_dims[2],
                 direction, dims, cart_comm);
  }

  // speed test
  // barrier
  MPI_Barrier(cart_comm);
  double time = MPI_Wtime();
  edt_init_mpi(data.get(), output, 1, data_block_dims[2], data_block_dims[1],
               data_block_dims[0], coords);

  if (1) {
    for (int i = 0; i < data_block_dims[x_dir]; i++) // y
    {
      for (int j = 0; j < data_block_dims[y_dir]; j++) // x
      {
        edt_core_mpi(output, output_stride[direction], 3, direction,
                     data_block_dims[direction], data_block_dims[x_dir],
                     data_block_dims[y_dir], output_stride[x_dir],
                     output_stride[y_dir], i, j, ff, g, mpi_rank, size, coords,
                     data_block_dims[0], data_block_dims[1], data_block_dims[2],
                     direction, dims, cart_comm);
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
    for (int i = 0; i < data_block_dims[x_dir]; i++) // y
    {
      for (int j = 0; j < data_block_dims[y_dir]; j++) // x
      {
        edt_core_mpi(output, output_stride[direction], 3, direction,
                     data_block_dims[direction], data_block_dims[x_dir],
                     data_block_dims[y_dir], output_stride[x_dir],
                     output_stride[y_dir], i, j, ff, g, mpi_rank, size, coords,
                     data_block_dims[0], data_block_dims[1], data_block_dims[2],
                     direction, dims, cart_comm);
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
    for (int i = 0; i < data_block_dims[x_dir]; i++) // y
    {
      for (int j = 0; j < data_block_dims[y_dir]; j++) // x
      {
        edt_core_mpi(output, output_stride[direction], 3, direction,
                     data_block_dims[direction], data_block_dims[x_dir],
                     data_block_dims[y_dir], output_stride[x_dir],
                     output_stride[y_dir], i, j, ff, g, mpi_rank, size, coords,
                     data_block_dims[0], data_block_dims[1], data_block_dims[2],
                     direction, dims, cart_comm);
      }
    }
    MPI_Barrier(cart_comm);
    // printf("rank %d finished dim 2\n", mpi_rank);
  }

  calculate_distance(output, distance.data(), data_block_dims[2],
                     data_block_dims[1], data_block_dims[0], coords);

  MPI_Barrier(cart_comm);

  time = MPI_Wtime() - time;
  if (mpi_rank == 0)
  {
    printf("Rank %d, time: %f\n", mpi_rank, time);
  }

  // calculate distance

  // write the output to a file
  char output_filename[100];
  sprintf(output_filename, "%s/edt_%d_%d_%d.i32", dir_prefix.c_str(), coords[0], coords[1],
          coords[2]);
  writefile<int>(output_filename, output, num_elements * 3);

  // write the distance to a file
  char distance_filename[100];
  sprintf(distance_filename, "%s/distance_%d_%d_%d.f32", dir_prefix.c_str(), coords[0],
          coords[1], coords[2]);
  writefile<float>(distance_filename, distance.data(), num_elements);

  free(output);
  free(g);
  free(f);
  free(ff);

  MPI_Finalize();
  return 0;
}
