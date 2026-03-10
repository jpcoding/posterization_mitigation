#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <vector>

#include "mpi/data_exchange.hpp"
#include "utils/file_utils.hpp"
#include "utils/qcat_ssim.hpp"

int main(int argc, char** argv) {
    int mpi_rank, size;

    int dims[3];
    dims[0] = atoi(argv[1]);
    dims[1] = atoi(argv[2]);
    dims[2] = atoi(argv[3]);
    std::string dir_prefix  = argv[4];  // directory containing block files
    std::string orig_prefix = argv[5];  // e.g. "velocity_x"
    std::string decomp_suffix = argv[6]; // e.g. ".decomp.f32" (appended after coord suffix)

    int orig_dims[3];
    orig_dims[0] = atoi(argv[7]);
    orig_dims[1] = atoi(argv[8]);
    orig_dims[2] = atoi(argv[9]);

    int block_dims[3];
    for (int i = 0; i < 3; i++) block_dims[i] = orig_dims[i] / dims[i];
    size_t block_size = (size_t)block_dims[0] * block_dims[1] * block_dims[2];

    int periods[3] = {0, 0, 0};
    int coords[3]  = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);
    MPI_Cart_coords(cart_comm, mpi_rank, 3, coords);

    if (mpi_rank == 0) {
        printf("Processes: %d, grid: (%d,%d,%d), block: (%d,%d,%d)\n",
               size, dims[0], dims[1], dims[2],
               block_dims[0], block_dims[1], block_dims[2]);
    }

    // Read local block files directly — no merging needed
    char filename[256];
    size_t n = 0;

    sprintf(filename, "%s/%s_%d_%d_%d.f32",
            dir_prefix.c_str(), orig_prefix.c_str(), coords[0], coords[1], coords[2]);
    auto orig_data = readfile<float>(filename, n);

    sprintf(filename, "%s/%s_%d_%d_%d%s",
            dir_prefix.c_str(), orig_prefix.c_str(), coords[0], coords[1], coords[2],
            decomp_suffix.c_str());
    auto decomp_data = readfile<float>(filename, n);

    // Halo exchange setup
    const int win_size  = 7;
    const int win_shift = 2;
    const int extend_size = win_size - win_shift;  // 5

    // Expanded dims: add extend_size on each non-boundary side
    int w_dims[3];
    int data_offset[3];  // where local data sits in expanded buffer
    for (int i = 0; i < 3; i++) {
        w_dims[i] = block_dims[i];
        if (coords[i] != 0)          w_dims[i] += extend_size;
        if (coords[i] != dims[i]-1)  w_dims[i] += extend_size;
        data_offset[i] = (coords[i] == 0) ? 0 : extend_size;
    }
    size_t w_size = (size_t)w_dims[0] * w_dims[1] * w_dims[2];

    size_t src_strides[3] = {(size_t)block_dims[1] * block_dims[2], (size_t)block_dims[2], 1};
    size_t w_strides[3]   = {(size_t)w_dims[1] * w_dims[2], (size_t)w_dims[2], 1};

    std::vector<float> w_orig(w_size, 0.0f);
    std::vector<float> w_decomp(w_size, 0.0f);

    double t_exchange = MPI_Wtime();
    data_exchange3d_extended(orig_data.get(),   block_dims, src_strides,
                            w_orig.data(),     w_dims,     w_strides,
                            extend_size, coords, dims, cart_comm);
    data_exchange3d_extended(decomp_data.get(), block_dims, src_strides,
                            w_decomp.data(),   w_dims,     w_strides,
                            extend_size, coords, dims, cart_comm);
    if (mpi_rank == 0) printf("Halo exchange: %.4f s\n", MPI_Wtime() - t_exchange);

    // SSIM: each rank owns windows whose start falls within its local block.
    // Non-last ranks: last start at block_dims[i] - win_shift (window reaches into ghost).
    // Last rank: last start at block_dims[i] - win_size (window stays within data).
    size_t max_offset[3];
    for (int i = 0; i < 3; i++) {
        max_offset[i] = (coords[i] == dims[i]-1)
                        ? (block_dims[i] - win_size)
                        : (block_dims[i] - win_shift);
    }

    double t_ssim = MPI_Wtime();
    double local_ssim_sum = 0.0;
    double local_nw = 0.0;

    for (size_t o2 = 0; o2 <= max_offset[0]; o2 += win_shift) {
        for (size_t o1 = 0; o1 <= max_offset[1]; o1 += win_shift) {
            for (size_t o0 = 0; o0 <= max_offset[2]; o0 += win_shift) {
                local_nw++;
                local_ssim_sum += PM::SSIM_3d_calcWindow(
                    w_orig.data(), w_decomp.data(),
                    w_dims[1], w_dims[2],
                    data_offset[2] + o0,
                    data_offset[1] + o1,
                    data_offset[0] + o2,
                    win_size, win_size, win_size);
            }
        }
    }
    if (mpi_rank == 0) printf("SSIM compute: %.4f s\n", MPI_Wtime() - t_ssim);

    double global_ssim_sum = 0.0, global_nw = 0.0;
    MPI_Reduce(&local_ssim_sum, &global_ssim_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_nw,       &global_nw,       1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    if (mpi_rank == 0) printf("SSIM = %.6f\n", global_ssim_sum / global_nw);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
