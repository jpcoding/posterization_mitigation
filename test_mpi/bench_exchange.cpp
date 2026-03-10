#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "mpi/data_exchange.hpp"

template <typename T>
double run(int* block_dims, int* w_block_dims, int extend_size,
           int* coords, int* dims, MPI_Comm cart_comm,
           int nwarmup, int nreps, bool nb) {
    size_t src_size = (size_t)block_dims[0] * block_dims[1] * block_dims[2];
    size_t dst_size = (size_t)w_block_dims[0] * w_block_dims[1] * w_block_dims[2];
    std::vector<T> src(src_size, (T)1);
    std::vector<T> dst(dst_size, (T)0);
    size_t ss[3] = {(size_t)block_dims[1]   * block_dims[2],   (size_t)block_dims[2],   1};
    size_t ds[3] = {(size_t)w_block_dims[1] * w_block_dims[2], (size_t)w_block_dims[2], 1};

    auto call = [&]() {
        if (nb)
            data_exchange3d_extended_nb(src.data(), block_dims, ss,
                                        dst.data(), w_block_dims, ds,
                                        extend_size, coords, dims, cart_comm);
        else
            data_exchange3d_extended(src.data(), block_dims, ss,
                                     dst.data(), w_block_dims, ds,
                                     extend_size, coords, dims, cart_comm);
    };
    for (int r = 0; r < nwarmup; r++) call();
    MPI_Barrier(cart_comm);
    double t0 = MPI_Wtime();
    for (int r = 0; r < nreps; r++) call();
    MPI_Barrier(cart_comm);
    return (MPI_Wtime() - t0) / nreps;
}

// Persistent version: setup once outside the timed loop, only exchange() is timed.
template <typename T>
double run_persistent(int* block_dims, int* w_block_dims, int extend_size,
                      int* coords, int* dims, MPI_Comm cart_comm,
                      int nwarmup, int nreps) {
    size_t src_size = (size_t)block_dims[0] * block_dims[1] * block_dims[2];
    size_t dst_size = (size_t)w_block_dims[0] * w_block_dims[1] * w_block_dims[2];
    std::vector<T> src(src_size, (T)1);
    std::vector<T> dst(dst_size, (T)0);
    size_t ss[3] = {(size_t)block_dims[1]   * block_dims[2],   (size_t)block_dims[2],   1};
    size_t ds[3] = {(size_t)w_block_dims[1] * w_block_dims[2], (size_t)w_block_dims[2], 1};

    auto ctx = make_exchange_context(src.data(), block_dims, ss,
                                     dst.data(), w_block_dims, ds,
                                     extend_size, coords, dims, cart_comm);
    for (int r = 0; r < nwarmup; r++) exchange(ctx);
    MPI_Barrier(cart_comm);
    double t0 = MPI_Wtime();
    for (int r = 0; r < nreps; r++) exchange(ctx);
    MPI_Barrier(cart_comm);
    double t = (MPI_Wtime() - t0) / nreps;
    free_exchange_context(ctx);
    return t;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n       = 64;
    int extend  = 1;
    int nwarmup = 5;
    int nreps   = 50;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n")       && i+1 < argc) n       = atoi(argv[++i]);
        if (!strcmp(argv[i], "--extend")  && i+1 < argc) extend  = atoi(argv[++i]);
        if (!strcmp(argv[i], "--nwarmup") && i+1 < argc) nwarmup = atoi(argv[++i]);
        if (!strcmp(argv[i], "--nreps")   && i+1 < argc) nreps   = atoi(argv[++i]);
    }

    int dims[3]    = {0, 0, 0};
    int periods[3] = {0, 0, 0};
    MPI_Dims_create(world_size, 3, dims);
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, world_rank, 3, coords);

    int block_dims[3]   = {n, n, n};
    int w_block_dims[3];
    for (int d = 0; d < 3; d++) {
        int has_lo = (coords[d] != 0)         ? extend : 0;
        int has_hi = (coords[d] != dims[d]-1) ? extend : 0;
        w_block_dims[d] = n + has_lo + has_hi;
    }

    if (world_rank == 0)
        printf("ranks=%d  grid=(%d,%d,%d)  block=%d^3  extend=%d  nreps=%d\n",
               world_size, dims[0], dims[1], dims[2], n, extend, nreps);

    double t_blk  = run<int>(block_dims, w_block_dims, extend, coords, dims, cart_comm, nwarmup, nreps, false);
    double t_nb   = run<int>(block_dims, w_block_dims, extend, coords, dims, cart_comm, nwarmup, nreps, true);
    double t_pers = run_persistent<int>(block_dims, w_block_dims, extend, coords, dims, cart_comm, nwarmup, nreps);

    double t_blk_max, t_nb_max, t_pers_max;
    MPI_Reduce(&t_blk,  &t_blk_max,  1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&t_nb,   &t_nb_max,   1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&t_pers, &t_pers_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (world_rank == 0) {
        printf("blocking  : %7.4f ms/call  (1.00x)\n", t_blk_max  * 1e3);
        printf("nb        : %7.4f ms/call  (%.2fx)\n",  t_nb_max   * 1e3, t_blk_max / t_nb_max);
        printf("persistent: %7.4f ms/call  (%.2fx)\n",  t_pers_max * 1e3, t_blk_max / t_pers_max);
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
