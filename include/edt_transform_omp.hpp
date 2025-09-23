#ifndef EDT_TRANSFORM_OMP_HPP
#define EDT_TRANSFORM_OMP_HPP

#include <math.h>
#include <omp.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <vector>

#include "utils/file_utils.hpp"
#include "utils/timer.hpp"

namespace PM2 {

using npy_intp = int;
using npy_uint32 = unsigned int;
using npy_int32 = int;
using npy_double = double;
using npy_int8 = char;
#define NPY_MAXDIMS 5

template <typename T_distance, typename T_int>
class EDT_OMP {
    struct Distance_and_Index {
        std::unique_ptr<T_distance[]> distance;
        std::unique_ptr<size_t[]> indexes;
    };

   public:
    EDT_OMP() = default;

    void set_num_threads(int num_threads) { this->num_threads = num_threads; }

    double get_edt_time() { return edt_time; }

    double get_distance_time() { return distance_time; }

    double get_total_time() { return edt_time + distance_time; }

    void reset_timer() {
        this->edt_time = 0;
        this->distance_time = 0;
    }

    static void VoronoiFT(int *pf, int len, int *coor, int rank, int d, size_t stride, size_t cstride, int **f,
                          int *g) {
        int l = -1, ii, maxl, idx1, idx2;
        int jj;

        for (ii = 0; ii < len; ii++)
            for (jj = 0; jj < rank; jj++) f[ii][jj] = pf[ii * stride + jj];

        for (ii = 0; ii < len; ii++) {
            if (f[ii][0] >= 0) {
                double fd = f[ii][d];
                double wR = 0;
                for (jj = 0; jj < rank; jj++) {
                    if (jj != d) {
                        int tw = f[ii][jj] - coor[jj];
                        wR += tw * tw;
                    }
                }
                while (l >= 1) {
                    double a, b, c, uR = 0, vR = 0, f1;
                    idx1 = g[l];
                    f1 = f[idx1][d];
                    idx2 = g[l - 1];
                    a = f1 - f[idx2][d];
                    b = fd - f1;
                    c = a + b;
                    for (jj = 0; jj < rank; jj++) {
                        if (jj != d) {
                            double cc = coor[jj];
                            double tu = f[idx2][jj] - cc;
                            double tv = f[idx1][jj] - cc;
                            uR += tu * tu;
                            vR += tv * tv;
                        }
                    }
                    if (c * vR - b * uR - a * wR - a * b * c <= 0) break;
                    --l;
                }
                ++l;
                g[l] = ii;
            }
        }
        maxl = l;
        if (maxl >= 0) {
            l = 0;
            for (ii = 0; ii < len; ii++) {
                double delta1 = 0, t;
                for (jj = 0; jj < rank; jj++) {
                    t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];

                    delta1 += t * t;
                }
                while (l < maxl) {
                    double delta2 = 0.0;
                    for (jj = 0; jj < rank; jj++) {
                        t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];

                        delta2 += t * t;
                    }
                    if (delta1 <= delta2) break;
                    delta1 = delta2;
                    ++l;
                }
                idx1 = g[l];
                for (jj = 0; jj < rank; jj++) pf[ii * stride + jj] = f[idx1][jj];
            }
        }
    }

    void ComputeFT2D3D__(char *pi, int *pf, int *ishape, const size_t *istrides, const size_t *fstrides, int rank) {
        std::vector<int> coor(rank, 0);
        if (rank == 2) {
            omp_set_num_threads(num_threads);
            int max_dim = std::max(ishape[0], ishape[1]);
            std::vector<std::vector<int>> local_f(num_threads, std::vector<int>(max_dim * 2));
            std::vector<std::vector<int>> local_g(num_threads, std::vector<int>(max_dim));
            std::vector<std::vector<int *>> local_f_ptrs(num_threads, std::vector<int *>(max_dim));
            std::vector<std::array<int, 2>> coor_local(num_threads, {0, 0});
            for (int i = 0; i < num_threads; i++) {
                for (int j = 0; j < max_dim; j++) {
                    local_f_ptrs[i][j] = local_f[i].data() + j * 2;
                }
            }

#pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < ishape[1]; i++) {
                int thread_id = omp_get_thread_num();
                coor_local[thread_id][1] = i;
                for (int j = 0; j < ishape[0]; j++) {
                    size_t idx = i * istrides[1] + j * istrides[0];
                    if (pi[idx] != edge_tag) {  // non-boundary points
                        pf[idx * 2] = -1;
                    } else {
                        pf[idx * 2] = j;
                        pf[idx * 2 + 1] = i;
                    }
                }
                VoronoiFT(pf + i * fstrides[1], ishape[0], coor_local[thread_id].data(), rank, 0, fstrides[0],
                          fstrides[2], local_f_ptrs[thread_id].data(), local_g[thread_id].data());
            }
#pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < ishape[0]; i++) {
                int thread_id = omp_get_thread_num();
                coor_local[thread_id][0] = i;
                VoronoiFT(pf + i * fstrides[0], ishape[1], coor_local[thread_id].data(), rank, 1, fstrides[1],
                          fstrides[2], local_f_ptrs[thread_id].data(), local_g[thread_id].data());
            }

        } else if (rank == 3) {
            omp_set_num_threads(num_threads);
            int max_dim = std::max(ishape[0], std::max(ishape[1], ishape[2]));
            // use malloc to allocate the memory
            int *local_f = (int *)malloc(num_threads * max_dim * 3 * sizeof(int));
            int *local_g = (int *)malloc(num_threads * max_dim * sizeof(int));
            int **local_f_ptrs = (int **)malloc(num_threads * max_dim * sizeof(int *));
            int *coor_locals = (int *)malloc(num_threads * 3 * sizeof(int));
            for (int i = 0; i < num_threads; i++) {
                for (int j = 0; j < max_dim; j++) {
                    local_f_ptrs[i * max_dim + j] = local_f + i * max_dim * 3 + j * 3;
                }
            }
            // printf("cache line allocation time = %f\n", global_timer.stop());

#pragma omp parallel for collapse(2) num_threads(num_threads)
            for (int i = 0; i < ishape[0]; i++)  // 384
            {
                for (int j = 0; j < ishape[1]; j++)  // 384
                {
                    for (int k = 0; k < ishape[2]; k++) {
                        size_t idx = i * istrides[0] + j * istrides[1] + k * istrides[2];
                        if (pi[idx] != edge_tag) {
                            pf[idx * 3] = -1;
                        } else {
                            pf[idx * 3] = i;
                            pf[fstrides[3] + idx * 3] = j;
                            pf[fstrides[3] * 2 + idx * 3] = k;
                        }
                    }
                }
            }

            int direction = 0;
            int x_dir = (direction + 1) % 3;
            int y_dir = (direction + 2) % 3;

#pragma omp parallel for collapse(2) num_threads(num_threads)
            for (int i = 0; i < ishape[x_dir]; i++)  // 384
            {
                for (int j = 0; j < ishape[y_dir]; j++)  // 384
                {
                    int cur_thread_id = omp_get_thread_num();
                    int *coor_local = coor_locals + cur_thread_id * 3;
                    coor_local[direction] = 0;
                    coor_local[y_dir] = j;
                    coor_local[x_dir] = i;
                    VoronoiFT(pf + i * fstrides[x_dir] + j * fstrides[y_dir], ishape[direction], coor_local, rank,
                              direction, fstrides[direction], 1, local_f_ptrs + max_dim * cur_thread_id,
                              local_g + cur_thread_id * max_dim);
                }
            }

            direction = 1;
            x_dir = (direction + 1) % 3;
            y_dir = (direction + 2) % 3;

// the second dimension
#pragma omp parallel for collapse(2) num_threads(num_threads)
            for (int i = 0; i < ishape[x_dir]; i++) {
                for (int j = 0; j < ishape[y_dir]; j++) {
                    int cur_thread_id = omp_get_thread_num();
                    int *coor_local = coor_locals + cur_thread_id * 3;
                    coor_local[y_dir] = j;
                    coor_local[direction] = 0;
                    coor_local[x_dir] = i;
                    VoronoiFT(pf + i * fstrides[x_dir] + j * fstrides[y_dir], ishape[direction], coor_local, rank,
                              direction, fstrides[direction], 1, local_f_ptrs + max_dim * cur_thread_id,
                              local_g + cur_thread_id * max_dim);
                }
            }
            direction = 2;
            x_dir = (direction + 1) % 3;
            y_dir = (direction + 2) % 3;

// the first dimension
#pragma omp parallel for collapse(2) num_threads(num_threads)
            for (int i = 0; i < ishape[x_dir]; i++) {
                for (int j = 0; j < ishape[y_dir]; j++) {
                    int cur_thread_id = omp_get_thread_num();
                    int *coor_local = coor_locals + cur_thread_id * 3;
                    coor_local[y_dir] = j;
                    coor_local[x_dir] = i;
                    coor_local[direction] = 0;
                    VoronoiFT(pf + i * fstrides[x_dir] + j * fstrides[y_dir], ishape[direction], coor_local, rank,
                              direction, fstrides[direction], 1, local_f_ptrs + max_dim * cur_thread_id,
                              local_g + cur_thread_id * max_dim);
                }
            }
            free(local_f_ptrs);
            free(local_f);
            free(local_g);
        }
    }

    void ComputeFT2D3D(char *pi, int *pf, int *ishape, const size_t *istrides, const size_t *fstrides, int rank) {
        std::vector<int> coor(rank, 0);
        if (rank == 2) {
            omp_set_num_threads(num_threads);

            int max_dim = std::max(ishape[0], ishape[1]);

            // -------------------------
            // Allocate per-thread buffers
            // -------------------------
            std::vector<int *> local_f(num_threads);
            std::vector<int *> local_g(num_threads);
            std::vector<int **> local_f_ptrs(num_threads);
            std::vector<int *> coor_locals(num_threads);

            for (int t = 0; t < num_threads; t++) {
                local_f[t] = (int *)aligned_alloc(64, max_dim * 2 * sizeof(int));
                local_g[t] = (int *)aligned_alloc(64, max_dim * sizeof(int));
                coor_locals[t] = (int *)aligned_alloc(64, 2 * sizeof(int));
                // build pointer array for this thread
                local_f_ptrs[t] = (int **)malloc(max_dim * sizeof(int *));
                for (int j = 0; j < max_dim; j++) {
                    local_f_ptrs[t][j] = local_f[t] + j * 2;
                }
            }

            // -------------------------
            // Initialize pf (seed points)
            // -------------------------
            #pragma omp parallel for collapse(1) num_threads(num_threads)
            for (int i = 0; i < ishape[0]; i++) {
                for (int j = 0; j < ishape[1]; j++) {
                    size_t idx = i * istrides[0] + j * istrides[1];
                    if (pi[idx] != edge_tag) {
                        pf[idx * 2] = -1;
                    } else {
                        pf[idx * 2] = j;
                        pf[idx * 2 + 1] = i;
                    }
                }
            }

            // -------------------------
            // First sweep: direction 0 (y sweep)
            // -------------------------
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < ishape[1]; i++) {
                int tid = omp_get_thread_num();
                coor_locals[tid][1] = i;  // y-coordinate

                VoronoiFT(pf + i * fstrides[1], ishape[0], coor_locals[tid], rank, 0, fstrides[0], fstrides[2],
                          local_f_ptrs[tid], local_g[tid]);
            }

            // -------------------------
            // Second sweep: direction 1 (x sweep)
            // -------------------------
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < ishape[0]; i++) {
                int tid = omp_get_thread_num();
                coor_locals[tid][0] = i;  // x-coordinate

                VoronoiFT(pf + i * fstrides[0], ishape[1], coor_locals[tid], rank, 1, fstrides[1], fstrides[2],
                          local_f_ptrs[tid], local_g[tid]);
            }

            // -------------------------
            // Cleanup per-thread buffers
            // -------------------------
            for (int t = 0; t < num_threads; t++) {
                free(local_f[t]);
                free(local_g[t]);
                free(local_f_ptrs[t]);
                free(coor_locals[t]);
            }

        } else if (rank == 3) {
            omp_set_num_threads(num_threads);

            int max_dim = std::max(ishape[0], std::max(ishape[1], ishape[2]));

            // Allocate per-thread buffers
            std::vector<int *> local_f(num_threads);
            std::vector<int *> local_g(num_threads);
            std::vector<int **> local_f_ptrs(num_threads);
            std::vector<int *> coor_locals(num_threads);

            // #pragma omp parallel for num_threads(num_threads)
            // for (int t = 0; t < num_threads; t++) {
            //     local_f[t] = (int *)aligned_alloc(64, max_dim*3*sizeof(int));
            //     local_g[t] = (int *) aligned_alloc(64, max_dim*sizeof(int));
            //     coor_locals[t] = (int *) aligned_alloc(64, 3*sizeof(int));
            //     local_f_ptrs[t] = (int **) malloc(max_dim*sizeof(int*));
            //     for (int j = 0; j < max_dim; j++)
            //         local_f_ptrs[t][j] = local_f[t] + j*3;

            //     std::fill(local_f[t], local_f[t] + max_dim*3, 0);
            //     std::fill(local_g[t], local_g[t] + max_dim, 0);
            //     std::fill(coor_locals[t], coor_locals[t] + 3, 0);
            // }





            // #pragma omp parallel for  num_threads(num_threads)
            for (int t = 0; t < num_threads; t++) {
                local_f[t] = (int *)aligned_alloc(64, max_dim * 3 * sizeof(int));
                local_g[t] = (int *)aligned_alloc(64, max_dim * sizeof(int));
                coor_locals[t] = (int *)aligned_alloc(64, 3 * sizeof(int));
                // local_f[t] = (int *)malloc(max_dim * 3 * sizeof(int));
                // local_g[t] = (int *)malloc(max_dim * sizeof(int));
                // coor_locals[t] = (int *)malloc(3 * sizeof(int));
                // build pointer array for this thread
                local_f_ptrs[t] = (int **)malloc(max_dim * sizeof(int *));
                for (int j = 0; j < max_dim; j++) {
                    local_f_ptrs[t][j] = local_f[t] + j * 3;
                }
            }

            // // -------------------------
            // // NUMA-friendly first touch
            // // -------------------------
            #pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();
                std::fill(local_f[tid], local_f[tid] + max_dim * 3, 0);
                std::fill(local_g[tid], local_g[tid] + max_dim, 0);
                std::fill(coor_locals[tid], coor_locals[tid] + 3, 0);
            }

            // Initialization of pf (seed edges)
            #pragma omp parallel for collapse(2) num_threads(num_threads)
            for (int i = 0; i < ishape[0]; i++) {
                for (int j = 0; j < ishape[1]; j++) {
                    for (int k = 0; k < ishape[2]; k++) {
                        size_t idx = i * istrides[0] + j * istrides[1] + k * istrides[2];
                        if (pi[idx] != edge_tag) {
                            pf[idx * 3] = -1;
                        } else {
                            pf[idx * 3] = i;
                            pf[fstrides[3] + idx * 3] = j;
                            pf[fstrides[3] * 2 + idx * 3] = k;
                        }
                    }
                }
            }


            // direction = 0
            {
                int direction = 0;
                int x_dir = (direction + 1) % 3;
                int y_dir = (direction + 2) % 3;

#pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int i = 0; i < ishape[x_dir]; i++) {
                    for (int j = 0; j < ishape[y_dir]; j++) {
                        int tid = omp_get_thread_num();
                        int *coor_local = coor_locals[tid];

                        coor_local[direction] = 0;
                        coor_local[x_dir] = i;
                        coor_local[y_dir] = j;

                        VoronoiFT(pf + i * fstrides[x_dir] + j * fstrides[y_dir], ishape[direction], coor_local, rank,
                                  direction, fstrides[direction], 1, local_f_ptrs[tid], local_g[tid]);
                    }
                }
            }

            // direction = 1
            {
                int direction = 1;
                int x_dir = (direction + 1) % 3;
                int y_dir = (direction + 2) % 3;

#pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int i = 0; i < ishape[x_dir]; i++) {
                    for (int j = 0; j < ishape[y_dir]; j++) {
                        int tid = omp_get_thread_num();
                        int *coor_local = coor_locals[tid];

                        coor_local[direction] = 0;
                        coor_local[x_dir] = i;
                        coor_local[y_dir] = j;

                        VoronoiFT(pf + i * fstrides[x_dir] + j * fstrides[y_dir], ishape[direction], coor_local, rank,
                                  direction, fstrides[direction], 1, local_f_ptrs[tid], local_g[tid]);
                    }
                }
            }

            // direction = 2
            {
                int direction = 2;
                int x_dir = (direction + 1) % 3;
                int y_dir = (direction + 2) % 3;

#pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int i = 0; i < ishape[x_dir]; i++) {
                    for (int j = 0; j < ishape[y_dir]; j++) {
                        int tid = omp_get_thread_num();
                        int *coor_local = coor_locals[tid];

                        coor_local[direction] = 0;
                        coor_local[x_dir] = i;
                        coor_local[y_dir] = j;

                        VoronoiFT(pf + i * fstrides[x_dir] + j * fstrides[y_dir], ishape[direction], coor_local, rank,
                                  direction, fstrides[direction], 1, local_f_ptrs[tid], local_g[tid]);
                    }
                }
            }

            for (int t = 0; t < num_threads; t++) {
                free(local_f[t]);
                free(local_g[t]);
                free(local_f_ptrs[t]);
                free(coor_locals[t]);
            }
        }
    }

    void ComputeFT2D3D_single(char *pi, int *pf, int *ishape, const size_t *istrides, const size_t *fstrides,
                              int rank) {
        if (rank == 2) {
            int max_dim = std::max(ishape[0], ishape[1]);
            std::vector<int> local_f(max_dim * 2);
            std::vector<int> local_g(max_dim);
            std::vector<int *> local_f_ptrs(max_dim);
            std::array<int, 2> coor_local = {0, 0};

            for (int i = 0; i < num_threads; i++) {
                for (int j = 0; j < max_dim; j++) {
                    local_f_ptrs[i][j] = local_f[i] + j * 2;
                }
            }

            for (int i = 0; i < ishape[1]; i++) {
                for (int j = 0; j < ishape[0]; j++) {
                    size_t idx = i * istrides[1] + j * istrides[0];
                    if (pi[idx] != edge_tag) {  // non-boundary points
                        pf[idx * 2] = -1;
                    } else {
                        pf[idx * 2] = j;
                        pf[idx * 2 + 1] = i;
                    }
                }
                VoronoiFT(pf + i * fstrides[1], ishape[0], coor_local.data(), rank, 0, fstrides[0], fstrides[2],
                          local_f_ptrs.data(), local_g.data());
            }
            for (int i = 0; i < ishape[0]; i++) {
                coor_local[0] = i;
                VoronoiFT(pf + i * fstrides[0], ishape[1], coor_local.data(), rank, 1, fstrides[1], fstrides[2],
                          local_f_ptrs.data(), local_g.data());
            }

        } else if (rank == 3) {
            int max_dim = std::max(ishape[0], std::max(ishape[1], ishape[2]));
            // use malloc to allocate the memory
            int *local_f = (int *)malloc(max_dim * 3 * sizeof(int));
            int *local_g = (int *)malloc(max_dim * sizeof(int));
            int **local_f_ptrs = (int **)malloc(max_dim * sizeof(int *));
            int coor_local[3] = {0, 0, 0};
            for (int j = 0; j < max_dim; j++) {
                local_f_ptrs[j] = local_f + j * 3;
            }
            // printf("cache line allocation time = %f\n", global_timer.stop());

            int direction = 0;
            int x_dir = (direction + 1) % 3;
            int y_dir = (direction + 2) % 3;

            for (int i = 0; i < ishape[0]; i++)  // 384 2
            {
                for (int j = 0; j < ishape[1]; j++)  // 384 1
                {
                    for (int k = 0; k < ishape[2]; k++) {
                        size_t idx = i * istrides[0] + j * istrides[1] + k * istrides[2];
                        if (pi[idx] != edge_tag) {
                            pf[idx * 3] = -1;
                        } else {
                            pf[idx * 3] = i;
                            pf[1 + idx * 3] = j;
                            pf[1 * 2 + idx * 3] = k;
                        }
                    }
                }
            }

            for (int i = 0; i < ishape[x_dir]; i++)  // 384 2
            {
                for (int j = 0; j < ishape[y_dir]; j++)  // 384 1
                {
                    coor_local[direction] = 0;
                    coor_local[x_dir] = i;
                    coor_local[y_dir] = j;
                    VoronoiFT(pf + i * fstrides[x_dir] + j * fstrides[y_dir], ishape[direction], coor_local, rank,
                              direction, fstrides[direction], 1, local_f_ptrs, local_g);
                }
            }
            direction = 1;
            x_dir = (direction + 1) % 3;
            y_dir = (direction + 2) % 3;

            for (int i = 0; i < ishape[x_dir]; i++) {      // 2
                for (int j = 0; j < ishape[y_dir]; j++) {  // 0
                    coor_local[y_dir] = j;
                    coor_local[direction] = 0;
                    coor_local[x_dir] = i;
                    VoronoiFT(pf + i * fstrides[x_dir] + j * fstrides[y_dir], ishape[direction], coor_local, rank,
                              direction, fstrides[direction], 1, local_f_ptrs, local_g);
                }
            }
            direction = 2;
            x_dir = (direction + 1) % 3;
            y_dir = (direction + 2) % 3;

            for (int i = 0; i < ishape[x_dir]; i++) {
                for (int j = 0; j < ishape[y_dir]; j++) {
                    int cur_thread_id = 0;
                    coor_local[y_dir] = j;
                    coor_local[x_dir] = i;
                    coor_local[direction] = 0;
                    VoronoiFT(pf + i * fstrides[x_dir] + j * fstrides[y_dir], ishape[direction], coor_local, rank,
                              direction, fstrides[direction], 1, local_f_ptrs, local_g);
                }
            }

            free(local_f_ptrs);
            free(local_f);
            free(local_g);
        }
    }

    /* Exact euclidean feature transform, as described in: C. R. Maurer,
         Jr., R. Qi, V. Raghavan, "A linear time algorithm for computing
         exact euclidean distance transforms of binary images in arbitrary
         dimensions. IEEE Trans. PAMI 25, 265-270, 2003. */

    std::tuple<std::vector<T_distance>, std::vector<size_t>> calculate_distance_and_index(T_int *festures,
                                                                                          size_t *index_strides, int N,
                                                                                          int *feature_dims) {
        size_t size = 1;
        for (int i = 0; i < N; i++) {
            size *= feature_dims[i];
        }
        global_timer.start();
        std::vector<T_distance> distance;
        distance.resize(size);
        std::vector<size_t> indexes;
        indexes.resize(size);

        double dist = 0;
        size_t global_idx = 0;
        T_int x, y, z;
        global_timer.start();
        if (N == 2) {
#pragma omp parallel for collapse(1) num_threads(this->num_threads) private(dist, global_idx, x, y)
            for (int i = 0; i < feature_dims[0]; i++) {
                for (int j = 0; j < feature_dims[1]; j++) {
                    global_idx = i * feature_dims[1] + j;
                    x = festures[global_idx * 2];
                    y = festures[global_idx * 2 + index_strides[2]];
                    dist = (x - i) * (x - i) + (y - j) * (y - j);
                    distance[global_idx] = sqrt(dist);
                    indexes[global_idx] = x * feature_dims[1] + y;
                }
            }
        } else if (N == 3) {
            size_t d1xd2 = feature_dims[1] * feature_dims[2];
#pragma omp parallel for collapse(2) num_threads(this->num_threads) private(dist, global_idx, x, y, z)
            for (int i = 0; i < feature_dims[0]; i++) {
                for (int j = 0; j < feature_dims[1]; j++) {
                    for (int k = 0; k < feature_dims[2]; k++) {
                        global_idx = i * d1xd2 + j * feature_dims[2] + k;
                        x = festures[global_idx * 3];
                        y = festures[global_idx * 3 + index_strides[3]];
                        z = festures[global_idx * 3 + index_strides[3] * 2];
                        dist = (x - i) * (x - i) + (y - j) * (y - j) + (z - k) * (z - k);
                        distance[global_idx] = sqrt(dist);
                        indexes[global_idx] = x * d1xd2 + y * feature_dims[2] + z;
                    }
                }
            }
        }
        distance_time = global_timer.stop();
        return {std::move(distance), std::move(indexes)};
        ;
    }

    Distance_and_Index calculate_distance_and_index_(T_int *festures, size_t *index_strides, int N, int *feature_dims) {
        size_t size = 1;
        for (int i = 0; i < N; i++) {
            size *= feature_dims[i];
        }
        global_timer.start();
        std::unique_ptr<T_distance[]> distance(static_cast<T_distance *>(std::malloc(size * sizeof(T_distance))));
        std::unique_ptr<size_t[]> indexes(static_cast<size_t *>(std::malloc(size * sizeof(size_t))));
        double dist = 0;
        size_t global_idx = 0;
        T_int x, y, z;
        if (N == 2) {
#pragma omp parallel for collapse(1) num_threads(this->num_threads) private(dist, global_idx, x, y)
            for (int i = 0; i < feature_dims[0]; i++) {
                for (int j = 0; j < feature_dims[1]; j++) {
                    global_idx = i * feature_dims[1] + j;
                    x = festures[global_idx * 2];
                    y = festures[global_idx * 2 + index_strides[2]];
                    dist = (x - i) * (x - i) + (y - j) * (y - j);
                    distance[global_idx] = sqrt(dist);
                    indexes[global_idx] = x * feature_dims[1] + y;
                }
            }
        } else if (N == 3) {
            size_t d1xd2 = feature_dims[1] * feature_dims[2];
#pragma omp parallel for collapse(1) num_threads(this->num_threads) private(dist, global_idx, x, y, z)
            for (int i = 0; i < feature_dims[0]; i++) {
                for (int j = 0; j < feature_dims[1]; j++) {
                    for (int k = 0; k < feature_dims[2]; k++) {
                        global_idx = i * d1xd2 + j * feature_dims[2] + k;
                        x = festures[global_idx * 3];
                        y = festures[global_idx * 3 + index_strides[3]];
                        z = festures[global_idx * 3 + index_strides[3] * 2];
                        dist = (x - i) * (x - i) + (y - j) * (y - j) + (z - k) * (z - k);
                        distance[global_idx] = sqrt(dist);
                        indexes[global_idx] = x * d1xd2 + y * feature_dims[2] + z;
                    }
                }
            }
        }
        distance_time = global_timer.stop();
        return {std::move(distance), std::move(indexes)};
        ;
    }

    std::vector<T_distance> calculate_distance(T_int *festures, int *index_strides, int N, int *feature_dims) {
        size_t size = 1;
        for (int i = 0; i < N; i++) {
            size *= feature_dims[i];
        }

        std::vector<T_distance> distance(size);
        Timer timer;
        timer.start();
        if (N == 2) {
#pragma omp parallel for collapse(2) num_threads(this->num_threads)
            for (int i = 0; i < feature_dims[0]; i++) {
                for (int j = 0; j < feature_dims[1]; j++) {
                    size_t global_idx = i * feature_dims[1] + j;
                    T_int x = festures[global_idx * 2];
                    T_int y = festures[global_idx * 2 + index_strides[2]];
                    double dist = (x - i) * (x - i) + (y - j) * (y - j);
                    distance[global_idx] = sqrt(dist);
                }
            }
        } else if (N == 3) {
#pragma omp parallel for collapse(3) num_threads(this->num_threads)
            for (int i = 0; i < feature_dims[0]; i++) {
                for (int j = 0; j < feature_dims[1]; j++) {
                    for (int k = 0; k < feature_dims[2]; k++) {
                        size_t global_idx = i * feature_dims[1] * feature_dims[2] + j * feature_dims[2] + k;
                        T_int x = festures[global_idx * 3];
                        T_int y = festures[global_idx * 3 + index_strides[3]];
                        T_int z = festures[global_idx * 3 + index_strides[3] * 2];
                        double dist = (x - i) * (x - i) + (y - j) * (y - j) + (z - k) * (z - k);
                        distance[global_idx] = sqrt(dist);
                    }
                }
            }
        }
        distance_time = timer.stop();
        return distance;
    }

    int NI_EuclideanFeatureTransform(char *input, int *features, int N, int *dims, int num_threads = 64) {
        int ii;
        int coor[NPY_MAXDIMS], mx = 0, jj;
        int *tmp = NULL, **f = NULL, *g = NULL;
        char *pi;
        int *pf;

        pi = (input);
        pf = (features);
        for (ii = 0; ii < N; ii++) {
            coor[ii] = 0;
            if (dims[ii] > mx) {
                mx = dims[ii];
            }
        }
        std::vector<int> strides(N);
        std::vector<int> index_strides(N + 1);
        strides[N - 1] = 1;
        index_strides[N] = 1;
        if (N == 2) {
            strides[0] = (size_t)dims[1];
            index_strides[1] = (size_t)2;
            index_strides[0] = (size_t)dims[1] * 2;
        } else if (N == 3) {
            strides[1] = (size_t)dims[2];
            strides[0] = (size_t)dims[1] * dims[2];
            index_strides[2] = (size_t)dims[2];
            index_strides[1] = (size_t)dims[1] * dims[2];
            index_strides[0] = (size_t)dims[0] * dims[1] * dims[2];

        } else {
            exit(0);
        }

        /* Some temporaries */
        // f = (int **)malloc(mx * sizeof(int *));
        // g = (int *)malloc(mx * sizeof(int));
        // tmp = (int *)malloc(mx * N * sizeof(int));
        // for (jj = 0; jj < mx; jj++) {
        //     f[jj] = tmp + jj * N;
        // }

        /* First call of recursive feature transform */
        global_timer.start();
        ComputeFT2D3D(pi, pf, dims, strides.data(), index_strides.data(), N);
        edt_time = global_timer.stop();

        // calculate_distance<int, int>(features, index_strides.data(), N, dims);
        //  auto distance = calculate_distance<double, int> (features, index_strides.data(), N, dims);
        //  for(int i = 0; i < distance.size(); i++)
        //  {
        //      std::cout << "distance[" << i << "] = " << distance[i] << std::endl;
        //  }

        // free(f);
        // free(g);
        // free(tmp);

        return 0;
    }

    // do not use. the performance is terrible becase of the vector memory allocation
    std::tuple<std::vector<T_distance>, std::vector<size_t>> NI_EuclideanFeatureTransform_(char *input, int N,
                                                                                           int *dims,
                                                                                           int num_threads = 1) {
        global_timer.start();
        char *pi;
        int *pf;
        size_t input_size = 1;
        pi = (input);
        for (int ii = 0; ii < N; ii++) {
            input_size *= dims[ii];
        }
        // std::vector<int> features(input_size * N);
        int *features = (int *)malloc(input_size * N * sizeof(int));
        // pf = features.data();
        pf = features;
        std::vector<size_t> strides(N);
        std::vector<size_t> index_strides(N + 1);
        strides[N - 1] = 1;
        index_strides[N] = 1;
        if (N == 2) {
            strides[0] = (size_t)dims[1];
            index_strides[0] = (size_t)dims[1] * 2;
            index_strides[1] = (size_t)2;
        } else if (N == 3) {
            strides[1] = (size_t)dims[2];
            strides[0] = (size_t)dims[1] * dims[2];
            index_strides[0] = (size_t)dims[1] * dims[2] * 3;
            index_strides[1] = (size_t)dims[2] * 3;
            index_strides[2] = (size_t)3;
        } else {
            exit(0);
        }
        // printf("aux time = %f \n", global_timer.stop());
        global_timer.start();
        if (num_threads == 1) {
            ComputeFT2D3D_single(pi, pf, dims, strides.data(), index_strides.data(), N);
        } else {
            ComputeFT2D3D(pi, pf, dims, strides.data(), index_strides.data(), N);
        }
        edt_time = global_timer.stop();
        auto result = calculate_distance_and_index(features, index_strides.data(), N, dims);
        free(features);
        return result;
    }

    Distance_and_Index NI_EuclideanFeatureTransform(char *input, int N, int *dims, int num_threads = 1) {
        char *pi;
        int *pf;
        size_t input_size = 1;
        pi = (input);
        for (int ii = 0; ii < N; ii++) {
            input_size *= dims[ii];
        }
        // std::vector<int> features(input_size * N);
        int *features = (int *)malloc(input_size * N * sizeof(int));
        // pf = features.data();
        pf = features;
        std::vector<size_t> strides(N);
        std::vector<size_t> index_strides(N + 1);
        strides[N - 1] = 1;
        index_strides[N] = 1;
        if (N == 2) {
            strides[0] = dims[1];
            index_strides[0] = dims[1] * 2;
            index_strides[1] = 2;
        } else if (N == 3) {
            strides[1] = dims[2];
            strides[0] = dims[1] * dims[2];
            index_strides[0] = dims[1] * dims[2] * 3;
            index_strides[1] = dims[2] * 3;
            index_strides[2] = 3;
        } else {
            exit(0);
        }
        // printf("aux time = %f \n", global_timer.stop());
        global_timer.start();
        if (num_threads == 1) {
            ComputeFT2D3D(pi, pf, dims, strides.data(), index_strides.data(), N);
        } else {
            ComputeFT2D3D(pi, pf, dims, strides.data(), index_strides.data(), N);
        }
        edt_time = global_timer.stop();
        auto result = calculate_distance_and_index_(features, index_strides.data(), N, dims);
        free(features);
        return result;
    }

   private:
    /* data */
    Timer timer;
    int num_threads = 1;
    double edt_time = 0;
    double distance_time = 0;
    Timer global_timer;
    char edge_tag = 1;
};
}  // namespace PM2

#endif  // EDT_TRANSFORM_HPP