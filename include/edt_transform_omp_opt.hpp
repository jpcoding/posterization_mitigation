#ifndef EDT_TRANSFORM_OMP_OPT_HPP
#define EDT_TRANSFORM_OMP_OPT_HPP

// Optimized 3D EDT — adaptive coordinate storage type.
//
// The internal "features" buffer stores the nearest boundary (x,y,z) for each
// voxel.  With int32 (the old default) that is 12 bytes/voxel = 1.6 GB for a
// 512³ volume, written and re-read by each of the 3 VoronoiFT passes.
//
// Here the storage type is selected at runtime from the max dimension:
//   max_dim ≤ 127   → int8_t  (3 bytes/voxel, 4× smaller)
//   max_dim ≤ 32767 → int16_t (6 bytes/voxel, 2× smaller)  ← covers all
//   otherwise       → int32_t                               practical 3D data
//
// Additionally, NI_EuclideanFeatureTransform_dist_only skips allocating and
// filling the index array entirely (saves ~1 GB for 512³ when only distance
// is needed, e.g. EDT round 2 in the IDW compensation path).
//
// EDT_OMP in edt_transform_omp.hpp is kept unchanged for comparison.
// This class supports 3D only; use EDT_OMP for 2D.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <omp.h>
#include <vector>

namespace PM2 {

template <typename T_distance>
class EDT_OMP_Opt {
   public:
    struct Distance_and_Index {
        std::unique_ptr<T_distance[]> distance;
        std::unique_ptr<size_t[]>     indexes;
    };

    struct Distance_and_Packed_Index {
        std::unique_ptr<T_distance[]> distance;
        std::unique_ptr<uint32_t[]>   indexes;
    };

    void set_num_threads(int n) { num_threads = n; }

    // Full result: distance + nearest-feature flat index.
    // Needed for EDT round 1 (sign propagation uses indexes).
    Distance_and_Index NI_EuclideanFeatureTransform(char *input, int N, int *dims,
                                                    int /*unused*/ = 1) {
        if (N != 3) return {};  // 2D not supported here; use EDT_OMP
        int max_dim = *std::max_element(dims, dims + N);
        if (max_dim <= 127)   return edt_full<int8_t> (input, dims);
        if (max_dim <= 32767) return edt_full<int16_t>(input, dims);
                              return edt_full<int32_t>(input, dims);
    }

    // Full result: distance + packed nearest-feature index.
    // Packing uses 10 bits per coordinate: (x<<20)|(y<<10)|z for dims[0],dims[1],dims[2].
    // Use when all dimensions fit in 10 bits; this halves index bandwidth vs size_t.
    Distance_and_Packed_Index NI_EuclideanFeatureTransform_packed(
        char *input, int N, int *dims, int /*unused*/ = 1) {
        if (N != 3) return {};
        int max_dim = *std::max_element(dims, dims + N);
        if (max_dim > 1023) return {};
        if (max_dim <= 127)   return edt_full_packed<int8_t> (input, dims);
        if (max_dim <= 32767) return edt_full_packed<int16_t>(input, dims);
                              return edt_full_packed<int32_t>(input, dims);
    }

    // Distance-only: no index array allocated or computed.
    // Use for EDT round 2 in the IDW path where indexes are never consumed.
    std::unique_ptr<T_distance[]> NI_EuclideanFeatureTransform_dist_only(
        char *input, int N, int *dims, int /*unused*/ = 1) {
        if (N != 3) return nullptr;
        int max_dim = *std::max_element(dims, dims + N);
        if (max_dim <= 127)   return edt_dist_only<int8_t> (input, dims);
        if (max_dim <= 32767) return edt_dist_only<int16_t>(input, dims);
                              return edt_dist_only<int32_t>(input, dims);
    }

   private:
    int num_threads = 1;
    static constexpr char edge_tag = 1;

    // -----------------------------------------------------------------------
    // VoronoiFT templated on coordinate storage type TCoord.
    // Internal arithmetic uses int/double (safe for all TCoord widths).
    // pf layout: AoS — pf[voxel * stride_between_voxels + coord_idx]
    // -----------------------------------------------------------------------
    template <typename TCoord>
    static void VoronoiFT(TCoord *pf, int len, int *coor, int rank, int d,
                          size_t stride, int **f, int *g) {
        int l = -1, ii, maxl, idx1, idx2, jj;

        for (ii = 0; ii < len; ii++)
            for (jj = 0; jj < rank; jj++)
                f[ii][jj] = (int)pf[ii * stride + jj];

        for (ii = 0; ii < len; ii++) {
            if (f[ii][0] < 0) continue;  // sentinel: voxel has no nearby boundary
            double fd = f[ii][d];
            double wR = 0;
            for (jj = 0; jj < rank; jj++) {
                if (jj != d) { int tw = f[ii][jj] - coor[jj]; wR += tw * tw; }
            }
            while (l >= 1) {
                idx1 = g[l]; idx2 = g[l - 1];
                double a = f[idx1][d] - f[idx2][d];
                double b = fd - f[idx1][d];
                double c = a + b, uR = 0, vR = 0;
                for (jj = 0; jj < rank; jj++) {
                    if (jj != d) {
                        double cc = coor[jj];
                        double tu = f[idx2][jj] - cc, tv = f[idx1][jj] - cc;
                        uR += tu * tu; vR += tv * tv;
                    }
                }
                if (c * vR - b * uR - a * wR - a * b * c <= 0) break;
                --l;
            }
            g[++l] = ii;
        }
        maxl = l;
        if (maxl < 0) return;
        l = 0;
        for (ii = 0; ii < len; ii++) {
            double delta1 = 0, t;
            for (jj = 0; jj < rank; jj++) {
                t = (jj == d) ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];
                delta1 += t * t;
            }
            while (l < maxl) {
                double delta2 = 0;
                for (jj = 0; jj < rank; jj++) {
                    t = (jj == d) ? f[g[l+1]][jj] - ii : f[g[l+1]][jj] - coor[jj];
                    delta2 += t * t;
                }
                if (delta1 <= delta2) break;
                delta1 = delta2; ++l;
            }
            idx1 = g[l];
            for (jj = 0; jj < rank; jj++)
                pf[ii * stride + jj] = (TCoord)f[idx1][jj];
        }
    }

    // -----------------------------------------------------------------------
    // 3D feature transform.  Fills pf so that for each voxel (i,j,k),
    //   pf[flat(i,j,k)*3 + 0] = x-coord of nearest boundary voxel
    //   pf[flat(i,j,k)*3 + 1] = y-coord
    //   pf[flat(i,j,k)*3 + 2] = z-coord
    // fstrides[d] = step in TCoord elements between adjacent voxels along dim d.
    // -----------------------------------------------------------------------
    template <typename TCoord>
    void ComputeFT3D(char *pi, TCoord *pf, int *ishape,
                     const size_t *istrides, const size_t *fstrides) {
        // Initialise seed points; sentinel -1 marks non-boundary
        #pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 0; i < ishape[0]; i++) {
            for (int j = 0; j < ishape[1]; j++) {
                for (int k = 0; k < ishape[2]; k++) {
                    size_t idx = i * istrides[0] + j * istrides[1] + k;
                    if (pi[idx] != edge_tag) {
                        pf[idx * 3] = (TCoord)-1;
                    } else {
                        pf[idx * 3                ] = (TCoord)i;
                        pf[idx * 3 + fstrides[3]  ] = (TCoord)j;
                        pf[idx * 3 + fstrides[3]*2] = (TCoord)k;
                    }
                }
            }
        }

        int max_dim = std::max({ishape[0], ishape[1], ishape[2]});
        std::vector<int *> lf(num_threads), lg(num_threads);
        std::vector<int **> lf_ptrs(num_threads);
        std::vector<int *> lcoor(num_threads);
        for (int t = 0; t < num_threads; t++) {
            lf[t]      = (int *)aligned_alloc(64, max_dim * 3 * sizeof(int));
            lg[t]      = (int *)aligned_alloc(64, max_dim * sizeof(int));
            lcoor[t]   = (int *)aligned_alloc(64, 3 * sizeof(int));
            lf_ptrs[t] = (int **)malloc(max_dim * sizeof(int *));
            for (int j = 0; j < max_dim; j++) lf_ptrs[t][j] = lf[t] + j * 3;
        }
        // NUMA first-touch
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            std::fill(lf[tid], lf[tid] + max_dim * 3, 0);
            std::fill(lg[tid], lg[tid] + max_dim, 0);
            std::fill(lcoor[tid], lcoor[tid] + 3, 0);
        }

        for (int direction = 0; direction < 3; direction++) {
            int x_dir = (direction + 1) % 3;
            int y_dir = (direction + 2) % 3;
            #pragma omp parallel for collapse(2) num_threads(num_threads)
            for (int i = 0; i < ishape[x_dir]; i++) {
                for (int j = 0; j < ishape[y_dir]; j++) {
                    int tid = omp_get_thread_num();
                    lcoor[tid][direction] = 0;
                    lcoor[tid][x_dir]     = i;
                    lcoor[tid][y_dir]     = j;
                    VoronoiFT<TCoord>(
                        pf + i * fstrides[x_dir] + j * fstrides[y_dir],
                        ishape[direction], lcoor[tid], 3, direction,
                        fstrides[direction], lf_ptrs[tid], lg[tid]);
                }
            }
        }

        for (int t = 0; t < num_threads; t++) {
            free(lf[t]); free(lg[t]); free(lf_ptrs[t]); free(lcoor[t]);
        }
    }

    // -----------------------------------------------------------------------
    // Convert features buffer → distance (and optionally flat index).
    // out_idx may be nullptr to skip index computation entirely.
    // -----------------------------------------------------------------------
    template <typename TCoord>
    void compute_distances(const TCoord *features, const int *dims,
                           T_distance *out_dist, size_t *out_idx) {
        size_t d1xd2 = (size_t)dims[1] * dims[2];
        bool need_idx = (out_idx != nullptr);
        #pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 0; i < dims[0]; i++) {
            for (int j = 0; j < dims[1]; j++) {
                for (int k = 0; k < dims[2]; k++) {
                    size_t gi = (size_t)i * d1xd2 + (size_t)j * dims[2] + k;
                    int x = (int)features[gi * 3    ];
                    int y = (int)features[gi * 3 + 1];
                    int z = (int)features[gi * 3 + 2];
                    double dist = (double)(x-i)*(x-i) + (double)(y-j)*(y-j) + (double)(z-k)*(z-k);
                    out_dist[gi] = (T_distance)std::sqrt(dist);
                    if (need_idx)
                        out_idx[gi] = (size_t)x * d1xd2 + (size_t)y * dims[2] + z;
                }
            }
        }
    }

    template <typename TCoord>
    void compute_distances_packed(const TCoord *features, const int *dims,
                                  T_distance *out_dist, uint32_t *out_idx) {
        #pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 0; i < dims[0]; i++) {
            for (int j = 0; j < dims[1]; j++) {
                for (int k = 0; k < dims[2]; k++) {
                    size_t gi = (size_t)i * dims[1] * dims[2] + (size_t)j * dims[2] + k;
                    int x = (int)features[gi * 3    ];
                    int y = (int)features[gi * 3 + 1];
                    int z = (int)features[gi * 3 + 2];
                    double dist = (double)(x-i)*(x-i) + (double)(y-j)*(y-j) + (double)(z-k)*(z-k);
                    out_dist[gi] = (T_distance)std::sqrt(dist);
                    out_idx[gi] = ((uint32_t)x << 20) | ((uint32_t)y << 10) | (uint32_t)z;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Common setup: allocate features buffer and compute fstrides for 3D AoS.
    // fstrides[d] = TCoord elements between adjacent voxels along dim d.
    // fstrides[3] = 1 (component stride within same voxel).
    // -----------------------------------------------------------------------
    static void make_strides(const int *dims,
                             size_t istrides[3], size_t fstrides[4]) {
        istrides[0] = (size_t)dims[1] * dims[2];
        istrides[1] = (size_t)dims[2];
        istrides[2] = 1;
        fstrides[0] = (size_t)dims[1] * dims[2] * 3;  // dim-0 step
        fstrides[1] = (size_t)dims[2] * 3;             // dim-1 step
        fstrides[2] = 3;                               // dim-2 step
        fstrides[3] = 1;                               // component step
    }

    // -----------------------------------------------------------------------
    // Full pipeline: distance + index
    // -----------------------------------------------------------------------
    template <typename TCoord>
    Distance_and_Index edt_full(char *input, int *dims) {
        size_t sz = (size_t)dims[0] * dims[1] * dims[2];
        TCoord *features = (TCoord *)malloc(sz * 3 * sizeof(TCoord));

        size_t istrides[3], fstrides[4];
        make_strides(dims, istrides, fstrides);
        int ishape[3] = { dims[0], dims[1], dims[2] };
        ComputeFT3D<TCoord>(input, features, ishape, istrides, fstrides);

        Distance_and_Index result;
        result.distance = std::unique_ptr<T_distance[]>(
            (T_distance *)malloc(sz * sizeof(T_distance)));
        result.indexes  = std::unique_ptr<size_t[]>(
            (size_t *)malloc(sz * sizeof(size_t)));
        compute_distances<TCoord>(features, dims,
                                  result.distance.get(), result.indexes.get());
        free(features);
        return result;
    }

    template <typename TCoord>
    Distance_and_Packed_Index edt_full_packed(char *input, int *dims) {
        size_t sz = (size_t)dims[0] * dims[1] * dims[2];
        TCoord *features = (TCoord *)malloc(sz * 3 * sizeof(TCoord));

        size_t istrides[3], fstrides[4];
        make_strides(dims, istrides, fstrides);
        int ishape[3] = { dims[0], dims[1], dims[2] };
        ComputeFT3D<TCoord>(input, features, ishape, istrides, fstrides);

        Distance_and_Packed_Index result;
        result.distance = std::unique_ptr<T_distance[]>(
            (T_distance *)malloc(sz * sizeof(T_distance)));
        result.indexes = std::unique_ptr<uint32_t[]>(
            (uint32_t *)malloc(sz * sizeof(uint32_t)));
        compute_distances_packed<TCoord>(features, dims,
                                         result.distance.get(), result.indexes.get());
        free(features);
        return result;
    }

    // -----------------------------------------------------------------------
    // Distance-only pipeline: skips the 1-GB index array for large volumes.
    // -----------------------------------------------------------------------
    template <typename TCoord>
    std::unique_ptr<T_distance[]> edt_dist_only(char *input, int *dims) {
        size_t sz = (size_t)dims[0] * dims[1] * dims[2];
        TCoord *features = (TCoord *)malloc(sz * 3 * sizeof(TCoord));

        size_t istrides[3], fstrides[4];
        make_strides(dims, istrides, fstrides);
        int ishape[3] = { dims[0], dims[1], dims[2] };
        ComputeFT3D<TCoord>(input, features, ishape, istrides, fstrides);

        std::unique_ptr<T_distance[]> dist(
            (T_distance *)malloc(sz * sizeof(T_distance)));
        compute_distances<TCoord>(features, dims, dist.get(), nullptr);
        free(features);
        return dist;
    }
};

}  // namespace PM2

#endif  // EDT_TRANSFORM_OMP_OPT_HPP
