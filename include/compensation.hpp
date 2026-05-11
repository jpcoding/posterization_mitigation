#ifndef COMPENSATION_HPP
#define COMPENSATION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "compute_grad.hpp"
#include "edt_transform.hpp"
#include "edt_transform_omp.hpp"
#include "edt_transform_omp_opt.hpp"
#include "get_boundary.hpp"
#include "utils/file_utils.hpp"
#include "utils/smooth_distance.hpp"
#include "utils/timer.hpp"

namespace PM {
enum class CPUIndexMode {
    PackedXYZ32,
    Flat32
};

enum class WeightMode {
    IDW,        // w = d2/(d1+d2)  [default, p=1 Shepard]
    PowerIDW,   // w = d2^p/(d1^p+d2^p)  [p=2 classic Shepard]
    Smoothstep, // w = smoothstep(d2/(d1+d2)), C1-smooth, no parameters
    RBF         // two-point inverse-multiquadric RBF (needs indexes from EDT)
};

template <typename T_data, typename T_quant>
class Compensation {
   public:
    Compensation(int n, int *dims, T_data *dec_data, T_quant *quant, double compensation_value) {
        this->N = n;
        this->dec_data = dec_data;
        this->quant_index = quant;
        this->input_size = 1;
        for (int i = 0; i < n; i++) {
            this->dims.push_back(dims[i]);
            this->input_size *= dims[i];
        }
        this->strides.resize(n);
        this->strides[n - 1] = 1;
        for (int i = n - 2; i >= 0; i--) {
            this->strides[i] = this->strides[i + 1] * dims[i + 1];
        }
        this->comepnsation_value = compensation_value;
        this->compensation_map.resize(this->input_size, 0);
    }

    ~Compensation() {}

    void set_edt_thread_num(int num_threads) { this->edt_thread_num = num_threads; }

    void set_use_rbf(bool use_rbf) { this->weight_mode = use_rbf ? WeightMode::RBF : WeightMode::IDW; }
    void set_weight_mode(WeightMode m) { this->weight_mode = m; }
    void set_idw_power(double p) { this->idw_power = p; }

    void set_frequent_quant_index(T_quant index) { this->frequent_quant_index = index; }

    void set_downsample_edt_round2(bool v) { this->downsample_r2_factor = v ? 2 : 0; }
    // factor: 0=full-res (off), 2=2x, 4=4x, 8=8x downsample for EDT round 2
    void set_downsample_r2_factor(int factor) { this->downsample_r2_factor = factor; }

    // r_i = c_sign * c_geom  (both off by default — baseline unchanged)
    //
    // c_sign: scale compensation by the fraction of immediate face-neighbors
    //   with a different quant_index whose implied sign agrees with sign_map[i].
    //   If no neighbors differ, c_sign = 1 (unchallenged — handled by c_geom).
    //
    // c_geom: attenuate compensation for points far from any boundary.
    //   c_geom = min(1, geo_scale / d1), where d1 is EDT round-1 distance.
    //   Voxels with d1 <= geo_scale keep full compensation; voxels with d1 > geo_scale
    //   are attenuated proportionally. Smaller geo_scale ⇒ more attenuation.
    //   Controls harm in large homogeneous plateau regions (baryon_density etc).
    //
    // geo_auto: derive geo_scale adaptively from the d1 distribution (no original data needed).
    //   geo_scale = max(percentile(d1, geo_percentile), geo_scale_min).
    //   Defaults (validated on NYX + Hurricane × 3 ebs × 4 ds_factors = 312 configs):
    //     geo_percentile = 80, geo_scale_min = 1.0
    //   The high percentile keeps full compensation on most of the field (the bottom 80% of d1)
    //   and only attenuates the deep-plateau tail. The floor of 1 voxel prevents the
    //   percentile from collapsing on dense-boundary fields at small eb, where every voxel
    //   is within <1 voxel of an edge and a naive percentile would kill all compensation.
    void set_sign_certainty(bool v) { use_sign_certainty = v; }
    void set_geo_attenuation(bool v) { use_geo_attenuation = v; }
    void set_geo_scale(double s) { geo_scale = s; geo_auto = false; }
    void set_geo_auto(bool v) { geo_auto = v; }
    void set_geo_percentile(double p) { geo_percentile = p; }  // 0–100, default 10
    // Floor applied to the auto-derived geo_scale. The percentile of d1 can collapse to
    // near zero on dense-boundary fields at small eb (every voxel within a fraction of a
    // voxel of an edge), which makes `min(1, geo_scale/d1)` kill nearly all compensation.
    // Setting geo_scale_min to e.g. 1.0 prevents that pathology while preserving adaptivity.
    void set_geo_scale_min(double m) { geo_scale_min = m; }
    // Plateau-width attenuation: r *= min(1, plateau_cutoff / (d1+d2)).
    // When the total plateau is wider than plateau_cutoff, compensation is scaled down
    // proportionally. Addresses the large-plateau failure mode independently of c_geom
    // (which only uses d1). Default cutoff = 20 voxels.
    void set_plateau_attenuation(bool v) { use_plateau_attenuation = v; }
    void set_plateau_cutoff(double c) { plateau_cutoff = c; }
    // Skip compensation entirely if boundary fraction < threshold (default 0.001).
    void set_edge_density_threshold(double t) { edge_density_threshold = t; }
    // Skip compensation if field sparsity (fraction of non-mode voxels) < threshold (default 0.01).
    // When sparsity < 0.01, non-mode voxels form incoherent scatter rather than coherent regions,
    // making EDT sign propagation unreliable at typical d1 distances.
    void set_sparsity_threshold(double t) { sparsity_threshold = t; }

    // Re-quantization self-validation clamp (Approach 4 from notes.md).
    // Provides a hard per-voxel guarantee: the compensated value dec_data[i]+comp[i] is
    // clamped to stay inside the original quantization bin [2k-1, 2k+1]*eb where
    // k = quant_index[i]. With the canonical bin-center decompressor (dec_data[i] = 2*eb*k),
    // this reduces to |comp[i]| <= eb. Zero compute cost; enables safely raising eta above 1.
    // Requires set_eb() to be called with the absolute error bound.
    void set_requant_clamp(bool v) { use_requant_clamp = v; }
    void set_eb(double v) { eb = v; }

    // Profiling: when enabled, after EDT rounds 1 and 2 the per-voxel distance arrays
    // are copied into member buffers and accessible via get_d_edge() / get_d_neutral().
    // Used by --profile_harm in the test driver to study where false corrections occur.
    // Adds ~2*sizeof(T_data)*N memory at peak (~256 MB at 512^3 float).
    void set_save_distance_maps(bool v) { save_distance_maps = v; }
    const std::vector<T_data>& get_d_edge() const { return d_edge_map; }
    const std::vector<T_data>& get_d_neutral() const { return d_neutral_map; }

    void set_cpu_index_mode(CPUIndexMode mode) { this->cpu_index_mode = mode; }

    template <typename T_data_sign>
    char get_sign(T_data_sign data) {
        char sign = (char)(((double)data > 0.0) - ((double)data < 0.0));
        return sign;
    }

    std::tuple<std::array<char, 4>, std::array<int, 4>> check_compensate_direction_distance_2d(size_t index) {
        int x, y;
        int tx, ty;
        char left, right, up, down;
        int d_left, d_right, d_up, d_down;
        x = index / this->dims[1];
        y = index % this->dims[1];
        int cur_quant_index = this->quant_index[index];

        tx = x, ty = y - 1;
        while (ty > 0) {
            int cur_idx = tx * this->dims[1] + ty;
            if (this->quant_index[cur_idx] != cur_quant_index) {
                left = get_sign(cur_quant_index - this->quant_index[cur_idx]);
                break;
            }
            ty--;
        }
        d_left = y - ty - 1;

        tx = x, ty = y + 1;
        while (ty < this->dims[1] - 1) {
            int cur_idx = tx * this->dims[1] + ty;
            if (this->quant_index[cur_idx] != cur_quant_index) {
                right = get_sign(this->quant_index[cur_idx] - cur_quant_index);
                break;
            }
            ty++;
        }
        d_right = ty - y - 1;
        tx = x - 1, ty = y;
        while (tx > 0) {
            int cur_idx = tx * this->dims[1] + ty;
            if (this->quant_index[cur_idx] != cur_quant_index) {
                up = get_sign(cur_quant_index - this->quant_index[cur_idx]);
                break;
            }
            tx--;
        }
        d_up = x - tx - 1;
        tx = x + 1, ty = y;
        while (tx < this->dims[0] - 1) {
            int cur_idx = tx * this->dims[1] + ty;
            if (this->quant_index[cur_idx] != cur_quant_index) {
                down = get_sign(this->quant_index[cur_idx] - cur_quant_index);
                break;
            }
            tx++;
        }
        d_down = tx - x - 1;
        std::array<char, 4> compensate_direction{left, right, up, down};
        std::array<int, 4> change_distance{d_left, d_right, d_up, d_down};
        return std::make_tuple(compensate_direction, change_distance);
    }

    std::tuple<std::array<char, 6>, std::array<int, 6>> check_compensate_direction_distance_3d(size_t index) {
        int x, y, z;
        int tx, ty, tz;
        char left = 0, right = 0, up = 0, down = 0, front = 0, back = 0;
        int d_left = 0, d_right = 0, d_up = 0, d_down = 0, d_front = 0, d_back = 0;
        x = index / (dims[1] * dims[2]);  // slowest dim
        y = (index / dims[2]) % dims[1];
        z = index % dims[2];  // fastest dim
        int cur_quant_index = quant_index[index];

        tx = x, ty = y - 1, tz = z;
        while (ty > 0) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                left = get_sign(cur_quant_index - quant_index[cur_idx]);
                break;
            }
            ty--;
        }
        d_left = y - ty - 1;
        tx = x, ty = y + 1, tz = z;
        while (ty < dims[1]) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                right = get_sign(quant_index[cur_idx] - cur_quant_index);
                break;
            }
            ty++;
        }
        d_right = ty - y - 1;
        tx = x, ty = y, tz = z - 1;
        while (tz > 0) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                up = get_sign(cur_quant_index - quant_index[cur_idx]);
                break;
            }
            tz--;
        }
        d_up = z - tz - 1;
        tx = x, ty = y, tz = z + 1;
        while (tz < dims[2]) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                down = get_sign(quant_index[cur_idx] - cur_quant_index);
                break;
            }
            tz++;
        }
        d_down = tz - z - 1;
        tx = x - 1, ty = y, tz = z;
        while (tx > 0) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                front = get_sign(cur_quant_index - quant_index[cur_idx]);
                break;
            }
            tx--;
        }
        d_front = x - tx - 1;
        tx = x + 1, ty = y, tz = z;
        while (tx < dims[0]) {
            int cur_idx = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (quant_index[cur_idx] != cur_quant_index) {
                back = get_sign(quant_index[cur_idx] - cur_quant_index);
                break;
            }
            tx++;
        }
        d_back = tx - x - 1;
        std::array<char, 6> compensate_direction{left, right, up, down, front, back};
        std::array<int, 6> change_distance{d_left, d_right, d_up, d_down, d_front, d_back};
        return std::make_tuple(compensate_direction, change_distance);
    }

    double find_opposite_distance_2d(double *distance_array, size_t *index_array, char *boundary_map, size_t cur_index,
                                     size_t near_index, T_data *compensation_map, int max_extend = 10) {
        int x, y;
        x = cur_index / dims[1];
        y = cur_index % dims[1];
        char cur_sign = get_sign(compensation_map[near_index]);
        double dx, dy;
        int near_x, near_y;
        near_x = near_index / dims[1];
        near_y = near_index % dims[1];
        dx = near_x - x;
        dy = near_y - y;

        double norm = std::sqrt(dx * dx + dy * dy);

        dx = dx * 1.0 / norm;
        dy = dy * 1.0 / norm;
        int tx, ty;
        int max_steps = *std::max_element(dims.begin(), dims.end());
        double distance_to_opposite_boundary = 0;
        for (int i = 1; i < max_steps; i++) {
            tx = std::round(x - i * dx);
            ty = std::round(y - i * dy);
            size_t global_index = tx * dims[1] + ty;
            if (tx < 0 || tx >= dims[0] || ty < 0 || ty >= dims[1]) {
                break;
            }
            char next_sign = get_sign(compensation_map[index_array[global_index]]);
            if (next_sign != cur_sign) {
                // distance_to_opposite_boundary = sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y));
                return distance_to_opposite_boundary + 1;
            }
            distance_to_opposite_boundary = sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y));
        }
        distance_to_opposite_boundary = sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y));

        return distance_to_opposite_boundary;
    }

    double find_opposite_distance_3d(double *distance_array, size_t *index_array, char *boundary_mao, size_t cur_index,
                                     size_t near_index, T_data *compensation_map, int max_extend = 10) {
        int x, y, z;
        x = cur_index / (dims[1] * dims[2]);
        y = (cur_index / dims[2]) % dims[1];
        z = cur_index % dims[2];
        char cur_sign = get_sign(compensation_map[near_index]);
        double dx, dy, dz;
        int near_x, near_y, near_z;
        near_x = near_index / (dims[1] * dims[2]);
        near_y = (near_index / dims[2]) % dims[1];
        near_z = near_index % dims[2];
        dx = near_x - x;
        dy = near_y - y;
        dz = near_z - z;
        double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
        dx = dx / norm;
        dy = dy / norm;
        dz = dz / norm;
        int max_steps = *std::max_element(dims.begin(), dims.end());
        double distance_to_opposite_boundary = 0;
        int tx, ty, tz;
        for (int i = 1; i < max_steps; i++) {
            tx = int(x - i * dx);
            ty = int(y - i * dy);
            tz = int(z - i * dz);
            size_t global_index = tx * dims[1] * dims[2] + ty * dims[2] + tz;
            if (tx < 0 || tx >= dims[0] || ty < 0 || ty >= dims[1] || tz < 0 || tz >= dims[2]) {
                break;
            }
            char next_sign = get_sign(compensation_map[index_array[global_index]]);
            if (next_sign != cur_sign and boundary_mao[global_index] == 0) {
                // distance_to_opposite_boundary =
                // sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y) + 1.0 * (tz - z) * (tz - z));
                return distance_to_opposite_boundary + 1;
                // return distance_to_opposite_boundary;
            }
            distance_to_opposite_boundary =
                sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y) + 1.0 * (tz - z) * (tz - z));
        }
        // distance_to_opposite_boundary = 0;
        distance_to_opposite_boundary = sqrt(1.0 * (tx - x) * (tx - x) + 1.0 * (ty - y) * (ty - y));

        return distance_to_opposite_boundary;
    }

    std::vector<T_data> get_compensation_map_2d() {
        auto bounday_and_sign = get_boundary_and_sign_map_2d(quant_index, N, dims.data(), edt_thread_num);
        auto boundary_map = std::get<0>(bounday_and_sign);
        auto sign_map = std::get<1>(bounday_and_sign);
        char edge_tag = 1;

        double num_edges = 0;
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == edge_tag) {
                num_edges++;
            }
        }
        if (num_edges / input_size < 0.001) {
            return compensation_map;
        }

        auto timer = Timer();

        timer.start();
        auto edt_omp = PM2::EDT_OMP<T_data, int>();
        edt_omp.set_num_threads(edt_thread_num);
        auto edt_result = edt_omp.NI_EuclideanFeatureTransform(boundary_map.data(), N, dims.data(), edt_thread_num);
        auto distance_array = std::move(edt_result.distance);
        auto indexes = std::move(edt_result.indexes);
#pragma omp parallel for num_threads(edt_thread_num)
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] != edge_tag)  // non-boundary points ·
            {
                sign_map[i] = sign_map[indexes[i]];
            }
        }
        auto boundary_map2 = get_boundary(sign_map.data(), N, dims.data());

#pragma omp parallel for num_threads(edt_thread_num)
        for (int i = 0; i < input_size; i++) {
            if (boundary_map2[i] == edge_tag && boundary_map[i] == edge_tag) {
                boundary_map2[i] = 0;  // boundary lable
            }
        }


        auto rbf = [](double r) -> double {
            // return std::exp(-0.3*r);
            // return (1/r) / (1/r + 1); //?
            return 1 / sqrt(1 + r * r);  // inverse_multiquadric
            // cubic, thin-plate, gaussian, multiquadric, inverse multiquadric
            // thin-plate:
            // return r*r * log(r);
        };
        // calculate the distance between two points // 2d cases
        auto cal_distance = [this](int i, int j) -> double {
            int x1 = i / (dims[1]);
            int y1 = i % (dims[1]);
            int x2 = j / (dims[1]);
            int y2 = j % (dims[1]);
            return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        };

        timer.start();
        edt_omp.reset_timer();
        auto edt_result2 = edt_omp.NI_EuclideanFeatureTransform(boundary_map2.data(), N, dims.data(), edt_thread_num);
        auto distance_array2 = std::move(edt_result2.distance);
        auto indexes2 = std::move(edt_result2.indexes);

        if (weight_mode == WeightMode::RBF) {
            #pragma omp parallel for num_threads(edt_thread_num)
            for (size_t i = 0; i < input_size; i++) {
                double distance1 = distance_array[i] + 0.5;
                double distance2 = distance_array2[i] + 0.5;
                char sign = sign_map[i];
                if (1) {
                    // double compensation_value = 0;
                    double d0 = cal_distance(indexes[i], indexes2[i]);
                    double a = rbf(0);
                    double b = rbf(d0);
                    double w0 = a / (a * a - b * b);
                    double w1 = b / (-a * a + b * b);
                    double sx = (w0 * rbf(distance1) + w1 * rbf(distance2));
                    // clamp sx between [-1 1]
                    if (sx > 1) {
                        sx = 1;
                    } else if (sx < -1) {
                        sx = -1;
                    }
                    compensation_map[i] = comepnsation_value * sx * sign;
                } else {
                    double magnitude = (1 / distance1) / (1 / distance1 + 1 / distance2);
                    compensation_map[i] = sign * magnitude * comepnsation_value;
                }
            }
        } else {
            #pragma omp parallel for num_threads(edt_thread_num)
            for (size_t i = 0; i < input_size; i++) {
                if (1) {
                    double distance1 = distance_array[i] + 0.5;
                    double distance2 = distance_array2[i] + 0.5;
                    char sign = sign_map[i];
                    double width = distance2 + distance1;
                    double magnitude = (1 / distance1) / (1 / distance1 + 1 / distance2);
                    compensation_map[i] = sign * magnitude * comepnsation_value;
                }
            }
        }

        return compensation_map;
    }

    std::vector<T_data> get_compensation_map_3d() {
        if (use_requant_clamp && !(eb > 0.0)) {
            std::cerr << "Warning: --requant_clamp enabled but eb<=0 (forgot set_eb?); "
                         "clamp disabled for this run." << std::endl;
            use_requant_clamp = false;
        }
        Timer stage_timer;
        double stage_boundary = 0.0;
        double stage_edt1 = 0.0;
        double stage_fill_sign = 0.0;
        double stage_neutral_boundary = 0.0;
        double stage_downsample_boundary = 0.0;
        double stage_edt2 = 0.0;
        double stage_comp = 0.0;

        stage_timer.start();
        auto bounday_and_sign = get_boundary_and_sign_map_3d(quant_index, N, dims.data(), edt_thread_num);
        stage_boundary = stage_timer.stop();
        auto boundary_map = std::get<0>(bounday_and_sign);
        auto sign_map = std::get<1>(bounday_and_sign);
        char edge_tag = 1;

        // Sparsity guard: sparsity = fraction of non-mode voxels.
        // When sparsity < threshold, non-mode voxels form incoherent scatter rather than
        // coherent regions, so EDT sign propagation is unreliable at typical d1 distances.
        {
            size_t count = 0;
            for (size_t i = 0; i < input_size; i++)
                if (quant_index[i] == frequent_quant_index) count++;
            double sparsity = 1.0 - (double)count / input_size;
            std::cout << "Sparsity " << sparsity << std::endl;
            if (sparsity < sparsity_threshold) {
                std::cout << "Sparsity " << sparsity
                          << " < " << sparsity_threshold << ", skipping compensation" << std::endl;
                return compensation_map;  // all-zero
            }
        }

        // Edge-density guard: if fewer than 0.1% of voxels are on a boundary the EDT
        // sign propagation spans huge distances and is unreliable → skip compensation.
        // (Same guard as get_compensation_map_2d.)
        {
            size_t num_edges = 0;
            for (size_t i = 0; i < input_size; i++)
                if (boundary_map[i] == edge_tag) num_edges++;
            double edge_density = (double)num_edges / input_size;
            std::cout << "EdgeDensity " << edge_density << std::endl;
            if (edge_density < edge_density_threshold) {
                std::cout << "EdgeDensity " << edge_density
                          << " < " << edge_density_threshold << ", skipping compensation" << std::endl;
                return compensation_map;  // all-zero
            }
        }

        // EDT_OMP_Opt auto-selects int8/int16/int32 coord storage based on max dim,
        // halving or quartering the features buffer bandwidth vs the old EDT_OMP.
        auto edt_omp = PM2::EDT_OMP_Opt<T_data>();
        edt_omp.set_num_threads(edt_thread_num);
        const int max_dim = *std::max_element(dims.begin(), dims.end());
        const bool can_use_packed_indexes = (max_dim <= 1023);
        const bool can_use_flat32_indexes = (input_size <= std::numeric_limits<uint32_t>::max());
        const bool use_packed_indexes =
            (cpu_index_mode == CPUIndexMode::PackedXYZ32) && can_use_packed_indexes;
        const bool use_flat32_indexes =
            (cpu_index_mode == CPUIndexMode::Flat32) && can_use_flat32_indexes;
        std::unique_ptr<T_data[]> distance_array;
        std::unique_ptr<size_t[]> indexes;
        std::unique_ptr<uint32_t[]> uint32_indexes;
        stage_timer.start();
        if (use_packed_indexes) {
            auto edt_result = edt_omp.NI_EuclideanFeatureTransform_packed(
                boundary_map.data(), N, dims.data(), edt_thread_num);
            distance_array = std::move(edt_result.distance);
            uint32_indexes = std::move(edt_result.indexes);
        } else if (use_flat32_indexes) {
            auto edt_result = edt_omp.NI_EuclideanFeatureTransform_flat32(
                boundary_map.data(), N, dims.data(), edt_thread_num);
            distance_array = std::move(edt_result.distance);
            uint32_indexes = std::move(edt_result.indexes);
        } else {
            auto edt_result = edt_omp.NI_EuclideanFeatureTransform(
                boundary_map.data(), N, dims.data(), edt_thread_num);
            distance_array = std::move(edt_result.distance);
            indexes = std::move(edt_result.indexes);
        }
        stage_edt1 = stage_timer.stop();

        // Profile hook: save d1 (round-1 EDT) and allocate d2 buffer to be filled inside comp loop.
        if (save_distance_maps) {
            d_edge_map.assign(distance_array.get(), distance_array.get() + input_size);
            d_neutral_map.assign(input_size, (T_data)0);
        }

        // Adaptive geo_scale: derive from d1 percentile — no original data needed.
        // For sparse-boundary fields (baryon_density): d1_p10 is large → aggressive attenuation.
        // For dense-boundary fields (velocity): d1_p10 is small → c_geom ≈ 1 (minimal effect).
        if (use_geo_attenuation && geo_auto) {
            // O(n) histogram over [0, max_d1] with 4096 bins to find percentile.
            double max_d1 = 0.0;
            for (size_t i = 0; i < input_size; i++)
                if (distance_array[i] > max_d1) max_d1 = distance_array[i];
            if (max_d1 > 0.0) {
                const int NBINS = 4096;
                std::vector<size_t> hist(NBINS, 0);
                double inv_bin = (NBINS - 1) / max_d1;
                for (size_t i = 0; i < input_size; i++) {
                    int b = (int)(distance_array[i] * inv_bin);
                    if (b >= NBINS) b = NBINS - 1;
                    hist[b]++;
                }
                size_t target = (size_t)(geo_percentile / 100.0 * input_size);
                size_t cum = 0;
                for (int b = 0; b < NBINS; b++) {
                    cum += hist[b];
                    if (cum >= target) {
                        geo_scale = (b + 1.0) / inv_bin;
                        break;
                    }
                }
            }
            if (geo_scale_min > 0.0 && geo_scale < geo_scale_min) {
                std::cout << "AdaptiveGeoScale (p" << geo_percentile << "): " << geo_scale
                          << "  floored to " << geo_scale_min << std::endl;
                geo_scale = geo_scale_min;
            } else {
                std::cout << "AdaptiveGeoScale (p" << geo_percentile << "): " << geo_scale << std::endl;
            }
        }

        auto packed_to_flat = [this](uint32_t packed) -> size_t {
            size_t x = (size_t)((packed >> 20) & 0x3FFu);
            size_t y = (size_t)((packed >> 10) & 0x3FFu);
            size_t z = (size_t)(packed & 0x3FFu);
            return x * (size_t)dims[1] * dims[2] + y * dims[2] + z;
        };

        stage_timer.start();
#pragma omp parallel for num_threads(edt_thread_num)
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] != edge_tag)  // non-boundary points ·
            {
                if (use_packed_indexes) {
                    sign_map[i] = sign_map[packed_to_flat(uint32_indexes[i])];
                } else if (use_flat32_indexes) {
                    sign_map[i] = sign_map[uint32_indexes[i]];
                } else {
                    sign_map[i] = sign_map[indexes[i]];
                }
            }
        }
        stage_fill_sign = stage_timer.stop();

        // dump the sign map

        stage_timer.start();
        // exclude_mask = boundary_map: positions that are on the original boundary are
        // excluded from boundary_map2, keeping only "neutral" (sign-change) boundaries
        auto boundary_map2 = get_boundary(sign_map.data(), N, dims.data(), edt_thread_num,
                                          boundary_map.data());
        stage_neutral_boundary = stage_timer.stop();

        auto rbf = [](double r) -> double {
            return 1 / sqrt(1 + r * r * 1.0);  // inverse_multiquadric
        };
        auto cal_distance = [this](int i, int j) -> double {
            int x1 = i / (dims[1] * dims[2]);
            int y1 = (i / dims[2]) % dims[1];
            int z1 = i % dims[2];
            int x2 = j / (dims[1] * dims[2]);
            int y2 = (j / dims[2]) % dims[1];
            int z2 = j % dims[2];
            return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
        };
        auto cal_distance_packed = [](uint32_t a, uint32_t b) -> double {
            int ax = (int)((a >> 20) & 0x3FFu);
            int ay = (int)((a >> 10) & 0x3FFu);
            int az = (int)(a & 0x3FFu);
            int bx = (int)((b >> 20) & 0x3FFu);
            int by = (int)((b >> 10) & 0x3FFu);
            int bz = (int)(b & 0x3FFu);
            return std::sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz));
        };
        auto cal_distance_flat32 = [this](uint32_t a, uint32_t b) -> double {
            size_t x1 = (size_t)a / ((size_t)dims[1] * dims[2]);
            size_t y1 = ((size_t)a / dims[2]) % dims[1];
            size_t z1 = (size_t)a % dims[2];
            size_t x2 = (size_t)b / ((size_t)dims[1] * dims[2]);
            size_t y2 = ((size_t)b / dims[2]) % dims[1];
            size_t z2 = (size_t)b % dims[2];
            double dx = (double)x1 - x2;
            double dy = (double)y1 - y2;
            double dz = (double)z1 - z2;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        };

        // Weight function: maps (d1, d2) → magnitude ∈ (0,1), always bounded without clamping.
        auto compute_weight = [this](double d1, double d2) -> double {
            switch (weight_mode) {
                case WeightMode::PowerIDW: {
                    double a = std::pow(d1, idw_power);
                    double b = std::pow(d2, idw_power);
                    return b / (a + b);
                }
                case WeightMode::Smoothstep: {
                    double t = d2 / (d1 + d2);  // IDW weight ∈ (0,1)
                    return t * t * (3.0 - 2.0 * t);
                }
                default:  // IDW
                    return d2 / (d1 + d2);
            }
        };

        if (downsample_r2_factor >= 2 && weight_mode != WeightMode::RBF) {
            // --- Downsampled EDT round 2 (factor×factor×factor logical-OR downsample) ---
            const int F = downsample_r2_factor;
            int ds_dims[3] = {(dims[0] + F - 1) / F, (dims[1] + F - 1) / F, (dims[2] + F - 1) / F};
            size_t ds_size = (size_t)ds_dims[0] * ds_dims[1] * ds_dims[2];

            // F×F×F logical-OR downsample of boundary_map2 (output-driven, OMP-parallel)
            stage_timer.start();
            std::vector<char> ds_boundary(ds_size, 0);
            const size_t s0 = (size_t)dims[1] * dims[2], s1 = dims[2];
            const size_t ds_s0 = (size_t)ds_dims[1] * ds_dims[2], ds_s1 = ds_dims[2];
#pragma omp parallel for collapse(2) num_threads(edt_thread_num)
            for (int dx = 0; dx < ds_dims[0]; dx++) {
                for (int dy = 0; dy < ds_dims[1]; dy++) {
                    const int x_end = std::min(dx * F + F, dims[0]);
                    const int y_end = std::min(dy * F + F, dims[1]);
                    for (int dz = 0; dz < ds_dims[2]; dz++) {
                        const int z_end = std::min(dz * F + F, dims[2]);
                        char val = 0;
                        for (int fx = dx * F; fx < x_end && !val; fx++)
                            for (int fy = dy * F; fy < y_end && !val; fy++)
                                for (int fz = dz * F; fz < z_end && !val; fz++)
                                    if (boundary_map2[(size_t)fx * s0 + fy * s1 + fz]) val = 1;
                        ds_boundary[dx * ds_s0 + dy * ds_s1 + dz] = val;
                    }
                }
            }
            stage_downsample_boundary = stage_timer.stop();

            // EDT on the downsampled volume (dist-only: indexes from ds EDT are unused)
            stage_timer.start();
            auto edt_ds = PM2::EDT_OMP_Opt<double>();
            edt_ds.set_num_threads(edt_thread_num);
            auto ds_distance = edt_ds.NI_EuclideanFeatureTransform_dist_only(ds_boundary.data(), N, ds_dims, edt_thread_num);
            stage_edt2 = stage_timer.stop();

            // IDW compensation with trilinear interpolation of downsampled d_neutral
            stage_timer.start();
            const double Fd = (double)F;
            size_t n_clamped = 0;
#pragma omp parallel for collapse(2) num_threads(edt_thread_num) reduction(+:n_clamped)
            for (int x = 0; x < dims[0]; x++) {
            for (int y = 0; y < dims[1]; y++) {
            for (int z = 0; z < dims[2]; z++) {
                size_t i = (size_t)x * s0 + y * s1 + z;
                {
                double d1 = distance_array[i] + 0.5;
                char sign = sign_map[i];

                // Map full-res coord to downsampled coord
                // DS voxel center j is at full-res F*j+(F-1)/2, so fx = (x+0.5)/F - 0.5
                double fx = (x + 0.5) / Fd - 0.5;
                double fy = (y + 0.5) / Fd - 0.5;
                double fz = (z + 0.5) / Fd - 0.5;

                int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy), z0 = (int)std::floor(fz);
                double wx = fx - x0, wy = fy - y0, wz = fz - z0;

                int x1 = std::min(x0 + 1, ds_dims[0] - 1); x0 = std::max(x0, 0);
                int y1 = std::min(y0 + 1, ds_dims[1] - 1); y0 = std::max(y0, 0);
                int z1 = std::min(z0 + 1, ds_dims[2] - 1); z0 = std::max(z0, 0);

                auto ds_idx = [&](int a, int b, int c) -> size_t {
                    return (size_t)a * ds_dims[1] * ds_dims[2] + b * ds_dims[2] + c;
                };

                double d2_raw = ds_distance[ds_idx(x0,y0,z0)] * (1-wx)*(1-wy)*(1-wz)
                              + ds_distance[ds_idx(x1,y0,z0)] * wx   *(1-wy)*(1-wz)
                              + ds_distance[ds_idx(x0,y1,z0)] * (1-wx)*wy   *(1-wz)
                              + ds_distance[ds_idx(x1,y1,z0)] * wx   *wy   *(1-wz)
                              + ds_distance[ds_idx(x0,y0,z1)] * (1-wx)*(1-wy)*wz
                              + ds_distance[ds_idx(x1,y0,z1)] * wx   *(1-wy)*wz
                              + ds_distance[ds_idx(x0,y1,z1)] * (1-wx)*wy   *wz
                              + ds_distance[ds_idx(x1,y1,z1)] * wx   *wy   *wz;
                // Scale ds units → full-res, then add the standard +0.5 offset
                double d2 = d2_raw * Fd + 0.5;
                if (save_distance_maps) d_neutral_map[i] = (T_data)d2;

                double magnitude = compute_weight(d1, d2);
                double r = 1.0;
                if (use_sign_certainty) {
                    int cur_q = quant_index[i];
                    int n_differ = 0, n_agree = 0;
                    const int ddx[6] = {-1,1,0,0,0,0};
                    const int ddy[6] = {0,0,-1,1,0,0};
                    const int ddz[6] = {0,0,0,0,-1,1};
                    for (int d = 0; d < 6; d++) {
                        int nx2 = x + ddx[d], ny2 = y + ddy[d], nz2 = z + ddz[d];
                        if (nx2 < 0 || nx2 >= dims[0] || ny2 < 0 || ny2 >= dims[1] || nz2 < 0 || nz2 >= dims[2]) continue;
                        int nq = quant_index[(size_t)nx2 * s0 + ny2 * s1 + nz2];
                        if (nq != cur_q) {
                            n_differ++;
                            char implied = (cur_q > nq) ? (char)1 : (char)-1;
                            if (implied == sign) n_agree++;
                        }
                    }
                    if (n_differ > 0) r *= (double)n_agree / n_differ;
                }
                if (use_geo_attenuation) {
                    r *= std::min(1.0, geo_scale / d1);
                }
                if (use_plateau_attenuation) {
                    r *= std::min(1.0, plateau_cutoff / (d1 + d2));
                }
                compensation_map[i] = sign * r * magnitude * comepnsation_value;
                if (use_requant_clamp) {
                    // Approach 4: clamp dec_data[i]+comp[i] to original quant bin [2k-1, 2k+1]*eb
                    double bin_center = 2.0 * eb * (double)quant_index[i];
                    double low = bin_center - eb;
                    double high = bin_center + eb;
                    double v = (double)dec_data[i] + (double)compensation_map[i];
                    if (v < low) {
                        compensation_map[i] = (T_data)(low - (double)dec_data[i]);
                        n_clamped++;
                    } else if (v > high) {
                        compensation_map[i] = (T_data)(high - (double)dec_data[i]);
                        n_clamped++;
                    }
                }
                } // inner block
            }}}   // z, y, x
            stage_comp = stage_timer.stop();
            if (use_requant_clamp) {
                std::cout << "RequantClamp (ds-IDW): " << n_clamped
                          << " / " << input_size
                          << " (" << (100.0 * n_clamped / input_size) << "%)" << std::endl;
            }
        } else if (weight_mode == WeightMode::RBF) {
            // --- Full-resolution EDT round 2, RBF path (needs indexes2 for cal_distance) ---
            std::unique_ptr<T_data[]> distance_array2;
            std::unique_ptr<size_t[]> indexes2;
            std::unique_ptr<uint32_t[]> uint32_indexes2;
            stage_timer.start();
            if (use_packed_indexes) {
                auto edt_result2 = edt_omp.NI_EuclideanFeatureTransform_packed(
                    boundary_map2.data(), N, dims.data(), edt_thread_num);
                distance_array2 = std::move(edt_result2.distance);
                uint32_indexes2 = std::move(edt_result2.indexes);
            } else if (use_flat32_indexes) {
                auto edt_result2 = edt_omp.NI_EuclideanFeatureTransform_flat32(
                    boundary_map2.data(), N, dims.data(), edt_thread_num);
                distance_array2 = std::move(edt_result2.distance);
                uint32_indexes2 = std::move(edt_result2.indexes);
            } else {
                auto edt_result2 = edt_omp.NI_EuclideanFeatureTransform(
                    boundary_map2.data(), N, dims.data(), edt_thread_num);
                distance_array2 = std::move(edt_result2.distance);
                indexes2 = std::move(edt_result2.indexes);
            }
            stage_edt2 = stage_timer.stop();
            stage_timer.start();
            size_t n_clamped = 0;
#pragma omp parallel for num_threads(edt_thread_num) reduction(+:n_clamped)
            for (size_t i = 0; i < input_size; i++) {
                double distance1 = distance_array[i] + 0.5;
                double distance2 = distance_array2[i] + 0.5;
                char sign;
                double d0;
                if (use_packed_indexes) {
                    sign = sign_map[packed_to_flat(uint32_indexes[i])];
                    d0 = cal_distance_packed(uint32_indexes[i], uint32_indexes2[i]);
                } else if (use_flat32_indexes) {
                    sign = sign_map[uint32_indexes[i]];
                    d0 = cal_distance_flat32(uint32_indexes[i], uint32_indexes2[i]);
                } else {
                    sign = sign_map[indexes[i]];
                    d0 = cal_distance(indexes[i], indexes2[i]);
                }
                double a = rbf(0);
                double b = rbf(d0);
                double w0 = a / (a * a - b * b);
                double w1 = b / (-a * a + b * b);
                double sx = (w0 * rbf(distance1) + w1 * rbf(distance2));
                if (sx > 1) sx = 1;
                else if (sx < -1) sx = -1;
                compensation_map[i] = comepnsation_value * sx * sign;
                if (use_requant_clamp) {
                    double bin_center = 2.0 * eb * (double)quant_index[i];
                    double low = bin_center - eb;
                    double high = bin_center + eb;
                    double v = (double)dec_data[i] + (double)compensation_map[i];
                    if (v < low) {
                        compensation_map[i] = (T_data)(low - (double)dec_data[i]);
                        n_clamped++;
                    } else if (v > high) {
                        compensation_map[i] = (T_data)(high - (double)dec_data[i]);
                        n_clamped++;
                    }
                }
            }
            stage_comp = stage_timer.stop();
            if (use_requant_clamp) {
                std::cout << "RequantClamp (RBF): " << n_clamped
                          << " / " << input_size
                          << " (" << (100.0 * n_clamped / input_size) << "%)" << std::endl;
            }
        } else {
            // --- Full-resolution EDT round 2, IDW path (dist-only: saves ~1 GB index array) ---
            stage_timer.start();
            auto distance_array2 = edt_omp.NI_EuclideanFeatureTransform_dist_only(
                boundary_map2.data(), N, dims.data(), edt_thread_num);
            stage_edt2 = stage_timer.stop();
            stage_timer.start();
            const size_t r2_s0 = (size_t)dims[1] * dims[2], r2_s1 = dims[2];
            size_t n_clamped = 0;
#pragma omp parallel for collapse(2) num_threads(edt_thread_num) reduction(+:n_clamped)
            for (int x = 0; x < dims[0]; x++) {
            for (int y = 0; y < dims[1]; y++) {
            for (int z = 0; z < dims[2]; z++) {
                size_t i = (size_t)x * r2_s0 + y * r2_s1 + z;
                double distance1 = distance_array[i] + 0.5;
                double distance2 = distance_array2[i] + 0.5;
                if (save_distance_maps) d_neutral_map[i] = (T_data)distance2;
                char sign = sign_map[i];
                double magnitude = compute_weight(distance1, distance2);
                double r = 1.0;
                if (use_sign_certainty) {
                    int cur_q = quant_index[i];
                    int n_differ = 0, n_agree = 0;
                    // d=0,2,4: negative-index step → implied = sign(cur_q - nq)
                    // d=1,3,5: positive-index step → implied = sign(nq   - cur_q)
                    // mirrors boundary detection: left=sign(cur-nb), right=sign(nb-cur)
                    const int ddx[6] = {-1,1,0,0,0,0};
                    const int ddy[6] = {0,0,-1,1,0,0};
                    const int ddz[6] = {0,0,0,0,-1,1};
                    for (int d = 0; d < 6; d++) {
                        int nx = x + ddx[d], ny = y + ddy[d], nz = z + ddz[d];
                        if (nx < 0 || nx >= dims[0] || ny < 0 || ny >= dims[1] || nz < 0 || nz >= dims[2]) continue;
                        int nq = quant_index[(size_t)nx * r2_s0 + ny * r2_s1 + nz];
                        if (nq != cur_q) {
                            n_differ++;
                            char implied = (d % 2 == 0) ?
                                ((cur_q > nq) ? (char)1 : (char)-1) :
                                ((nq > cur_q) ? (char)1 : (char)-1);
                            if (implied == sign) n_agree++;
                        }
                    }
                    if (n_differ > 0) r *= (double)n_agree / n_differ;
                }
                if (use_geo_attenuation) {
                    r *= std::min(1.0, geo_scale / distance1);
                }
                if (use_plateau_attenuation) {
                    r *= std::min(1.0, plateau_cutoff / (distance1 + distance2));
                }
                compensation_map[i] = sign * r * magnitude * comepnsation_value;
                if (use_requant_clamp) {
                    double bin_center = 2.0 * eb * (double)quant_index[i];
                    double low = bin_center - eb;
                    double high = bin_center + eb;
                    double v = (double)dec_data[i] + (double)compensation_map[i];
                    if (v < low) {
                        compensation_map[i] = (T_data)(low - (double)dec_data[i]);
                        n_clamped++;
                    } else if (v > high) {
                        compensation_map[i] = (T_data)(high - (double)dec_data[i]);
                        n_clamped++;
                    }
                }
            }}}  // z, y, x
            stage_comp = stage_timer.stop();
            if (use_requant_clamp) {
                std::cout << "RequantClamp (IDW): " << n_clamped
                          << " / " << input_size
                          << " (" << (100.0 * n_clamped / input_size) << "%)" << std::endl;
            }
        }

        std::cout << "StageTime boundary_detect: " << stage_boundary << std::endl;
        std::cout << "StageTime edt_round1: " << stage_edt1 << std::endl;
        std::cout << "StageTime fill_sign: " << stage_fill_sign << std::endl;
        std::cout << "StageTime neutral_boundary: " << stage_neutral_boundary << std::endl;
        std::cout << "StageTime downsample_boundary: " << stage_downsample_boundary << std::endl;
        std::cout << "StageTime edt_round2: " << stage_edt2 << std::endl;
        std::cout << "StageTime compensation: " << stage_comp << std::endl;
        std::cout << "StageTime edt_total: " << (stage_edt1 + stage_edt2) << std::endl;

        return compensation_map;
    }

    std::vector<T_data> get_distance_array1() { return distance_array1; }

    std::vector<T_data> get_compensation_map_3d(std::vector<int> &sign_map) {
        auto boundary_map = get_boundary(quant_index, N, dims.data());

        // write boundary map to file
        // writefile("boundary.int8", boundary_map.data(), input_size);
        // flip the boundary map tag
        std::vector<bool> boundary_mask(input_size, false);
        for (int i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1) {
                boundary_map[i] = 0;      // boundary lable
                boundary_mask[i] = true;  // boundary lable
            } else {
                boundary_map[i] = 1;
            }
        }
        auto timer = Timer();

        timer.start();
        auto edt_omp = PM2::EDT_OMP<T_data, int>();
        edt_omp.set_num_threads(edt_thread_num);
        auto edt_result = edt_omp.NI_EuclideanFeatureTransform_(boundary_map.data(), N, dims.data(), edt_thread_num);
        std::cout << "edt time = " << timer.stop() << std::endl;
        auto distance_array = std::get<0>(edt_result);
        auto indexes = std::get<1>(edt_result);

        // writefile("distance.f64", distance_array.data(), distance_array.size());
        sign_map.resize(input_size, 0);
        auto grad_computer = ComputeGrad<T_quant>(N, dims.data(), quant_index);
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 0)  // boundary points
            {
                auto [compensate_direction, change_distance] = check_compensate_direction_distance_3d(i);
                auto max_iter = std::max_element(change_distance.begin(), change_distance.end());
                auto min_iter = std::min_element(change_distance.begin(), change_distance.end());
                int direction = std::distance(change_distance.begin(), min_iter);
                double sign = std::pow(-1.0, direction + 1) * compensate_direction[direction];
                double grad = grad_computer.get_grad(i);
                if (grad >= 1.0) {
                    sign = 0;
                }
                compensation_map[i] = sign * comepnsation_value;
                sign_map[i] = sign;
            }
        }

        // complete the sign map
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1)  // non-boundary points ·
            {
                char sign = get_sign(compensation_map[indexes[i]]);
                sign_map[i] = sign;
            }
        }

        // dump the sign map
        // get the second boundry map
        auto boundary_map2 = get_boundary(sign_map.data(), N, dims.data());

        // filp and remove the boundary points
        for (int i = 0; i < input_size; i++) {
            if (boundary_map2[i] == 1 && boundary_mask[i] == false) {
                boundary_map2[i] = 0;  // boundary lable
            } else {
                boundary_map2[i] = 1;
            }
        }
        // writefile("boundary2.int8", boundary_map2.data(), boundary_map2.size());
        // get the second edt map

        timer.start();
        edt_omp.reset_timer();
        edt_omp.set_num_threads(edt_thread_num);
        auto edt_result2 = edt_omp.NI_EuclideanFeatureTransform_(boundary_map2.data(), N, dims.data(), edt_thread_num);
        std::cout << "edt time = " << timer.stop() << std::endl;
        auto distance_array2 = std::get<0>(edt_result2);
        auto indexes2 = std::get<1>(edt_result2);
        // dump the distance array
        // writefile("distance1.f32", distance_array.data(), input_size);
        // writefile("distance2.f32", distance_array2.data(), input_size);
        for (size_t i = 0; i < input_size; i++) {
            // old method
            // if (boundary_map[i] == 1)  // non-boundary points ·
            if (1) {
                double distance1 = distance_array[i] + 0.5;
                double distance2 = distance_array2[i] + 0.5;
                char sign = sign_map[i];
                // double width = distance2 + distance1;
                // double relative_r = (distance1 ) / (width);
                // double magnitude = (1 - relative_r) * (1 - relative_r);
                // double magnitude = std::pow(1 - relative_r, 1.5);
                double magnitude = (1 / distance1) / (1 / distance1 + 1 / distance2);
                compensation_map[i] = sign * magnitude * comepnsation_value;
            }
        }
        return compensation_map;
    }

    std::vector<T_data> get_compensation_map() {
        if (N == 2) {
            return get_compensation_map_2d();
        } else if (N == 3) {
            return get_compensation_map_3d();
        }
        return compensation_map;
    }

    std::vector<T_data> get_compensation_map_2d(std::vector<double> &distance_array,
                                                std::vector<double> &distance_array2, std::vector<int> &sign_map) {
        auto boundary_map = get_boundary(quant_index, N, dims.data());
        // flip the boundary map tag
        std::vector<bool> boundary_mask(input_size, false);
        sign_map.resize(input_size, 0);
        size_t edge_point_count = 0;
        for (int i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1) {
                boundary_map[i] = 0;      // boundary lable
                boundary_mask[i] = true;  // boundary lable
                edge_point_count++;
            } else {
                boundary_map[i] = 1;
            }
        }
        auto timer = Timer();
        // std::cout << "edge point count = " << edge_point_count << std::endl;
        if (edge_point_count == 0) {
            distance_array.resize(input_size, std::numeric_limits<double>::max());
            distance_array2.resize(input_size, std::numeric_limits<double>::max());

            return compensation_map;
        }

        // timer.start();
        auto edt_omp = PM2::EDT_OMP<double, int>();
        edt_omp.set_num_threads(edt_thread_num);
        auto edt_result1 = edt_omp.NI_EuclideanFeatureTransform_(boundary_map.data(), N, dims.data());
        // std::cout << "edt time = " << timer.stop() << std::endl;
        distance_array = std::move(std::get<0>(edt_result1));
        auto indexes = std::move(std::get<1>(edt_result1));
        // std::vector<int> sign_map(input_size, 0);

        auto grad_computer = ComputeGrad<T_quant>(N, dims.data(), quant_index);
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 0)  // boundary points
            {
                auto [compensate_direction, change_distance] = check_compensate_direction_distance_2d(i);
                auto max_iter = std::max_element(change_distance.begin(), change_distance.end());
                auto min_iter = std::min_element(change_distance.begin(), change_distance.end());
                int direction = std::distance(change_distance.begin(), min_iter);
                double sign = std::pow(-1.0, direction + 1) * compensate_direction[direction];
                double grad = grad_computer.get_grad(i);
                if (grad >= 1.0) {
                    sign = 0;
                }
                compensation_map[i] = sign * comepnsation_value;
                sign_map[i] = sign;
            }
        }
        // complete the sign map
        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1)  // non-boundary points ·
            {
                char sign = get_sign(compensation_map[indexes[i]]);
                sign_map[i] = sign;
            }
        }
        // get the second boundry map
        auto boundary_map2 = get_boundary(sign_map.data(), N, dims.data());
        // filp and remove the boundary points
        for (int i = 0; i < input_size; i++) {
            if (boundary_map2[i] == 1 && boundary_mask[i] == false) {
                boundary_map2[i] = 0;  // boundary lable
            } else {
                boundary_map2[i] = 1;
            }
        }
        // get the second edt map
        timer.start();
        edt_omp.reset_timer();
        auto edt_result2 = edt_omp.NI_EuclideanFeatureTransform_(boundary_map2.data(), N, dims.data());
        // distance_array2 = std::move(std::get<0>(edt_result2));
        distance_array2 = std::get<0>(edt_result2);
        // auto indexes2 = std::move(std::get<1>(edt_result2));

        for (size_t i = 0; i < input_size; i++) {
            if (boundary_map[i] == 1)  // non-boundary points ·
            {
                double distance1 = distance_array[i] + 0.5;
                double distance2 = distance_array2[i] + 0.5;
                char sign = sign_map[i];
                double width = distance2 + distance1;
                // double relative_r = (distance1 ) / (width);
                // double magnitude = (1 - relative_r) * (1 - relative_r);
                // double magnitude = std::pow(1 - relative_r, 1.5);
                double magnitude = (1 / distance1) / (1 / distance1 + 1 / distance2);
                compensation_map[i] = sign * magnitude * comepnsation_value;
            }
        }
        return compensation_map;
    }

   private:
    int N;
    std::vector<int> dims;
    std::vector<size_t> strides;
    T_data *dec_data;
    T_quant *quant_index;
    size_t input_size;
    double comepnsation_value;
    std::vector<T_data> compensation_map;
    char *boundary_map;
    int edt_thread_num = 8;  // the thread number for edt computing
    double edt_time = 0.0;
    WeightMode weight_mode = WeightMode::IDW;
    double idw_power = 2.0;        // exponent for WeightMode::PowerIDW
    int downsample_r2_factor = 0;  // 0=off, 2/4/8=downsample factor for EDT round 2
    bool use_sign_certainty = false;
    bool use_geo_attenuation = false;
    double geo_scale = 3.0;
    bool use_plateau_attenuation = false;
    double plateau_cutoff = 20.0;  // voxels; full comp when d1+d2 <= cutoff
    bool geo_auto = false;        // if true, derive geo_scale from d1 percentile
    double geo_percentile = 80.0; // percentile of d1 used as geo_scale when geo_auto=true
    double geo_scale_min = 1.0;   // floor applied to auto-derived geo_scale (voxels)
    double edge_density_threshold = 0.001; // skip compensation if boundary fraction < this
    double sparsity_threshold = 0.10;     // skip compensation if non-mode fraction < this
    bool use_requant_clamp = false;       // Approach 4: clamp comp[i] to original quant bin
    double eb = 0.0;                      // absolute error bound; required when use_requant_clamp
    bool save_distance_maps = false;      // copy d1/d2 into member buffers for --profile_harm
    std::vector<T_data> d_edge_map;       // EDT round-1 distance (d1) when save_distance_maps
    std::vector<T_data> d_neutral_map;    // EDT round-2 distance (d2) when save_distance_maps
                                          // 0.10 empirically validated on NYX+Hurricane:
                                          // catches background-dominant fields (Q* mixing ratios)
                                          // while preserving compensation for smooth fields
    CPUIndexMode cpu_index_mode = CPUIndexMode::PackedXYZ32;
    std::vector<T_data> distance_array1;
    T_quant frequent_quant_index = 0;
};
}  // namespace PM

#endif  // COMPENSATION_HPP
