#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "CLI/CLI.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include "compensation.hpp"
#include "utils/qcat_ssim.hpp"
#include "utils/stats.hpp"
#include "utils/timer.hpp"

using Real = float;
namespace SZ = SZ3;


template <typename T>
inline int quantization_(T data, double abs_eb)
{   
    double recipPrecision = 1/(2*abs_eb);
    double dataRecip = data*recipPrecision;
    int s = dataRecip>=-0.5?0:1;
    return (int)(dataRecip+0.5) - s;

    // double recip = 1.0 / (2.0 * abs_eb);
    // return std::lround(static_cast<double>(data) * recip);
}



int main(int argc, char **argv) {
    CLI::App app{"OMP version of compensation using EDT method"};
    argv = app.ensure_utf8(argv);
    int N = 0;
    int num_threads = 1;
    std::vector<int> dims;
    std::string input_file;
    std::string eb_mode;
    double eb = 0.0;
    std::string quantized_file;
    std::string compensation_file;
    std::string cpu_index_mode = "packed";
    std::string weight_mode_str = "idw";
    double idw_power = 2.0;
    bool no_ssim = false;
    int downsample_r2 = 0;
    double compensation_factor = 0.9;
    app.add_option("-N", N, "number of dimensions")->required();
    dims.resize(N, 0);
    app.add_option("-d", dims, "dimensions")->required();
    app.add_option("-i", input_file, "input file")->required();
    app.add_option("-m", eb_mode, "eb mode")->required();
    app.add_option("-e", eb, "eb")->required();
    app.add_option("-q", quantized_file, "quantized file")->required();
    app.add_option("-c", compensation_file, "compensation file")->required();
    app.add_option("--cpu_index_mode", cpu_index_mode, "CPU EDT index mode: packed or flat32")->default_val("packed");
    app.add_option("-t", num_threads, "number of threads")->default_val(1)->check(CLI::Range(1, 256));
    app.add_option("--weight_mode", weight_mode_str, "weight function: idw (default), power_idw, smoothstep, rbf")->default_val("idw");
    app.add_option("--idw_power", idw_power, "exponent p for --weight_mode=power_idw (default 2.0)")->default_val(2.0);
    app.add_option("--no_ssim", no_ssim, "do not calculate ssim")->default_val(false);
    app.add_option("--downsample_r2", downsample_r2, "downsample factor for EDT round 2 (0=off, 2/4/8=Fx downsample)")->default_val(0);
    app.add_option("--eta", compensation_factor, "compensation_factor")-> default_val(0.9);
    bool sign_certainty = false;
    bool geo_attenuation = false;
    double geo_scale = 3.0;
    app.add_option("--sign_certainty", sign_certainty, "r_i c_sign: scale by fraction of differing neighbors agreeing on sign")->default_val(false);
    app.add_option("--geo_attenuation", geo_attenuation, "r_i c_geom: attenuate compensation for large plateaus (min(1, geo_scale/d1))")->default_val(false);
    app.add_option("--geo_scale", geo_scale, "geo_attenuation threshold in voxels (overrides --geo_auto)")->default_val(3.0);
    bool geo_auto = false;
    double geo_percentile = 10.0;
    app.add_option("--geo_auto", geo_auto, "derive geo_scale from d1 percentile, no original data needed")->default_val(false);
    app.add_option("--geo_percentile", geo_percentile, "percentile of d1 used as geo_scale when --geo_auto=1 (default 80)")->default_val(80.0);
    double geo_scale_min = 1.0;
    app.add_option("--geo_scale_min", geo_scale_min,
                   "floor for auto-derived geo_scale in voxels (default 1.0; prevents small-eb collapse on dense-boundary fields)")
        ->default_val(1.0);
    bool plateau_attenuation = false;
    double plateau_cutoff = 20.0;
    app.add_option("--plateau_attenuation", plateau_attenuation,
                   "r_i c_plateau: attenuate when total plateau width d1+d2 > cutoff (min(1, cutoff/(d1+d2)))")->default_val(false);
    app.add_option("--plateau_cutoff", plateau_cutoff,
                   "plateau width cutoff in voxels for --plateau_attenuation (default 20)")->default_val(20.0);
    double edge_density_threshold = 0.001;
    app.add_option("--edge_density_threshold", edge_density_threshold,
                   "skip compensation if boundary fraction < this (default 0.001)")->default_val(0.001);
    double sparsity_threshold = 0.10;
    app.add_option("--sparsity_threshold", sparsity_threshold,
                   "skip compensation if non-mode fraction < this (default 0.10)")->default_val(0.10);
    bool requant_clamp = false;
    app.add_option("--requant_clamp", requant_clamp,
                   "Approach 4: clamp dec_data[i]+comp[i] to the original quant bin [2k-1,2k+1]*eb (hard guarantee)")
        ->default_val(false);
    std::string profile_harm_prefix;
    app.add_option("--profile_harm", profile_harm_prefix,
                   "if set, save sidecar files <prefix>_d_edge.f32, <prefix>_d_neutral.f32, <prefix>_quant.i32, <prefix>_comp.f32 for offline harm analysis")
        ->default_val("");
    bool smoothness_clamp = false;
    double smoothness_alpha = 1.0;
    app.add_option("--smoothness_clamp", smoothness_clamp,
                   "Approach 2: post-comp pass clamping v[i] toward neighbor-implied smoothness band")
        ->default_val(false);
    app.add_option("--smoothness_alpha", smoothness_alpha,
                   "band-width factor for --smoothness_clamp (band = alpha*2*eb); alpha<1 tightens smoothness")
        ->default_val(1.0);
    bool near_edge_attenuation = false;
    double near_edge_d_min = 2.0;
    app.add_option("--near_edge_attenuation", near_edge_attenuation,
                   "Approach 3: attenuate near-boundary voxels (r *= min(1, d1/d_min))")
        ->default_val(false);
    app.add_option("--near_edge_d_min", near_edge_d_min,
                   "d_min in voxels for --near_edge_attenuation (default 2.0)")
        ->default_val(2.0);
    bool sign_ratio_attenuation = false;
    double sign_ratio_threshold = 1.0;
    app.add_option("--sign_ratio_attenuation", sign_ratio_attenuation,
                   "attenuate when d2/d1 < threshold: r *= (d2/d1)/threshold (sign ambiguity proxy)")
        ->default_val(false);
    app.add_option("--sign_ratio_threshold", sign_ratio_threshold,
                   "threshold for --sign_ratio_attenuation (default 1.0)")
        ->default_val(1.0);
    CLI11_PARSE(app, argc, argv);
    std::printf("cpu index mode = %s\n", cpu_index_mode.c_str());

    size_t data_size = 1;
    for (int i = 0; i < N; i++) {
        data_size *= dims[i];
    }
    std::vector<Real> original_data(data_size, 0);
    readfile(input_file.c_str(), data_size, original_data.data());
    double max = double(*std::max_element(original_data.begin(), original_data.end()));
    double min = double(*std::min_element(original_data.begin(), original_data.end()));
    double rel_eb = eb;
    if (eb_mode == "abs") {
        eb = eb;
        rel_eb = eb / (max - min);

    } else if (eb_mode == "rel") {
        rel_eb = eb;
        eb = eb * (max - min);
    }
    double range = max - min;
    bool operation = true;
    // if (range < 1e-10) {
    //     operation = false;
    // }

    // make a copy of the original data
    std::vector<Real> dec_data(data_size, 0);
    std::copy(original_data.begin(), original_data.end(), dec_data.begin());
    std::vector<int> quant_inds(data_size, 0);
    printf("max: %f, min: %f\n", max, min);
    printf("absolute eb: %.6E\n", eb);
    // create a linear quantizer
    auto quantizer = SZ::LinearQuantizer<Real>(eb, 32768);
    quantizer.set_eb(eb);
    // iterate the input data and quantize it
    if (1) {
        for (size_t i = 0; i < data_size; i++) {
            quant_inds[i] = quantization_(original_data[i], eb);
            double temp = 2*eb * quant_inds[i]; 
            dec_data[i] = temp;
            // check if eb is satisfied, if not use the original
            if (std::abs(dec_data[i] - original_data[i]) > eb) {
                dec_data[i] = original_data[i];
                quant_inds[i] = 0;
            }
            // quant_inds[i] = quantizer.quantize_and_overwrite(dec_data[i], 0.0) - quantizer.get_radius();
            // if (quant_inds[i] == -quantizer.get_radius() )
            // {
            //     quant_inds[i] = 0;
            // }
        }
    }
    int frequent_quant_index = get_most_frequent_quantization_index(quant_inds);
    printf("frequent quantization index = %d\n", frequent_quant_index);

    double psnr, nrmse, max_diff;
    verify(original_data.data(), dec_data.data(), data_size, psnr, nrmse, max_diff);



    int max_quant = *std::max_element(quant_inds.begin(), quant_inds.end());
    int min_quant = *std::min_element(quant_inds.begin(), quant_inds.end());
    if (max_quant == min_quant) {
        printf("max quant = min quant = %d \n", max_quant);
        operation = false;
    }

    // check quantization index

    writefile(quantized_file.c_str(), dec_data.data(), data_size);

    // writefile("quant_index.i32", quant_inds.data(), data_size);

    // write quantized data
    // writefile((fs::path(argv[N + 2]).filename().string() + ".qcd").c_str(), dec_data.data(), data_size);

    // verify the data

    // cast dims to size_t

    if (no_ssim == false) {
        std::vector<size_t> dims_(3);
        for (int i = 0; i < N; i++) {
            dims_[i] = dims[i];
        }
        auto ssim = PM::calculateSSIM(original_data.data(), dec_data.data(), N, dims_.data());
        printf("SSIM = %f\n", ssim);
    }

    // compensation using edt method

    // Fast pre-check: if >99.9% of voxels share one quant_index, the boundary density is
    // guaranteed < 0.001 (≤ 6*(1-f)) and EDT sign propagation is unreliable.
    // The edge-density guard inside get_compensation_map() is the principled check;
    // this is just a cheap O(n) early exit for the most extreme cases.
    size_t count = 0;
    for (size_t i = 0; i < data_size; i++) {
        if (quant_inds[i] == frequent_quant_index) count++;
    }
    double mode_ratio = (double)count / data_size;
    double sparsity   = 1.0 - mode_ratio;
    printf("mode count = %zu\n", count);
    printf("sparsity = %f\n", sparsity);
    if (sparsity < 1.0 / 6000.0) {  // fast pre-check: edge density guaranteed < 0.001
        printf("sparsity=%.6f too low (< 1/6000), skipping compensation\n", sparsity);
        operation = false;
    }
    
    auto timer = Timer();
    timer.start();
    if (operation) {
        auto compensator = PM::Compensation<Real, int>(N, dims.data(), dec_data.data(), quant_inds.data(),
                                                       max_diff * compensation_factor);
        if (cpu_index_mode == "packed") {
            compensator.set_cpu_index_mode(PM::CPUIndexMode::PackedXYZ32);
        } else if (cpu_index_mode == "flat32") {
            compensator.set_cpu_index_mode(PM::CPUIndexMode::Flat32);
        } else {
            std::fprintf(stderr, "invalid --cpu_index_mode: %s\n", cpu_index_mode.c_str());
            return 1;
        }
        compensator.set_frequent_quant_index(frequent_quant_index);
        compensator.set_edt_thread_num(num_threads);
        {
            PM::WeightMode wm = PM::WeightMode::IDW;
            if (weight_mode_str == "power_idw")  wm = PM::WeightMode::PowerIDW;
            else if (weight_mode_str == "smoothstep") wm = PM::WeightMode::Smoothstep;
            else if (weight_mode_str == "rbf")    wm = PM::WeightMode::RBF;
            compensator.set_weight_mode(wm);
            compensator.set_idw_power(idw_power);
        }
        compensator.set_downsample_r2_factor(downsample_r2);
        compensator.set_sign_certainty(sign_certainty);
        compensator.set_geo_attenuation(geo_attenuation);
        compensator.set_plateau_attenuation(plateau_attenuation);
        compensator.set_plateau_cutoff(plateau_cutoff);
        compensator.set_edge_density_threshold(edge_density_threshold);
        compensator.set_sparsity_threshold(sparsity_threshold);
        compensator.set_requant_clamp(requant_clamp);
        compensator.set_eb(eb);
        compensator.set_smoothness_clamp(smoothness_clamp);
        compensator.set_smoothness_alpha(smoothness_alpha);
        compensator.set_near_edge_attenuation(near_edge_attenuation);
        compensator.set_near_edge_d_min(near_edge_d_min);
        compensator.set_sign_ratio_attenuation(sign_ratio_attenuation);
        compensator.set_sign_ratio_threshold(sign_ratio_threshold);
        if (!profile_harm_prefix.empty()) {
            compensator.set_save_distance_maps(true);
        }
        if (geo_auto) {
            compensator.set_geo_auto(true);
            compensator.set_geo_percentile(geo_percentile);
            compensator.set_geo_scale_min(geo_scale_min);
        } else {
            compensator.set_geo_scale(geo_scale);
        }

        auto compensation_map = compensator.get_compensation_map();

        // --- Harm-rate: per-voxel quality comparison before/after compensation ---
        {
            size_t n_harmed = 0, n_helped = 0, n_neutral = 0;
            size_t n_harm_wrong_sign = 0;   // comp pushed away from truth (sign error)
            size_t n_harm_overshoot = 0;    // right sign but |comp| > 2*|err_before|
            size_t n_harm_zero_err = 0;     // dec was already exact; any comp is harmful
            double sum_harm_delta_sq = 0.0, sum_help_delta_sq = 0.0;
            const double eps = 1e-30;
            for (size_t i = 0; i < data_size; i++) {
                double orig      = original_data[i];
                double err_before = dec_data[i] - orig;           // signed
                double comp       = compensation_map[i];
                double err_after  = err_before + comp;
                double before_abs = std::abs(err_before);
                double after_abs  = std::abs(err_after);
                double delta_sq   = err_after * err_after - err_before * err_before;
                if (after_abs > before_abs + eps) {
                    n_harmed++;
                    sum_harm_delta_sq += delta_sq;
                    if (before_abs < eps) {
                        n_harm_zero_err++;
                    } else if (comp * err_before > 0) {
                        // comp has same sign as err_before → pushes away from truth
                        n_harm_wrong_sign++;
                    } else {
                        // right direction but went too far
                        n_harm_overshoot++;
                    }
                } else if (after_abs < before_abs - eps) {
                    n_helped++;
                    sum_help_delta_sq -= delta_sq;
                } else {
                    n_neutral++;
                }
            }
            double harm_rate    = (double)n_harmed / data_size;
            double benefit_rate = (double)n_helped / data_size;
            double harm_rms_delta = (n_harmed > 0) ? std::sqrt(sum_harm_delta_sq / n_harmed) : 0.0;
            double help_rms_delta = (n_helped > 0) ? std::sqrt(sum_help_delta_sq / n_helped) : 0.0;
            printf("harm_rate = %.6f  (%zu / %zu voxels)\n", harm_rate, n_harmed, data_size);
            printf("benefit_rate = %.6f  (%zu / %zu voxels)\n", benefit_rate, n_helped, data_size);
            printf("harm_rms_err_increase = %.6E\n", harm_rms_delta);
            printf("benefit_rms_err_decrease = %.6E\n", help_rms_delta);
            if (n_harmed > 0) {
                printf("harm_breakdown: wrong_sign=%zu (%.1f%%)  overshoot=%zu (%.1f%%)  zero_err=%zu (%.1f%%)\n",
                    n_harm_wrong_sign, 100.0 * n_harm_wrong_sign / n_harmed,
                    n_harm_overshoot,  100.0 * n_harm_overshoot  / n_harmed,
                    n_harm_zero_err,   100.0 * n_harm_zero_err   / n_harmed);
            }
        }

        // writefile("compensation_map.f32", compensation_map.data(), data_size);
        if (!profile_harm_prefix.empty()) {
            const auto& d_edge    = compensator.get_d_edge();
            const auto& d_neutral = compensator.get_d_neutral();
            std::string p = profile_harm_prefix;
            writefile((p + "_d_edge.f32").c_str(),    d_edge.data(),         data_size);
            writefile((p + "_d_neutral.f32").c_str(), d_neutral.data(),      data_size);
            writefile((p + "_quant.i32").c_str(),     quant_inds.data(),     data_size);
            writefile((p + "_comp.f32").c_str(),      compensation_map.data(), data_size);
            writefile((p + "_dec.f32").c_str(),       dec_data.data(),       data_size);
            printf("profile sidecars written with prefix: %s\n", p.c_str());

            // --- Sign-ratio stratification ---
            // Hypothesis: wrong-sign harm events cluster at low d2/d1 (sign ambiguity).
            // d2/d1 approximates how confident the sign assignment is:
            //   small ratio → voxel is close to a sign flip relative to nearest boundary
            //   large ratio → voxel is deep in its plateau, sign is unambiguous
            if (!d_edge.empty() && !d_neutral.empty()) {
                const double bins[] = {0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9};
                const int nbins = 6;
                size_t total_harm_ws[nbins]   = {};
                size_t total_harm_os[nbins]   = {};
                size_t total_help[nbins]       = {};
                const double eps = 1e-30;
                for (size_t i = 0; i < data_size; i++) {
                    double d1 = d_edge[i];
                    double d2 = d_neutral[i];
                    double ratio = (d1 > eps) ? (d2 / d1) : 1e9;
                    int b = nbins - 1;
                    for (int k = 0; k < nbins; k++) { if (ratio < bins[k]) { b = k; break; } }
                    double orig       = original_data[i];
                    double err_before = dec_data[i] - orig;  // NOTE: dec_data not yet updated here
                    double comp       = compensation_map[i];
                    double err_after  = err_before + comp;
                    if (std::abs(err_after) > std::abs(err_before) + eps) {
                        if (std::abs(err_before) < eps || comp * err_before > 0) total_harm_ws[b]++;
                        else total_harm_os[b]++;
                    } else if (std::abs(err_after) < std::abs(err_before) - eps) {
                        total_help[b]++;
                    }
                }
                printf("d2/d1 ratio stratification (wrong_sign | overshoot | benefit per bin):\n");
                double lo = 0;
                for (int b = 0; b < nbins; b++) {
                    size_t tot = total_harm_ws[b] + total_harm_os[b] + total_help[b];
                    double ws_frac = tot > 0 ? 100.0 * total_harm_ws[b] / tot : 0;
                    printf("  [%.1f, %.1f): harm_ws=%zu  harm_os=%zu  help=%zu  (wrong_sign=%.1f%% of affected)\n",
                        lo, bins[b], total_harm_ws[b], total_harm_os[b], total_help[b], ws_frac);
                    lo = bins[b];
                }
            }
        }
        // add the compensation map to the dec_data
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < data_size; i++) {
            dec_data[i] += compensation_map[i];
        }
    }
    std::cout << "compensation time = " << timer.stop() << std::endl;
    // verify the compensated data
    verify(original_data.data(), dec_data.data(), data_size, psnr, nrmse, max_diff);
    if (no_ssim == false) {
        std::vector<size_t> dims_(3);
        for (int i = 0; i < N; i++) {
            dims_[i] = dims[i];
        }
        auto ssim = PM::calculateSSIM(original_data.data(), dec_data.data(), N, dims_.data());
        printf("SSIM = %f\n", ssim);
    }
    // write the compensation map to file
    writefile(compensation_file.c_str(), dec_data.data(), data_size);
    return 0;
}
