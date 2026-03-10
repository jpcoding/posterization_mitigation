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
    app.add_option("--geo_percentile", geo_percentile, "percentile of d1 used as geo_scale when --geo_auto=1 (default 10)")->default_val(10.0);
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
        if (geo_auto) {
            compensator.set_geo_auto(true);
            compensator.set_geo_percentile(geo_percentile);
        } else {
            compensator.set_geo_scale(geo_scale);
        }

        auto compensation_map = compensator.get_compensation_map();

        // --- Harm-rate: per-voxel quality comparison before/after compensation ---
        {
            size_t n_harmed = 0, n_helped = 0, n_neutral = 0;
            double sum_harm_delta_sq = 0.0, sum_help_delta_sq = 0.0;
            const double eps = 1e-30;
            for (size_t i = 0; i < data_size; i++) {
                double orig  = original_data[i];
                double before_err = std::abs(dec_data[i] - orig);
                double after_err  = std::abs(dec_data[i] + compensation_map[i] - orig);
                double delta_sq   = after_err * after_err - before_err * before_err;
                if (after_err > before_err + eps) {
                    n_harmed++;
                    sum_harm_delta_sq += delta_sq;
                } else if (after_err < before_err - eps) {
                    n_helped++;
                    sum_help_delta_sq -= delta_sq; // positive
                } else {
                    n_neutral++;
                }
            }
            double harm_rate   = (double)n_harmed / data_size;
            double benefit_rate = (double)n_helped / data_size;
            // convert mean squared error change to dB: 10*log10(after_mse / before_mse) — reported as avg per harmed voxel
            double harm_rms_delta  = (n_harmed > 0)  ? std::sqrt(sum_harm_delta_sq  / n_harmed)  : 0.0;
            double help_rms_delta  = (n_helped > 0)  ? std::sqrt(sum_help_delta_sq  / n_helped)  : 0.0;
            printf("harm_rate = %.6f  (%zu / %zu voxels)\n", harm_rate, n_harmed, data_size);
            printf("benefit_rate = %.6f  (%zu / %zu voxels)\n", benefit_rate, n_helped, data_size);
            printf("harm_rms_err_increase = %.6E\n", harm_rms_delta);
            printf("benefit_rms_err_decrease = %.6E\n", help_rms_delta);
        }

        // writefile("compensation_map.f32", compensation_map.data(), data_size);
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
