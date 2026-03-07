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
    bool use_rbf;
    bool no_ssim = false;
    bool downsample_r2 = false;
    double compensation_factor = 0.9;
    app.add_option("-N", N, "number of dimensions")->required();
    dims.resize(N, 0);
    app.add_option("-d", dims, "dimensions")->required();
    app.add_option("-i", input_file, "input file")->required();
    app.add_option("-m", eb_mode, "eb mode")->required();
    app.add_option("-e", eb, "eb")->required();
    app.add_option("-q", quantized_file, "quantized file")->required();
    app.add_option("-c", compensation_file, "compensation file")->required();
    app.add_option("-t", num_threads, "number of threads")->default_val(1)->check(CLI::Range(1, 256));
    app.add_option("--use_rbf", use_rbf, "use rbf")->default_val(false);
    app.add_option("--no_ssim", no_ssim, "do not calculate ssim")->default_val(false);
    app.add_option("--downsample_r2", downsample_r2, "downsample EDT round 2 by 2x in each dim (faster, ~-0.07 dB PSNR)")->default_val(false);
    app.add_option("--eta", compensation_factor, "compensation_factor")-> default_val(0.9);
    CLI11_PARSE(app, argc, argv);

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

    double threshold = 0.5;
    size_t count = 0;
    for (size_t i = 0; i < data_size; i++) {
        if (quant_inds[i] == frequent_quant_index) {
            count++;
        }
    }
    printf("zero count = %zu\n", count);
    printf("zero ratio = %f\n", (double)count / data_size);
    if (count > threshold * data_size) {
        printf("too many zeros, will not compensate\n");
        // return 0;
    }
    
    auto timer = Timer();
    timer.start();
    if (operation) {
        auto compensator = PM::Compensation<Real, int>(N, dims.data(), dec_data.data(), quant_inds.data(),
                                                       max_diff * compensation_factor);
        compensator.set_frequent_quant_index(frequent_quant_index);
        compensator.set_edt_thread_num(num_threads);
        compensator.set_use_rbf(use_rbf);
        compensator.set_downsample_edt_round2(downsample_r2);

        auto compensation_map = compensator.get_compensation_map();

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
