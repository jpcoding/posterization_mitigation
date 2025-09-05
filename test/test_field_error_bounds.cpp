#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <glob.h>
#include "SZ3/utils/FileUtil.hpp"

template <typename T>
inline int quantization_(T data, double abs_eb)
{   
    double recipPrecision = 1/(2*abs_eb);
    double dataRecip = data*recipPrecision;
    int s = dataRecip>=-0.5?0:1;
    return (int)(dataRecip+0.5) - s;
}

template <typename T>
double calculate_max_error(const std::vector<T>& original, const std::vector<T>& reconstructed) {
    double max_error = 0.0;
    for (size_t i = 0; i < original.size(); i++) {
        double error = std::abs(static_cast<double>(original[i]) - static_cast<double>(reconstructed[i]));
        max_error = std::max(max_error, error);
    }
    return max_error;
}

std::vector<std::string> glob_files(const std::string& pattern) {
    glob_t glob_result;
    std::vector<std::string> files;
    
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (return_value == 0) {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            files.push_back(std::string(glob_result.gl_pathv[i]));
        }
    }
    globfree(&glob_result);
    
    std::sort(files.begin(), files.end());
    return files;
}

void analyze_field(const std::string& filepath, const std::vector<double>& eb_list) {
    const size_t file_size = 500 * 500 * 500;
    
    // Read the data
    std::vector<float> data(file_size);
    SZ3::readfile(filepath.c_str(), file_size, data.data());
    
    // Calculate data statistics
    auto minmax = std::minmax_element(data.begin(), data.end());
    double data_min = static_cast<double>(*minmax.first);
    double data_max = static_cast<double>(*minmax.second);
    double data_range = data_max - data_min;
    
    // Get filename only
    std::filesystem::path p(filepath);
    std::string filename = p.filename().string();
    
    printf("%-12s  %8.2e  ", filename.c_str(), data_range);
    
    // Test each error bound
    for (double rel_eb : eb_list) {
        double abs_eb = rel_eb * data_range;
        
        // Create quantized data
        std::vector<float> qdata(file_size);
        bool has_error = false;
        
        for (size_t i = 0; i < file_size; i++) {
            int quant_idx = quantization_(data[i], abs_eb);
            qdata[i] = static_cast<float>(2.0 * abs_eb * quant_idx);
            
            // Check if there's any difference
            if (std::abs(data[i] - qdata[i]) > 1e-12) {
                has_error = true;
            }
        }
        
        // Calculate max error
        double max_error = calculate_max_error(data, qdata);
        
        // Determine status
        std::string status;
        if (max_error > 1e-10) {
            status = "✓";  // Clear quantization error
        } else if (has_error || max_error > 0) {
            status = "⚠";  // Marginal (very small error or perfect reconstruction)
        } else {
            status = "✗";  // Perfect reconstruction (likely too small)
        }
        
        printf("     %s     ", status.c_str());
    }
    printf("\n");
}

int main(int argc, char** argv) {
    // Error bound list to test
    std::vector<double> eb_list = {3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2};
    
    // Data directory
    std::string data_dir = "/scratch/pji228/useful/hpez_qoi/data/s3d_500_500_500";
    std::string pattern = data_dir + "/field_*.f32";
    
    // Find all field files
    auto field_files = glob_files(pattern);
    
    if (field_files.empty()) {
        printf("No field files found in %s\n", data_dir.c_str());
        return 1;
    }
    
    printf("C++ ERROR BOUND COMPATIBILITY TEST\n");
    printf("===================================\n");
    printf("Legend: ✓ = Works well, ⚠ = Marginal, ✗ = Too small\n\n");
    
    // Print header
    printf("Field         Range     ");
    for (double eb : eb_list) {
        printf(" %7.1e", eb);
    }
    printf("\n");
    
    printf("------------  --------  ");
    for (size_t i = 0; i < eb_list.size(); i++) {
        printf("---------");
    }
    printf("\n");
    
    // Analyze each field
    for (const auto& field_file : field_files) {
        analyze_field(field_file, eb_list);
    }
    
    printf("\n");
    
    // Test field_5 specifically with detailed output
    std::string field5_path = data_dir + "/field_5.f32";
    if (std::filesystem::exists(field5_path)) {
        printf("\nDETAILED ANALYSIS FOR FIELD_5:\n");
        printf("==============================\n");
        
        const size_t file_size = 500 * 500 * 500;
        std::vector<float> data(file_size);
        SZ3::readfile(field5_path.c_str(), file_size, data.data());
        
        auto minmax = std::minmax_element(data.begin(), data.end());
        double data_range = static_cast<double>(*minmax.second) - static_cast<double>(*minmax.first);
        
        printf("Data range: %.6e\n", data_range);
        printf("Error bound tests:\n");
        printf("  rel_eb     abs_eb        max_error     status\n");
        printf("  -------    ----------    ----------    ------\n");
        
        for (double rel_eb : eb_list) {
            double abs_eb = rel_eb * data_range;
            
            // Test quantization
            std::vector<float> qdata(file_size);
            for (size_t i = 0; i < file_size; i++) {
                int quant_idx = quantization_(data[i], abs_eb);
                qdata[i] = static_cast<float>(2.0 * abs_eb * quant_idx);
            }
            
            double max_error = calculate_max_error(data, qdata);
            
            std::string status;
            if (max_error > 1e-10) {
                status = "WORKS";
            } else if (max_error > 0) {
                status = "MARGINAL";
            } else {
                status = "PERFECT";
            }
            
            printf("  %.1e    %.6e    %.6e    %s\n", 
                   rel_eb, abs_eb, max_error, status.c_str());
        }
    }
    
    printf("\nAnalyzed %zu field files.\n", field_files.size());
    printf("Your observation: field_5 works with eb >= 5e-4\n");
    
    return 0;
}
