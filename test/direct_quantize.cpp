#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
// #include "SZ3/api/sz.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include<algorithm>



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



namespace SZ=SZ3; 
int main(int argc, char** argv)
{

    if (argc < 2) {
        printf("Usage: %s <input_file> <relative_error_bound> <outputfile_name> \n", argv[0]);
        return 0;
    }

    std::filesystem::path p{argv[1]} ;
    if (!std::filesystem::exists(p)) {
        printf("File %s does not exist\n", argv[1]);
        return 0;
    }

    size_t file_size = std::filesystem::file_size(p)/sizeof(float);


    std::vector<int> quant_inds(file_size, 0);
    std::vector<float> input_data(file_size, 0);

    SZ::readfile(argv[1],  file_size, input_data.data());
    double max = double(*std::max_element(input_data.begin(), input_data.end()));
    double min = double(*std::min_element(input_data.begin(), input_data.end()));
    printf("max: %f, min: %f\n", max, min);
    double eb = atof(argv[2])*(max - min);
    printf("relative eb: %.6f\n", atof(argv[2]));
    printf("absolute eb: %.6f\n", eb);
    std::string outname = argv[3];

    // create a linear quantizer
    auto quantizer = SZ::LinearQuantizer<float>();
    quantizer.set_eb(eb);

    // iterate the input data and quantize it
    for (size_t i = 0; i < file_size; i++) {
        // quant_inds[i] = quantizer.quantize_and_overwrite(input_data[i],0);
        float orig_copy = input_data[i]; 
        quant_inds[i] = quantization_(input_data[i], eb);

        double temp = 2*eb * quant_inds[i]; 
        input_data[i] = float(temp);
        if (std::abs(input_data[i] - orig_copy) > eb) {
            input_data[i] = orig_copy;
            quant_inds[i] = 0;
        }
    }

    // write the quantized data to a file
    std::string output_file =  outname+ ".quant.i32";
    std::string out_data_file = outname + ".out"; 
    printf("Writing quantized data to %s\n", output_file.c_str());
    SZ::writefile(output_file.c_str(), quant_inds.data(),file_size);
    SZ::writefile(out_data_file.c_str(), input_data.data(),file_size);    

    return 0;
}


