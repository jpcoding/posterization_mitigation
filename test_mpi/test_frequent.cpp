#include <mpi.h>
#include <stdio.h>
#include <cstddef>
#include <cstdlib>
#include <string>
#include "CLI/CLI.hpp"
#include "utils/file_utils.hpp"
#include "utils/stats.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
namespace SZ = SZ3;

template <typename T>
inline int quantization_(T data, double abs_eb)
{   
    double recipPrecision = 1/(2*abs_eb);
    double dataRecip = data*recipPrecision;
    int s = dataRecip>=-0.5?0:1;
    return (int)(dataRecip+0.5) - s;
}


int main(int argc, char** argv) {

    std::unique_ptr<float[]> data; 
    size_t num_elements = 0; 
    data = readfile<float>(argv[1], num_elements);

    double rel_eb  = atof(argv[2]);

    float data_max = *std::max_element(data.get(), data.get() + num_elements);
    float data_min = *std::min_element(data.get(), data.get() + num_elements);
    double data_range = data_max - data_min;
    double abs_eb = rel_eb * data_range;

    auto quantizer = SZ::LinearQuantizer<float>();
    quantizer.set_eb(abs_eb);

    std::vector<int> quant_inds(num_elements, 0);
    for (size_t i = 0; i < num_elements; i++) {
        // quant_inds[i] = quantization_(data[i], abs_eb);
        quant_inds[i] = quantizer.quantize_and_overwrite(data[i], 0) - 32768;
    }
    int most_frequent = 0;
    most_frequent = get_most_frequent_quantization_index<int>(quant_inds) ;
    std::cout << "most frequent quantization index = " << most_frequent << std::endl; 


    std::cout << "num_elements = " << num_elements << std::endl; 
    return 0;
}

