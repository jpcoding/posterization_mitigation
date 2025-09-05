#include <iostream>
#include <vector>
#include <cassert>
#include "utils/sampling.hpp"

using namespace PM;

void test_2d_sampling() {
    std::cout << "Testing 2D sampling..." << std::endl;
    
    // Create a simple 5x5 2D array
    int dims[] = {5, 5};
    std::vector<float> data = {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,  // center is 13 at (2,2)
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    
    // Test center point sampling
    float center_point = sample_center_point(data.data(), dims, 2);
    std::cout << "Center point (should be 13): " << center_point << std::endl;
    assert(center_point == 13.0f);
    
    // Test center region sampling (3x3 region)
    auto region = sample_center_region(data.data(), dims, 2, 3);
    std::cout << "Center 3x3 region: ";
    for (const auto& val : region) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Expected: 7, 8, 9, 12, 13, 14, 17, 18, 19
    assert(region.size() == 9);
    
    // Test get center coordinates
    auto coords = get_center_coordinates(dims, 2);
    std::cout << "Center coordinates: (" << coords[0] << ", " << coords[1] << ")" << std::endl;
    assert(coords[0] == 2 && coords[1] == 2);
}

void test_3d_sampling() {
    std::cout << "\nTesting 3D sampling..." << std::endl;
    
    // Create a simple 3x3x3 3D array
    int dims[] = {3, 3, 3};
    std::vector<int> data(27);
    
    // Fill with sequential values
    for (int i = 0; i < 27; i++) {
        data[i] = i + 1;
    }
    
    // Test center point sampling (should be element at (1,1,1) = index 13+1 = 14)
    int center_point = sample_center_point(data.data(), dims, 3);
    std::cout << "3D Center point (should be 14): " << center_point << std::endl;
    assert(center_point == 14);
    
    // Test center coordinates
    auto coords = get_center_coordinates(dims, 3);
    std::cout << "3D Center coordinates: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")" << std::endl;
    assert(coords[0] == 1 && coords[1] == 1 && coords[2] == 1);
}

void test_custom_region_sampling() {
    std::cout << "\nTesting custom region sampling..." << std::endl;
    
    // Create a 6x4 2D array
    int dims[] = {6, 4};
    std::vector<float> data = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,  // center around (2.5, 1.5) -> (2, 1)
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    };
    
    // Test custom region sizes: 2x4 region
    int region_sizes[] = {2, 4};
    auto region = sample_center_region_custom(data.data(), dims, 2, region_sizes);
    
    std::cout << "Custom region (2x4): ";
    for (const auto& val : region) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Should sample 2 rows (around center row 2) and all 4 columns
    assert(region.size() == 8);
}

int main() {
    std::cout << "=== Testing Center Sampling Utility ===" << std::endl;
    
    test_2d_sampling();
    test_3d_sampling(); 
    test_custom_region_sampling();
    
    std::cout << "\nAll tests passed! ✓" << std::endl;
    std::cout << "\nUsage examples:" << std::endl;
    std::cout << "1. PM::sample_center_point(data, dims, ndims) - Get single center value" << std::endl;
    std::cout << "2. PM::sample_center_region(data, dims, ndims, size) - Get center region" << std::endl;
    std::cout << "3. PM::sample_center_region_custom(data, dims, ndims, sizes) - Get custom-sized region" << std::endl;
    std::cout << "4. PM::get_center_coordinates(dims, ndims) - Get center coordinates only" << std::endl;
    
    return 0;
}
