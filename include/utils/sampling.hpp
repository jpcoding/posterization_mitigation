#ifndef PM_SAMPLING_HPP_FILE
#define PM_SAMPLING_HPP_FILE

#include <vector>
#include <cassert>
#include <cstddef>
#include <algorithm>

namespace PM {

namespace detail {
/**
 * Helper function to recursively sample a region
 */
template<typename T>
void sample_region_recursive(const T* data, const int* dims, int ndims,
                            const std::vector<int>& start_coords,
                            const std::vector<int>& end_coords,
                            std::vector<int>& current_coords,
                            int current_dim,
                            std::vector<T>& result) {
    if (current_dim == ndims) {
        // Calculate linear index and add to result
        size_t index = 0;
        size_t stride = 1;
        for (int i = ndims - 1; i >= 0; i--) {
            index += current_coords[i] * stride;
            stride *= dims[i];
        }
        result.push_back(data[index]);
        return;
    }
    
    for (int i = start_coords[current_dim]; i <= end_coords[current_dim]; i++) {
        current_coords[current_dim] = i;
        sample_region_recursive(data, dims, ndims, start_coords, end_coords, 
                              current_coords, current_dim + 1, result);
    }
}
} // namespace detail

/**
 * Sample a single value from the center of multi-dimensional data
 * @param data Pointer to the data array (row-major order)
 * @param dims Dimensions of the data [dim0, dim1, ..., dimN-1]
 * @param ndims Number of dimensions
 * @return The center value
 */
template<typename T>
T sample_center_point(const T* data, const int* dims, int ndims) {
    assert(data != nullptr);
    assert(dims != nullptr);
    assert(ndims > 0);
    
    size_t center_index = 0;
    size_t stride = 1;
    
    // Calculate strides and center index
    for (int i = ndims - 1; i >= 0; i--) {
        assert(dims[i] > 0);
        center_index += (dims[i] / 2) * stride;
        stride *= dims[i];
    }
    
    return data[center_index];
}

/**
 * Sample a center region from multi-dimensional data
 * @param data Pointer to the data array (row-major order)
 * @param dims Dimensions of the data [dim0, dim1, ..., dimN-1]
 * @param ndims Number of dimensions
 * @param region_size Size of the region to sample in each dimension
 * @return Vector containing the sampled region values
 */
template<typename T>
std::vector<T> sample_center_region(const T* data, const int* dims, int ndims, int region_size) {
    assert(data != nullptr);
    assert(dims != nullptr);
    assert(ndims > 0);
    assert(region_size > 0);
    
    std::vector<T> result;
    
    // Calculate center coordinates
    std::vector<int> center_coords(ndims);
    for (int i = 0; i < ndims; i++) {
        center_coords[i] = dims[i] / 2;
    }
    
    // Calculate region bounds
    std::vector<int> start_coords(ndims);
    std::vector<int> end_coords(ndims);
    for (int i = 0; i < ndims; i++) {
        int half_region = region_size / 2;
        start_coords[i] = std::max(0, center_coords[i] - half_region);
        end_coords[i] = std::min(dims[i] - 1, center_coords[i] + half_region);
    }
    
    // Recursively sample the region
    std::vector<int> current_coords(ndims, 0);
    detail::sample_region_recursive(data, dims, ndims, start_coords, end_coords, current_coords, 0, result);
    
    return result;
}

/**
 * Sample a center region with custom size for each dimension
 * @param data Pointer to the data array (row-major order)  
 * @param dims Dimensions of the data [dim0, dim1, ..., dimN-1]
 * @param ndims Number of dimensions
 * @param region_sizes Size of the region to sample in each dimension
 * @return Vector containing the sampled region values
 */
template<typename T>
std::vector<T> sample_center_region_custom(const T* data, const int* dims, int ndims, const int* region_sizes) {
    assert(data != nullptr);
    assert(dims != nullptr);
    assert(region_sizes != nullptr);
    assert(ndims > 0);
    
    std::vector<T> result;
    
    // Calculate center coordinates
    std::vector<int> center_coords(ndims);
    for (int i = 0; i < ndims; i++) {
        center_coords[i] = dims[i] / 2;
    }
    
    // Calculate region bounds
    std::vector<int> start_coords(ndims);
    std::vector<int> end_coords(ndims);
    for (int i = 0; i < ndims; i++) {
        int half_region = region_sizes[i] / 2;
        start_coords[i] = std::max(0, center_coords[i] - half_region);
        end_coords[i] = std::min(dims[i] - 1, center_coords[i] + half_region);
    }
    
    // Recursively sample the region
    std::vector<int> current_coords(ndims, 0);
    detail::sample_region_recursive(data, dims, ndims, start_coords, end_coords, current_coords, 0, result);
    
    return result;
}

/**
 * Get center coordinates for multi-dimensional data
 * @param dims Dimensions of the data [dim0, dim1, ..., dimN-1]
 * @param ndims Number of dimensions
 * @return Vector of center coordinates
 */
inline std::vector<int> get_center_coordinates(const int* dims, int ndims) {
    assert(dims != nullptr);
    assert(ndims > 0);
    
    std::vector<int> center_coords(ndims);
    for (int i = 0; i < ndims; i++) {
        center_coords[i] = dims[i] / 2;
    }
    return center_coords;
}

} // namespace PM

#endif // PM_SAMPLING_HPP_FILE
