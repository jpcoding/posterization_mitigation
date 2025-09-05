#ifndef SMOOTH_DISTANCE_HPP
#define SMOOTH_DISTANCE_HPP

#include <vector>

namespace Smoothing {

namespace detail {

/**
 * @brief (Internal) Applies a 1D forward-and-backward IIR filter to a line of data.
 * @tparam T The floating point type (float or double).
 * @param data_line Pointer to the start of the data.
 * @param size The number of elements in the line.
 * @param alpha The smoothing factor (e.g., 0.2). Smaller alpha means more smoothing.
 */
template <typename T>
void apply_1d_filter(T* data_line, int size, double alpha) {
    if (size <= 1) return;

    // --- Forward pass ---
    for (int i = 1; i < size; ++i) {
        T prev_val = data_line[i - 1];
        T& curr_val = data_line[i];
        // The arithmetic is done in double precision, then cast back to T
        curr_val = static_cast<T>(prev_val * (1.0 - alpha) + curr_val * alpha);
    }

    // --- Backward pass ---
    for (int i = size - 2; i >= 0; --i) {
        T next_val = data_line[i + 1];
        T& curr_val = data_line[i];
        curr_val = static_cast<T>(next_val * (1.0 - alpha) + curr_val * alpha);
    }
}

} // namespace detail

/**
 * @brief Smooths a 3D array using a fast, separable IIR filter.
 *
 * This function applies a 1D smoothing filter sequentially along the Z, Y, and X
 * axes to approximate a 3D Gaussian blur in linear time.
 *
 * @tparam T The floating point type (float or double).
 * @param array_3d Pointer to the 3D data to be smoothed in place.
 * @param dims A vector containing the dimensions {dimX, dimY, dimZ}.
 * @param alpha The smoothing factor (e.g., 0.2). A smaller value creates more smoothing.
 */
template <typename T>
void smooth_3d_array(T* array_3d, const std::vector<int>& dims, double alpha) {
    if (dims.size() != 3) {
        return; // Or handle error appropriately
    }
    const int dimX = dims[0];
    const int dimY = dims[1];
    const int dimZ = dims[2];

    // 1. Smooth along Z-axis (contiguous memory access)
    for (int i = 0; i < dimX; ++i) {
        for (int j = 0; j < dimY; ++j) {
            T* start_of_line = &array_3d[i * dimY * dimZ + j * dimZ];
            detail::apply_1d_filter(start_of_line, dimZ, alpha);
        }
    }

    // 2. Smooth along Y-axis (non-contiguous, requires temporary buffer)
    std::vector<T> temp_buffer(dimY);
    for (int i = 0; i < dimX; ++i) {
        for (int k = 0; k < dimZ; ++k) {
            // Copy Y-column into the contiguous buffer
            for (int j = 0; j < dimY; ++j) {
                temp_buffer[j] = array_3d[i * dimY * dimZ + j * dimZ + k];
            }
            // Filter the buffer
            detail::apply_1d_filter(temp_buffer.data(), dimY, alpha);
            // Copy the smoothed data back
            for (int j = 0; j < dimY; ++j) {
                array_3d[i * dimY * dimZ + j * dimZ + k] = temp_buffer[j];
            }
        }
    }

    // 3. Smooth along X-axis (non-contiguous, requires temporary buffer)
    temp_buffer.resize(dimX);
    for (int j = 0; j < dimY; ++j) {
        for (int k = 0; k < dimZ; ++k) {
            // Copy X-row into the contiguous buffer
            for (int i = 0; i < dimX; ++i) {
                temp_buffer[i] = array_3d[i * dimY * dimZ + j * dimZ + k];
            }
            // Filter the buffer
            detail::apply_1d_filter(temp_buffer.data(), dimX, alpha);
            // Copy the smoothed data back
            for (int i = 0; i < dimX; ++i) {
                array_3d[i * dimY * dimZ + j * dimZ + k] = temp_buffer[i];
            }
        }
    }
}

} // namespace Smoothing

#endif // SMOOTH_DISTANCE_HPP