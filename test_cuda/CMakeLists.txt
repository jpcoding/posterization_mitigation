set(CUDA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/cuda)

add_executable(test_edt_cuda test_edt.cu)
add_executable(test_quantizer_cuda test_quantizer.cu)
add_executable(test_boundary_cuda test_boundary.cu)
add_executable(test_sz3_quantizer_cuda test_sz3_quantizer.cpp)
add_executable(test_sign_map_cuda test_sign_map.cu)
add_executable(test_compensation_cuda test_compensation.cu)
add_executable(test_edt_bf_cuda test_edt_bf.cu)

set(CUDA_TARGETS
    test_edt_cuda test_quantizer_cuda test_boundary_cuda
    test_sz3_quantizer_cuda test_sign_map_cuda test_compensation_cuda
    test_edt_bf_cuda)

foreach(target ${CUDA_TARGETS})
    target_include_directories(${target} BEFORE PRIVATE ${CUDA_INCLUDE_DIR})
endforeach()
