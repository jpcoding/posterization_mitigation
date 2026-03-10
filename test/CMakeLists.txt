add_executable(test_3d test_3d.cpp)
target_link_libraries(test_3d PUBLIC OpenMP::OpenMP_CXX )

add_executable(test_2d test_2d.cpp)
target_link_libraries(test_2d PUBLIC OpenMP::OpenMP_CXX )

add_executable(direct_quantize direct_quantize.cpp)
# add_executable(test_3d_idw test_3d_idw.cpp)
# target_include_directories(test_3d_idw PUBLIC "/scratch/pji228/gittmp/alglib/install/include")
# target_link_directories(test_3d_idw PUBLIC "/scratch/pji228/gittmp/alglib/install/lib64")
# target_link_libraries(test_3d_idw PUBLIC alglib OpenMP::OpenMP_CXX)
# add_executable(test_3d_rbf test_3d_rbf.cpp)
# target_include_directories(test_3d_rbf PUBLIC "/scratch/pji228/gittmp/alglib/install/include")
# target_link_directories(test_3d_rbf PUBLIC "/scratch/pji228/gittmp/alglib/install/lib64")
# target_link_libraries(test_3d_rbf PUBLIC alglib OpenMP::OpenMP_CXX)

add_executable(test_quantize_and_edt test_quantize_and_edt.cpp)
target_include_directories(test_quantize_and_edt PRIVATE /usr/include/openblas)
target_link_libraries(test_quantize_and_edt OpenMP::OpenMP_CXX )

add_executable(test_3d_as_slice test_3d_as_slice.cpp)
target_link_libraries(test_3d_as_slice OpenMP::OpenMP_CXX )

add_executable(test_3d_slices test_3d_slices.cpp)
target_link_libraries(test_3d_slices OpenMP::OpenMP_CXX )

add_executable(test_ssim test_ssim.cpp)
target_include_directories(test_ssim PRIVATE /usr/include/openblas)
target_link_libraries(test_ssim OpenMP::OpenMP_CXX )

add_executable(test_ssim_file test_ssim_file.cpp)
target_link_libraries(test_ssim_file OpenMP::OpenMP_CXX )

find_package(BLAS QUIET)
if(BLAS_FOUND)
    add_executable(test_ssim_blas test_ssim_blas.cpp)
    target_include_directories(test_ssim_blas PRIVATE /usr/include/openblas)
    target_link_libraries(test_ssim_blas OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})
    message(STATUS "BLAS found: building test_ssim_blas")
else()
    message(STATUS "BLAS not found: skipping test_ssim_blas")
endif()

add_executable(test_field_error_bounds test_field_error_bounds.cpp)
target_link_libraries(test_field_error_bounds OpenMP::OpenMP_CXX )

#define the install process 
install(TARGETS 
    test_3d 
    test_2d 
    direct_quantize 
    test_quantize_and_edt 
    test_3d_as_slice 
    test_3d_slices 
    test_ssim 
    test_ssim_file
    test_field_error_bounds
    RUNTIME DESTINATION bin/test/
)