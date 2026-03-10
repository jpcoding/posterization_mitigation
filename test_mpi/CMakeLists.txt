add_executable(test_edt_mpi test_edt_mpi.cpp)
target_link_libraries(test_edt_mpi PRIVATE MPI::MPI_CXX)
add_executable(test_quantizer_mpi test_quantizer_mpi.cpp)
target_link_libraries(test_quantizer_mpi PRIVATE MPI::MPI_CXX)
add_executable(test_boundary_mpi test_boundary_mpi.cpp)
target_link_libraries(test_boundary_mpi PRIVATE MPI::MPI_CXX)
add_executable(test_edt_mpi_function test_edt_mpi_function.cpp)
target_link_libraries(test_edt_mpi_function PRIVATE MPI::MPI_CXX)
add_executable(test_ssim_mpi test_ssim_mpi.cpp)
target_include_directories(test_ssim_mpi PRIVATE /usr/include/openblas)
target_link_libraries(test_ssim_mpi PRIVATE MPI::MPI_CXX)
add_executable(test_ssim_mpi_merged_file test_ssim_mpi_merged_file.cpp)
target_include_directories(test_ssim_mpi_merged_file PRIVATE /usr/include/openblas)
target_link_libraries(test_ssim_mpi_merged_file PRIVATE MPI::MPI_CXX)
add_executable(test_ssim_mpi_halo test_ssim_mpi_halo.cpp)
target_include_directories(test_ssim_mpi_halo PRIVATE /usr/include/openblas)
target_link_libraries(test_ssim_mpi_halo PRIVATE MPI::MPI_CXX)
add_executable(test_merge_file test_merge_file.cpp)
target_link_libraries(test_merge_file PRIVATE MPI::MPI_CXX)

add_executable(test_compensation_parallel test_compensation_parallel.cpp)
target_link_libraries(test_compensation_parallel PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX )

add_executable(test_data_exchange test_data_exchange.cpp)
target_link_libraries(test_data_exchange PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX )

add_executable(test_compensation_opt test_compensation_opt.cpp)
target_link_libraries(test_compensation_opt PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX )

add_executable(test_decompose_file test_decompose_file.cpp)
target_link_libraries(test_decompose_file PRIVATE MPI::MPI_CXX )


add_executable(test_sequantial test_sequantial.cpp)
target_link_libraries(test_sequantial OpenMP::OpenMP_CXX MPI::MPI_CXX )

add_executable(test_mpi_ranks test_mpi_ranks.cpp)
target_link_libraries(test_mpi_ranks OpenMP::OpenMP_CXX MPI::MPI_CXX )

add_executable(test_embassingly_parallel  test_embassingly_parallel.cpp)
target_link_libraries(test_embassingly_parallel  MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_frequent test_frequent.cpp)
