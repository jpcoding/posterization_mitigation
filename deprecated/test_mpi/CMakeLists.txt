add_executable(test_edt_mpi test_edt_mpi.cpp)
target_link_libraries(test_edt_mpi PRIVATE MPI::MPI_CXX)

add_executable(test_quantizer_mpi test_quantizer_mpi.cpp)
target_link_libraries(test_quantizer_mpi PRIVATE MPI::MPI_CXX)

add_executable(test_exact_parallel test_exact_parallel.cpp)
target_link_libraries(test_exact_parallel PRIVATE MPI::MPI_CXX)

add_executable(test_approximate_parallel test_approximate_parallel.cpp)
target_link_libraries(test_approximate_parallel PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX )

add_executable(test_edt_mpi_function test_edt_mpi_function.cpp)
target_link_libraries(test_edt_mpi_function PRIVATE MPI::MPI_CXX)

add_executable(test_ssim_mpi test_ssim_mpi.cpp)
target_link_libraries(test_ssim_mpi PRIVATE MPI::MPI_CXX)

add_executable(test_ssim_mpi_merged_file test_ssim_mpi_merged_file.cpp)
target_link_libraries(test_ssim_mpi_merged_file PRIVATE MPI::MPI_CXX)

add_executable(test_merge_file test_merge_file.cpp)
target_link_libraries(test_merge_file PRIVATE MPI::MPI_CXX)

add_executable(test_compensation_parallel test_compensation_parallel.cpp)
target_link_libraries(test_compensation_parallel PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX )

add_executable(test_data_exchange test_data_exchange.cpp)
target_link_libraries(test_data_exchange PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX )

add_executable(test_decompose_file test_decompose_file.cpp)
target_link_libraries(test_decompose_file PRIVATE MPI::MPI_CXX )

add_executable(test_sequantial test_sequantial.cpp)
target_link_libraries(test_sequantial OpenMP::OpenMP_CXX MPI::MPI_CXX )

add_executable(test_mpi_ranks test_mpi_ranks.cpp)
target_link_libraries(test_mpi_ranks OpenMP::OpenMP_CXX MPI::MPI_CXX )

add_executable(test_embarrassingly_parallel  test_embarrassingly_parallel.cpp)
target_link_libraries(test_embarrassingly_parallel  MPI::MPI_CXX OpenMP::OpenMP_CXX)


# Installation process
install(TARGETS 
    test_edt_mpi 
    test_quantizer_mpi 
    test_exact_parallel 
    test_approximate_parallel 
    test_edt_mpi_function 
    test_ssim_mpi 
    test_ssim_mpi_merged_file 
    test_merge_file 
    test_compensation_parallel 
    test_data_exchange 
    test_decompose_file 
    test_sequantial 
    test_mpi_ranks 
    test_embarrassingly_parallel 
    RUNTIME DESTINATION bin/test_mpi/
)