function(compile_costa_code cpp_file binary_name)
    add_executable(${binary_name} ${cpp_file})
    target_compile_features(${binary_name} PUBLIC cxx_std_23)

    target_link_libraries(${binary_name}
            PRIVATE MPI::MPI_CXX
            PRIVATE costa::costa
    )
endfunction()


compile_costa_code(basic_costa.cpp basic_costa.out)
compile_costa_code(gather_costa.cpp gather_costa.out)
compile_costa_code(scatter_costa.cpp scatter_costa.out)
compile_costa_code(change_block_size_costa.cpp change_block_size_costa.out)

add_executable(scatter_scalapack.out scatter_scalapack.cpp)
target_compile_features(scatter_scalapack.out PUBLIC cxx_std_23)

target_link_libraries(scatter_scalapack.out
        PRIVATE MPI::MPI_CXX
        PRIVATE SCALAPACK::SCALAPACK
)

add_executable(gather_scalapack.out gather_scalapack.cpp)
target_compile_features(gather_scalapack.out PUBLIC cxx_std_23)

target_link_libraries(gather_scalapack.out
        PRIVATE MPI::MPI_CXX
        PRIVATE SCALAPACK::SCALAPACK
)

add_executable(change_block_size_scalapack.out change_block_size_scalapack.cpp)
target_compile_features(change_block_size_scalapack.out PUBLIC cxx_std_23)

target_link_libraries(change_block_size_scalapack.out
        PRIVATE MPI::MPI_CXX
        PRIVATE SCALAPACK::SCALAPACK
)
