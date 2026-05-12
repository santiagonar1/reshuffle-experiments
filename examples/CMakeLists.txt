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

function(compile_scalapack_code cpp_file binary_name)
    add_executable(${binary_name} ${cpp_file})
    target_compile_features(${binary_name} PUBLIC cxx_std_23)

    target_link_libraries(${binary_name}
            PRIVATE MPI::MPI_CXX
            PRIVATE SCALAPACK::SCALAPACK
    )
endfunction()

compile_scalapack_code(scatter_scalapack.cpp scatter_scalapack.out)
compile_scalapack_code(gather_scalapack.cpp gather_scalapack.out)
compile_scalapack_code(change_block_size_scalapack.cpp change_block_size_scalapack.out)
