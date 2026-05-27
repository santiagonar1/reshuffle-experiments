function(compile_costa_code cpp_file binary_name)
    add_executable(${binary_name} ${cpp_file})
    target_compile_features(${binary_name} PUBLIC cxx_std_23)

    target_link_libraries(${binary_name}
            PRIVATE MPI::MPI_CXX
            PRIVATE costa::costa
    )
endfunction()


compile_costa_code(costa_basic.cpp costa_basic.out)
compile_costa_code(costa_gather.cpp costa_gather.out)
compile_costa_code(costa_scatter.cpp costa_scatter.out)
compile_costa_code(costa_change_block_size.cpp costa_change_block_size.out)

function(compile_scalapack_code cpp_file binary_name)
    add_executable(${binary_name} ${cpp_file})
    target_compile_features(${binary_name} PUBLIC cxx_std_23)

    target_link_libraries(${binary_name}
            PRIVATE MPI::MPI_CXX
            PRIVATE SCALAPACK::SCALAPACK
    )
endfunction()

compile_scalapack_code(scalapack_scatter.cpp scalapack_scatter.out)
compile_scalapack_code(scalapack_gather.cpp scalapack_gather.out)
compile_scalapack_code(scalapack_change_block_size.cpp scalapack_change_block_size.out)
