add_executable(basic_costa.out basic_costa.cpp)
target_compile_features(basic_costa.out PUBLIC cxx_std_23)

target_link_libraries(basic_costa.out
        PRIVATE MPI::MPI_CXX
        PRIVATE costa::costa
)

add_executable(gather_costa.out gather_costa.cpp)
target_compile_features(gather_costa.out PUBLIC cxx_std_23)

target_link_libraries(gather_costa.out
        PRIVATE MPI::MPI_CXX
        PRIVATE costa::costa
)

add_executable(scatter_costa.out scatter_costa.cpp)
target_compile_features(scatter_costa.out PUBLIC cxx_std_23)

target_link_libraries(scatter_costa.out
        PRIVATE MPI::MPI_CXX
        PRIVATE costa::costa
)

add_executable(change_block_size_costa.out change_block_size_costa.cpp)
target_compile_features(change_block_size_costa.out PUBLIC cxx_std_23)

target_link_libraries(change_block_size_costa.out
        PRIVATE MPI::MPI_CXX
        PRIVATE costa::costa
)

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
