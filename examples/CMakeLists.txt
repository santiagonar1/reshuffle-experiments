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
