find_package(benchmark REQUIRED)

add_executable(costa_benchmark.out costa_benchmark.cpp)
target_link_libraries(costa_benchmark.out
        PRIVATE costa::costa
        PRIVATE MPI::MPI_CXX
        PRIVATE benchmark::benchmark
)
target_include_directories(costa_benchmark.out
        PRIVATE include
)
