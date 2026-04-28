#include <benchmark/benchmark.h>
#include <chrono>
#include <mpi.h>

#include <costa/grid2grid/transform.hpp>
#include <costa/layout.hpp>

#include "null_reporter.hpp"

using SendType = double;

auto is_root(const MPI_Comm &comm) -> bool;
auto get_rank_id(const MPI_Comm &comm) -> int;
auto get_num_ranks(const MPI_Comm &comm) -> int;

void gather_benchmark(benchmark::State &state) {
    const auto rank = get_rank_id(MPI_COMM_WORLD);

    // ***************************************
    // DESCRIBING THE GLOBAL MATRIX GRID
    // ***************************************
    const auto global_num_values_per_dimension = static_cast<int>(state.range(0));

    constexpr auto submatrix_start = std::array{1, 1};// (1-based, required by costa)
    constexpr auto coordinates_initial_rank = std::array{0, 0};


    // ***************************************
    // CREATING THE INITIAL LAYOUT OBJECT
    // ***************************************
    constexpr auto initial_num_processors_per_dimension = 2;

    const auto initial_block_size =
            global_num_values_per_dimension / initial_num_processors_per_dimension;

    const auto initial_local_values_per_dimension =
            global_num_values_per_dimension / initial_num_processors_per_dimension;

    // ***************************************
    // CREATING THE FINAL LAYOUT OBJECT
    // ***************************************
    const auto final_local_values_per_dimension = rank == 0 ? global_num_values_per_dimension : 0;


    while (state.KeepRunning()) {
        constexpr auto final_num_processors_per_dimension = 1;
        const auto final_block_size = global_num_values_per_dimension;
        constexpr auto local_data_ordering = 'R';
        constexpr auto processor_grid_ordering = 'R';

        auto initial_local_data = std::vector<SendType>(initial_local_values_per_dimension *
                                                        initial_local_values_per_dimension);

        auto final_local_data = std::vector<SendType>(final_local_values_per_dimension *
                                                      final_local_values_per_dimension);

        const auto start = std::chrono::high_resolution_clock::now();

        auto initial_layout = costa::block_cyclic_layout(
                global_num_values_per_dimension, global_num_values_per_dimension,
                initial_block_size, initial_block_size, submatrix_start[0], submatrix_start[1],
                global_num_values_per_dimension, global_num_values_per_dimension,
                initial_num_processors_per_dimension, initial_num_processors_per_dimension,
                processor_grid_ordering, coordinates_initial_rank[0], coordinates_initial_rank[1],
                &initial_local_data[0], initial_local_values_per_dimension, local_data_ordering,
                rank);

        auto final_layout = costa::block_cyclic_layout(
                global_num_values_per_dimension, global_num_values_per_dimension, final_block_size,
                final_block_size, submatrix_start[0], submatrix_start[1],
                global_num_values_per_dimension, global_num_values_per_dimension,
                final_num_processors_per_dimension, final_num_processors_per_dimension,
                processor_grid_ordering, coordinates_initial_rank[0], coordinates_initial_rank[1],
                &final_local_data[0], final_local_values_per_dimension, local_data_ordering, rank);

        costa::transform<SendType>(initial_layout, final_layout, MPI_COMM_WORLD);

        const auto end = std::chrono::high_resolution_clock::now();

        const auto duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        const auto elapsed_seconds = duration.count();

        double max_elapsed_second{};
        MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);


        state.SetIterationTime(max_elapsed_second);
    }
}


constexpr auto START = 10;
constexpr auto LIMIT = 100;
constexpr auto STEP = 10;

BENCHMARK(gather_benchmark)->UseManualTime()->DenseRange(START, LIMIT, STEP);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    benchmark::Initialize(&argc, argv);

    if (is_root(MPI_COMM_WORLD))
        // root process will use a reporter from the usual set provided by
        // ::benchmark
        benchmark::RunSpecifiedBenchmarks();
    else {
        // reporting from other processes is disabled by passing a custom reporter
        NullReporter null;
        benchmark::RunSpecifiedBenchmarks(&null);
    }

    MPI_Finalize();
    return 0;
}


auto is_root(const MPI_Comm &comm) -> bool { return get_rank_id(comm) == 0; }

auto get_rank_id(const MPI_Comm &comm) -> int {
    int id{};
    MPI_Comm_rank(comm, &id);

    return id;
}

auto get_num_ranks(const MPI_Comm &comm) -> int {
    int num_ranks{};
    MPI_Comm_size(comm, &num_ranks);

    return num_ranks;
}