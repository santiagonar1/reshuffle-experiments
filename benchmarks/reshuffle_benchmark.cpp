#include <benchmark/benchmark.h>
#include <chrono>
#include <mpi.h>

#include <reshuffle.hpp>

#include "benchmark_config.hpp"
#include "null_reporter.hpp"

auto is_root(const MPI_Comm &comm) -> bool;
auto get_rank_id(const MPI_Comm &comm) -> int;
auto get_num_ranks(const MPI_Comm &comm) -> int;

void gather_benchmark(benchmark::State &state) {
    const auto global_num_values_per_dimension = static_cast<int>(state.range(0));
    const auto global_dimensions =
            reshuffle::Dimensions{global_num_values_per_dimension, global_num_values_per_dimension};

    // ***************************************
    // CREATING THE INITIAL LAYOUT OBJECT
    // ***************************************
    const auto initial_processor_grid =
            reshuffle::ProcessorGrid{gather::INITIAL_NUM_PROCESSORS_PER_DIMENSION,
                                     gather::INITIAL_NUM_PROCESSORS_PER_DIMENSION};
    const auto initial_distribution =
            reshuffle::distribution::BlockWise{global_dimensions, initial_processor_grid};

    const auto initial_local_values_per_dimension =
            global_num_values_per_dimension / gather::INITIAL_NUM_PROCESSORS_PER_DIMENSION;

    const auto initial_local_data = std::vector<common::SendType>(
            initial_local_values_per_dimension * initial_local_values_per_dimension);

    // ***************************************
    // CREATING THE FINAL LAYOUT OBJECT
    // ***************************************
    const auto final_distribution =
            reshuffle::distribution::get_all_values_in_root(global_dimensions);

    while (state.KeepRunning()) {
        const auto start = std::chrono::high_resolution_clock::now();

        const auto initial_context = reshuffle::Context{initial_distribution, MPI_COMM_WORLD};

        const auto final_context = reshuffle::Context{final_distribution, MPI_COMM_WORLD};

        // If this is made const, benchmark::DoNotOptimize generates a warning. Not sure why.
        auto data = reshuffle::shuffle(std::mdspan{initial_local_data.data(),
                                                   initial_local_values_per_dimension,
                                                   initial_local_values_per_dimension},
                                       initial_context, final_context);

        const auto end = std::chrono::high_resolution_clock::now();

        benchmark::DoNotOptimize(data);

        const auto duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        const auto elapsed_seconds = duration.count();

        double max_elapsed_second{};
        MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);


        state.SetIterationTime(max_elapsed_second);
    }
}

void scatter_benchmark(benchmark::State &state) {
    const auto rank = get_rank_id(MPI_COMM_WORLD);

    const auto global_num_values_per_dimension = static_cast<int>(state.range(0));
    const auto global_dimensions =
            reshuffle::Dimensions{global_num_values_per_dimension, global_num_values_per_dimension};

    // ***************************************
    // CREATING THE INITIAL LAYOUT OBJECT
    // ***************************************
    const auto initial_local_values_per_dimension = rank == 0 ? global_num_values_per_dimension : 0;

    const auto initial_local_data = std::vector<common::SendType>(
            initial_local_values_per_dimension * initial_local_values_per_dimension);

    const auto initial_processor_grid =
            reshuffle::ProcessorGrid{scatter::INITIAL_NUM_PROCESSORS_PER_DIMENSION,
                                     scatter::INITIAL_NUM_PROCESSORS_PER_DIMENSION};
    const auto initial_distribution =
            reshuffle::distribution::BlockWise{global_dimensions, initial_processor_grid};

    // ***************************************
    // CREATING THE FINAL LAYOUT OBJECT
    // ***************************************
    const auto final_processor_grid =
            reshuffle::ProcessorGrid{scatter::FINAL_NUM_PROCESSORS_PER_DIMENSION,
                                     scatter::FINAL_NUM_PROCESSORS_PER_DIMENSION};
    const auto final_distribution =
            reshuffle::distribution::BlockWise{global_dimensions, final_processor_grid};


    while (state.KeepRunning()) {
        const auto start = std::chrono::high_resolution_clock::now();

        const auto initial_context = reshuffle::Context{initial_distribution, MPI_COMM_WORLD};

        const auto final_context = reshuffle::Context{final_distribution, MPI_COMM_WORLD};

        // If this is made const, benchmark::DoNotOptimize generates a warning. Not sure why.
        auto data = reshuffle::shuffle(std::mdspan{initial_local_data.data(),
                                                   initial_local_values_per_dimension,
                                                   initial_local_values_per_dimension},
                                       initial_context, final_context);

        const auto end = std::chrono::high_resolution_clock::now();

        benchmark::DoNotOptimize(data);

        const auto duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        const auto elapsed_seconds = duration.count();

        double max_elapsed_second{};
        MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);


        state.SetIterationTime(max_elapsed_second);
    }
}

void change_block_size_benchmark(benchmark::State &state) {
    const auto global_num_values_per_dimension = static_cast<int>(state.range(0));
    const auto global_dimensions =
            reshuffle::Dimensions{global_num_values_per_dimension, global_num_values_per_dimension};

    // ***************************************
    // CREATING THE INITIAL LAYOUT OBJECT
    // ***************************************
    const auto initial_local_values_per_dimension =
            global_num_values_per_dimension / change_block::INITIAL_NUM_PROCESSORS_PER_DIMENSION;

    const auto initial_local_data = std::vector<common::SendType>(
            initial_local_values_per_dimension * initial_local_values_per_dimension);

    const auto initial_processor_grid =
            reshuffle::ProcessorGrid{change_block::INITIAL_NUM_PROCESSORS_PER_DIMENSION,
                                     change_block::INITIAL_NUM_PROCESSORS_PER_DIMENSION};

    const auto initial_block_sizes = reshuffle::Dimensions{change_block::INITIAL_BLOCK_SIZE,
                                                           change_block::INITIAL_BLOCK_SIZE};

    const auto initial_distribution = reshuffle::distribution::BlockCyclic{
            global_dimensions, initial_block_sizes, initial_processor_grid};

    // ***************************************
    // CREATING THE FINAL LAYOUT OBJECT
    // ***************************************
    const auto final_processor_grid =
            reshuffle::ProcessorGrid{change_block::FINAL_NUM_PROCESSORS_PER_DIMENSION,
                                     change_block::FINAL_NUM_PROCESSORS_PER_DIMENSION};

    const auto final_block_sizes =
            reshuffle::Dimensions{change_block::FINAL_BLOCK_SIZE, change_block::FINAL_BLOCK_SIZE};

    const auto final_distribution = reshuffle::distribution::BlockCyclic{
            global_dimensions, final_block_sizes, final_processor_grid};


    while (state.KeepRunning()) {
        const auto start = std::chrono::high_resolution_clock::now();

        const auto initial_context = reshuffle::Context{initial_distribution, MPI_COMM_WORLD};

        const auto final_context = reshuffle::Context{final_distribution, MPI_COMM_WORLD};

        // If this is made const, benchmark::DoNotOptimize generates a warning. Not sure why.
        auto data = reshuffle::shuffle(std::mdspan{initial_local_data.data(),
                                                   initial_local_values_per_dimension,
                                                   initial_local_values_per_dimension},
                                       initial_context, final_context);

        const auto end = std::chrono::high_resolution_clock::now();

        benchmark::DoNotOptimize(data);

        const auto duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        const auto elapsed_seconds = duration.count();

        double max_elapsed_second{};
        MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);


        state.SetIterationTime(max_elapsed_second);
    }
}

BENCHMARK(gather_benchmark)
        ->UseManualTime()
        ->DenseRange(common::START, common::LIMIT, common::STEP);
BENCHMARK(scatter_benchmark)
        ->UseManualTime()
        ->DenseRange(common::START, common::LIMIT, common::STEP);
BENCHMARK(change_block_size_benchmark)
        ->UseManualTime()
        ->DenseRange(common::START, common::LIMIT, common::STEP);

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