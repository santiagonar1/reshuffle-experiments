#include <benchmark/benchmark.h>
#include <chrono>
#include <mpi.h>

#include "benchmark_config.hpp"
#include "null_reporter.hpp"


auto is_root(const MPI_Comm &comm) -> bool;
auto get_rank_id(const MPI_Comm &comm) -> int;
auto get_num_ranks(const MPI_Comm &comm) -> int;

// BLACS declarations
extern "C" {
void Cblacs_pinfo(int *mypnum, int *nprocs);
void Cblacs_get(int context, int what, int *val);
void Cblacs_gridinit(int *context, const char *order, int nprow, int npcol);
void Cblacs_gridinfo(int context, int *nprow, int *npcol, int *myrow, int *mycol);
void Cblacs_gridexit(int context);
void Cblacs_exit(int status);

void descinit_(int *desc, const int *m, const int *n, const int *mb, const int *nb,
               const int *irsrc, const int *icsrc, const int *ictxt, const int *lld, int *info);
int numroc_(const int *n, const int *nb, const int *iproc, const int *isrcproc, const int *nprocs);

void Cpdgemr2d(int m, int n, const double *A, int ia, int ja, const int *descA, double *B, int ib,
               int jb, const int *descB, int context);
}

void gather_benchmark(benchmark::State &state) {
    const auto rank = get_rank_id(MPI_COMM_WORLD);
    constexpr int zero = 0;

    // ***************************************
    // DESCRIBING THE GLOBAL MATRIX GRID
    // ***************************************
    const auto global_num_values_per_dimension = static_cast<int>(state.range(0));

    // ***************************************
    // CREATING THE INITIAL CONTEXT
    // ***************************************
    constexpr int initial_num_processors_per_dimension = 2;
    const int initial_block_size =
            global_num_values_per_dimension / initial_num_processors_per_dimension;


    auto initial_rank_coordinates = std::array<int, 2>{};
    int initial_nprow{};
    int initial_npcol{};

    // ***************************************
    // CREATING THE FINAL CONTEXT
    // ***************************************
    const int final_block_size = global_num_values_per_dimension;

    while (state.KeepRunning()) {
        int initial_context{};
        Cblacs_get(0, 0, &initial_context);
        Cblacs_gridinit(&initial_context, "Row-major", initial_num_processors_per_dimension,
                        initial_num_processors_per_dimension);

        Cblacs_gridinfo(initial_context, &initial_nprow, &initial_npcol,
                        &initial_rank_coordinates[0], &initial_rank_coordinates[1]);

        const int initial_local_rows =
                numroc_(&global_num_values_per_dimension, &initial_block_size,
                        &initial_rank_coordinates[0], &zero, &initial_nprow);
        const int initial_local_cols =
                numroc_(&global_num_values_per_dimension, &initial_block_size,
                        &initial_rank_coordinates[1], &zero, &initial_npcol);

        const auto initial_local_data =
                std::vector<common::SendType>(initial_local_rows * initial_local_cols);

        constexpr int final_num_processor_per_dimension = 1;

        int final_context{};
        Cblacs_get(0, 0, &final_context);
        Cblacs_gridinit(&final_context, "Row-major", final_num_processor_per_dimension,
                        final_num_processor_per_dimension);

        const int final_local_dim = rank == 0 ? global_num_values_per_dimension : 0;

        auto final_local_data = std::vector<common::SendType>(final_local_dim * final_local_dim);

        const auto start = std::chrono::high_resolution_clock::now();

        auto initial_descriptor = std::array<int, 9>{};
        int info{};
        const int initial_lld = std::max(1, initial_local_rows);
        descinit_(initial_descriptor.data(), &global_num_values_per_dimension,
                  &global_num_values_per_dimension, &initial_block_size, &initial_block_size, &zero,
                  &zero, &initial_context, &initial_lld, &info);

        auto final_descriptor = std::array<int, 9>{};
        final_descriptor[1] = -1;// Invalid context marker for non-participants

        if (rank == 0) {
            const int final_lld = std::max(1, final_local_dim);
            descinit_(final_descriptor.data(), &global_num_values_per_dimension,
                      &global_num_values_per_dimension, &final_block_size, &final_block_size, &zero,
                      &zero, &final_context, &final_lld, &info);
        }

        Cpdgemr2d(global_num_values_per_dimension, global_num_values_per_dimension,
                  initial_local_data.data(), 1, 1, initial_descriptor.data(),
                  final_local_data.data(), 1, 1, final_descriptor.data(), initial_context);


        const auto end = std::chrono::high_resolution_clock::now();

        const auto duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        const auto elapsed_seconds = duration.count();

        double max_elapsed_second{};
        MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);


        state.SetIterationTime(max_elapsed_second);

        // Cleanup
        if (initial_rank_coordinates[0] >= 0) { Cblacs_gridexit(initial_context); }
        if (rank == 0) { Cblacs_gridexit(final_context); }
    }
}

void scatter_benchmark(benchmark::State &state) {
    const auto rank = get_rank_id(MPI_COMM_WORLD);
    constexpr int zero = 0;

    // ***************************************
    // DESCRIBING THE GLOBAL MATRIX GRID
    // ***************************************
    const auto global_num_values_per_dimension = static_cast<int>(state.range(0));

    // CREATING THE INITIAL CONTEXT
    // ***************************************

    const int initial_block_size = global_num_values_per_dimension;

    // ***************************************
    // CREATING THE FINAL CONTEXT
    // ***************************************
    constexpr auto final_num_processors_per_dimension = 2;

    const int final_block_size =
            global_num_values_per_dimension / final_num_processors_per_dimension;


    while (state.KeepRunning()) {
        constexpr auto initial_num_processors_per_dimension = 1;

        int initial_context{};
        Cblacs_get(0, 0, &initial_context);
        Cblacs_gridinit(&initial_context, "Row-major", initial_num_processors_per_dimension,
                        initial_num_processors_per_dimension);

        const auto initial_local_values_per_dimension =
                rank == 0 ? global_num_values_per_dimension : 0;

        auto initial_local_data = std::vector<common::SendType>(initial_local_values_per_dimension *
                                                                initial_local_values_per_dimension);


        int final_context{};
        Cblacs_get(0, 0, &final_context);
        Cblacs_gridinit(&final_context, "Row-major", final_num_processors_per_dimension,
                        final_num_processors_per_dimension);

        // Get grid info for destination
        auto final_rank_coordinates = std::array<int, 2>{};

        int dummy_final_num_processors_per_row{};
        int dummy_final_num_processors_per_column{};

        Cblacs_gridinfo(final_context, &dummy_final_num_processors_per_row,
                        &dummy_final_num_processors_per_column, &final_rank_coordinates[0],
                        &final_rank_coordinates[1]);

        const auto final_local_values_per_row =
                numroc_(&global_num_values_per_dimension, &final_block_size,
                        &final_rank_coordinates[0], &zero, &final_num_processors_per_dimension);
        const auto final_local_values_per_column =
                numroc_(&global_num_values_per_dimension, &final_block_size,
                        &final_rank_coordinates[1], &zero, &final_num_processors_per_dimension);

        auto final_local_data = std::vector<common::SendType>(final_local_values_per_row *
                                                              final_local_values_per_column);

        const auto start = std::chrono::high_resolution_clock::now();

        // Source descriptor (only rank 0 participates)
        auto initial_descriptor = std::array<int, 9>{};
        initial_descriptor[1] = -1;// Invalid context marker

        int info{};

        if (rank == 0) {
            const int initial_lld = initial_local_values_per_dimension;
            descinit_(initial_descriptor.data(), &global_num_values_per_dimension,
                      &global_num_values_per_dimension, &initial_block_size, &initial_block_size,
                      &zero, &zero, &initial_context, &initial_lld, &info);
        }


        // Destination descriptor and local storage
        auto final_descriptor = std::array<int, 9>{};

        const int final_lld = final_local_values_per_column;
        descinit_(final_descriptor.data(), &global_num_values_per_dimension,
                  &global_num_values_per_dimension, &final_block_size, &final_block_size, &zero,
                  &zero, &final_context, &final_lld, &info);

        Cpdgemr2d(global_num_values_per_dimension, global_num_values_per_dimension,
                  initial_local_data.data(), 1, 1, initial_descriptor.data(),
                  final_local_data.data(), 1, 1, final_descriptor.data(), final_context);

        const auto end = std::chrono::high_resolution_clock::now();

        const auto duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        const auto elapsed_seconds = duration.count();

        double max_elapsed_second{};
        MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);


        state.SetIterationTime(max_elapsed_second);

        // Cleanup
        if (final_rank_coordinates[1] >= 0) Cblacs_gridexit(final_context);
        if (rank == 0) Cblacs_gridexit(initial_context);
    }
}

void change_block_size_benchmark(benchmark::State &state) {
    constexpr int zero = 0;

    // ***************************************
    // DESCRIBING THE GLOBAL MATRIX GRID
    // ***************************************
    const auto global_num_values_per_dimension = static_cast<int>(state.range(0));

    // ***************************************
    // INITIAL CONTEXT: 2x2 grid, block size 2
    // ***************************************
    constexpr auto initial_num_proc_rows = 2;
    constexpr auto initial_num_proc_cols = 2;
    constexpr auto initial_block_size = 2;


    // ***************************************
    // FINAL CONTEXT: 2x2 grid, block size 4
    // ***************************************
    constexpr auto final_num_proc_rows = 2;
    constexpr auto final_num_proc_cols = 2;
    constexpr auto final_block_size = 4;


    while (state.KeepRunning()) {
        int initial_context{};
        Cblacs_get(0, 0, &initial_context);
        Cblacs_gridinit(&initial_context, "Row-major", initial_num_proc_rows,
                        initial_num_proc_cols);

        auto initial_rank_coords = std::array<int, 2>{};
        int dummy_nprow{}, dummy_npcol{};
        Cblacs_gridinfo(initial_context, &dummy_nprow, &dummy_npcol, &initial_rank_coords[0],
                        &initial_rank_coords[1]);

        const auto initial_local_rows =
                numroc_(&global_num_values_per_dimension, &initial_block_size,
                        &initial_rank_coords[0], &zero, &initial_num_proc_rows);
        const auto initial_local_cols =
                numroc_(&global_num_values_per_dimension, &initial_block_size,
                        &initial_rank_coords[1], &zero, &initial_num_proc_cols);

        int final_context{};
        Cblacs_get(0, 0, &final_context);
        Cblacs_gridinit(&final_context, "Row-major", final_num_proc_rows, final_num_proc_cols);

        auto final_rank_coords = std::array<int, 2>{};
        Cblacs_gridinfo(final_context, &dummy_nprow, &dummy_npcol, &final_rank_coords[0],
                        &final_rank_coords[1]);

        const auto final_local_rows = numroc_(&global_num_values_per_dimension, &final_block_size,
                                              &final_rank_coords[0], &zero, &final_num_proc_rows);
        const auto final_local_cols = numroc_(&global_num_values_per_dimension, &final_block_size,
                                              &final_rank_coords[1], &zero, &final_num_proc_cols);

        auto initial_local_data =
                std::vector<common::SendType>(initial_local_rows * initial_local_cols);
        auto final_local_data = std::vector<common::SendType>(final_local_rows * final_local_cols);

        const auto start = std::chrono::high_resolution_clock::now();

        auto initial_descriptor = std::array<int, 9>{};
        int info{};
        // LLD = number of local rows (column-major leading dimension)
        const int initial_lld = std::max(1, initial_local_rows);
        descinit_(initial_descriptor.data(), &global_num_values_per_dimension,
                  &global_num_values_per_dimension, &initial_block_size, &initial_block_size, &zero,
                  &zero, &initial_context, &initial_lld, &info);


        auto final_descriptor = std::array<int, 9>{};
        // LLD = number of local rows
        const int final_lld = std::max(1, final_local_rows);
        descinit_(final_descriptor.data(), &global_num_values_per_dimension,
                  &global_num_values_per_dimension, &final_block_size, &final_block_size, &zero,
                  &zero, &final_context, &final_lld, &info);


        Cpdgemr2d(global_num_values_per_dimension, global_num_values_per_dimension,
                  initial_local_data.data(), 1, 1, initial_descriptor.data(),
                  final_local_data.data(), 1, 1, final_descriptor.data(), final_context);

        const auto end = std::chrono::high_resolution_clock::now();

        const auto duration =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        const auto elapsed_seconds = duration.count();

        double max_elapsed_second{};
        MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);


        state.SetIterationTime(max_elapsed_second);

        Cblacs_gridexit(final_context);
        Cblacs_gridexit(initial_context);
    }
}

BENCHMARK(gather_benchmark)
        ->UseManualTime()
        ->DenseRange(common::START, common::LIMIT, common::STEP);
BENCHMARK(scatter_benchmark)
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