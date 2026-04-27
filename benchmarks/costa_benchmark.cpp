#include <benchmark/benchmark.h>
#include <mpi.h>

#include "null_reporter.hpp"

auto is_root(const MPI_Comm &comm) -> bool;
auto get_rank_id(const MPI_Comm &comm) -> int;
auto get_num_ranks(const MPI_Comm &comm) -> int;


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