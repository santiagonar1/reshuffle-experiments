#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mdspan>
#include <mpi.h>
#include <vector>

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

auto execute_sequentially(const std::function<void()> &f, MPI_Comm comm) -> void {
    int rank{};
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        int num_procs{};
        MPI_Comm_size(comm, &num_procs);

        f();
        for (int i = 1; i < num_procs; ++i) {
            int dummy = 0;
            MPI_Send(&dummy, 1, MPI_INT, i, 0, comm);
            MPI_Recv(&dummy, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
        }
    } else {
        int dummy = 0;
        MPI_Recv(&dummy, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        f();
        MPI_Send(&dummy, 1, MPI_INT, 0, 0, comm);
    }
}

template<typename Extents>
auto print_matrix(std::mdspan<const double, Extents> values, const int rank) -> void {
    std::cout << "Rank: " << rank << std::endl;

    if (values.empty()) {
        std::cout << "Empty matrix" << std::endl;
        return;
    }

    for (std::size_t i = 0; i < values.extent(0); ++i) {
        for (std::size_t j = 0; j < values.extent(1); ++j) {
            std::cout << std::setw(4) << values[i, j] << " ";
        }
        std::cout << std::endl;
    }
}

template<typename Extents>
auto print_matrix(std::mdspan<const double, Extents> values, MPI_Comm comm) -> void {
    int rank{};
    MPI_Comm_rank(comm, &rank);

    auto print_function = [values, rank]() -> void { print_matrix(values, rank); };
    execute_sequentially(print_function, comm);

    MPI_Barrier(comm);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    const auto comm = MPI_COMM_WORLD;

    int rank{};
    int num_procs{};
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    constexpr int zero = 0;

    if (num_procs != 4) {
        std::cerr << "ERROR: This demo requires 4 MPI ranks " << num_procs << std::endl;
        MPI_Abort(comm, 1);
    }

    // ***************************************
    // DESCRIBING THE GLOBAL MATRIX GRID
    // ***************************************
    constexpr int global_num_values_per_dimension = 8;

    // ***************************************
    // CREATING THE INITIAL CONTEXT
    // ***************************************
    constexpr int initial_num_processors_per_dimension = 2;
    constexpr int initial_block_size =
            global_num_values_per_dimension / initial_num_processors_per_dimension;

    int initial_context{};
    Cblacs_get(0, 0, &initial_context);
    Cblacs_gridinit(&initial_context, "Row-major", initial_num_processors_per_dimension,
                    initial_num_processors_per_dimension);

    auto initial_rank_coordinates = std::array<int, 2>{};
    int initial_nprow{};
    int initial_npcol{};

    Cblacs_gridinfo(initial_context, &initial_nprow, &initial_npcol, &initial_rank_coordinates[0],
                    &initial_rank_coordinates[1]);

    const int initial_local_rows = numroc_(&global_num_values_per_dimension, &initial_block_size,
                                           &initial_rank_coordinates[0], &zero, &initial_nprow);
    const int initial_local_cols = numroc_(&global_num_values_per_dimension, &initial_block_size,
                                           &initial_rank_coordinates[1], &zero, &initial_npcol);

    auto initial_local_data = std::vector<double>(initial_local_rows * initial_local_cols, rank);

    if (rank == 0) { std::cout << "------- Before data redistribution ----------\n\n"; }
    print_matrix(std::mdspan{std::as_const(initial_local_data).data(),
                             static_cast<std::size_t>(initial_local_rows),
                             static_cast<std::size_t>(initial_local_cols)},
                 comm);

    auto initial_descriptor = std::array<int, 9>{};
    int info{};
    const int initial_lld = std::max(1, initial_local_rows);
    descinit_(initial_descriptor.data(), &global_num_values_per_dimension,
              &global_num_values_per_dimension, &initial_block_size, &initial_block_size, &zero,
              &zero, &initial_context, &initial_lld, &info);

    // ***************************************
    // CREATING THE FINAL CONTEXT
    // ***************************************
    constexpr int final_num_processor_per_row = 1;
    constexpr int final_num_processor_per_col = 1;
    constexpr int final_block_size = global_num_values_per_dimension;

    int final_context{};
    Cblacs_get(0, 0, &final_context);
    Cblacs_gridinit(&final_context, "Row-major", final_num_processor_per_row,
                    final_num_processor_per_col);

    const int final_local_dim = rank == 0 ? global_num_values_per_dimension : 0;

    auto final_local_data = std::vector(final_local_dim * final_local_dim, 0.0);

    auto final_descriptor = std::array<int, 9>{};
    final_descriptor[1] = -1;// Invalid context marker for non-participants

    if (rank == 0) {
        const int final_lld = std::max(1, final_local_dim);
        descinit_(final_descriptor.data(), &global_num_values_per_dimension,
                  &global_num_values_per_dimension, &final_block_size, &final_block_size, &zero,
                  &zero, &final_context, &final_lld, &info);
    }

    // ***************************************
    // TRANSFORMING: INITIAL->FINAL
    // ***************************************
    // Use initial_context as the union context (all 4 ranks participate)
    Cpdgemr2d(global_num_values_per_dimension, global_num_values_per_dimension,
              initial_local_data.data(), 1, 1, initial_descriptor.data(), final_local_data.data(),
              1, 1, final_descriptor.data(), initial_context);

    if (rank == 0) { std::cout << "\n\n------- After data redistribution ----------\n\n"; }
    print_matrix(std::mdspan{std::as_const(final_local_data).data(),
                             static_cast<std::size_t>(final_local_dim),
                             static_cast<std::size_t>(final_local_dim)},
                 comm);

    // Cleanup
    if (initial_rank_coordinates[0] >= 0) { Cblacs_gridexit(initial_context); }
    if (rank == 0) { Cblacs_gridexit(final_context); }

    MPI_Finalize();
    return 0;
}