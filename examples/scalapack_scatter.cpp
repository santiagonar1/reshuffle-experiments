#include <functional>
#include <iomanip>
#include <iostream>
#include <mdspan>
#include <mpi.h>
#include <ranges>
#include <vector>

// BLACS declarations
extern "C" {
void Cblacs_pinfo(int *mypnum, int *nprocs);
void Cblacs_get(int context, int what, int *val);
void Cblacs_gridinit(int *context, const char *order, int nprow, int npcol);
void Cblacs_gridinfo(int context, int *nprow, int *npcol, int *myrow, int *mycol);
void Cblacs_gridexit(int context);
void Cblacs_exit(int status);

// ScaLAPACK descriptor routines
void descinit_(int *desc, const int *m, const int *n, const int *mb, const int *nb,
               const int *irsrc, const int *icsrc, const int *ictxt, const int *lld, int *info);
int numroc_(const int *n, const int *nb, const int *iproc, const int *isrcproc, const int *nprocs);

// pdgemr2d: redistribute matrix from one layout to another
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

    for (auto i = 0; i < values.extent(0); ++i) {
        for (auto j = 0; j < values.extent(1); ++j) {
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
    constexpr auto global_num_values_per_dimension = 8;

    // ***************************************
    // CREATING THE INITIAL CONTEXT
    // ***************************************
    constexpr auto initial_num_processors_per_row = 1;
    constexpr auto initial_num_processors_per_col = 1;

    constexpr int initial_block_size = global_num_values_per_dimension;

    int initial_context{};
    Cblacs_get(0, 0, &initial_context);
    Cblacs_gridinit(&initial_context, "Row-major", initial_num_processors_per_row,
                    initial_num_processors_per_col);

    const auto initial_local_values_per_dimension = rank == 0 ? global_num_values_per_dimension : 0;

    auto initial_local_data = std::views::iota(0, initial_local_values_per_dimension *
                                                          initial_local_values_per_dimension) |
                              std::ranges::to<std::vector<double>>();

    if (rank == 0) { std::cout << "------- Before data redistribution ----------\n\n"; }
    print_matrix(std::mdspan{std::as_const(initial_local_data).data(),
                             initial_local_values_per_dimension,
                             initial_local_values_per_dimension},
                 comm);

    // Source descriptor (only rank 0 participates)
    auto initial_descriptor = std::array<int, 9>{};
    initial_descriptor[1] = -1;// Invalid context marker

    int info{};

    if (rank == 0) {
        const int initial_lld = initial_local_values_per_dimension;
        descinit_(initial_descriptor.data(), &global_num_values_per_dimension,
                  &global_num_values_per_dimension, &initial_block_size, &initial_block_size, &zero,
                  &zero, &initial_context, &initial_lld, &info);
    }

    // ***************************************
    // CREATING THE FINAL CONTEXT
    // ***************************************
    constexpr auto final_num_processors_per_dimension = 2;

    constexpr int final_block_size =
            global_num_values_per_dimension / final_num_processors_per_dimension;

    int final_context{};
    Cblacs_get(0, 0, &final_context);
    Cblacs_gridinit(&final_context, "Row-major", final_num_processors_per_dimension,
                    final_num_processors_per_dimension);

    // Get grid info for destination
    auto final_rank_coordinates = std::array<int, 2>{};

    // Should be the same as `final_num_processors_per_dimension` unless the rank is not in the grid,
    // in which case both `dummy_final_num_processors_per_row` and `dummy_final_num_processors_per_column`
    // will be -1 (as well as `final_rank_coordinates`)
    int dummy_final_num_processors_per_row{};
    int dummy_final_num_processors_per_column{};

    Cblacs_gridinfo(final_context, &dummy_final_num_processors_per_row,
                    &dummy_final_num_processors_per_column, &final_rank_coordinates[0],
                    &final_rank_coordinates[1]);

    const auto final_local_values_per_row =
            numroc_(&global_num_values_per_dimension, &final_block_size, &final_rank_coordinates[0],
                    &zero, &final_num_processors_per_dimension);
    const auto final_local_values_per_column =
            numroc_(&global_num_values_per_dimension, &final_block_size, &final_rank_coordinates[1],
                    &zero, &final_num_processors_per_dimension);

    auto final_local_data =
            std::vector(final_local_values_per_row * final_local_values_per_column, 0.0);


    // Destination descriptor and local storage
    auto final_descriptor = std::array<int, 9>{};

    const int final_lld = final_local_values_per_column;
    descinit_(final_descriptor.data(), &global_num_values_per_dimension,
              &global_num_values_per_dimension, &final_block_size, &final_block_size, &zero, &zero,
              &final_context, &final_lld, &info);

    // ***************************************
    // TRANSFORMING: INITIAL->FINAL
    // ***************************************
    Cpdgemr2d(global_num_values_per_dimension, global_num_values_per_dimension,
              initial_local_data.data(), 1, 1, initial_descriptor.data(), final_local_data.data(),
              1, 1, final_descriptor.data(), final_context);

    if (rank == 0) { std::cout << "\n\n ------- After data redistribution ----------\n\n"; }
    print_matrix(std::mdspan{std::as_const(final_local_data).data(), final_local_values_per_row,
                             final_local_values_per_column},
                 comm);

    // Cleanup
    if (final_rank_coordinates[1] >= 0) Cblacs_gridexit(final_context);
    if (rank == 0) Cblacs_gridexit(initial_context);

    MPI_Finalize();
    return 0;
}