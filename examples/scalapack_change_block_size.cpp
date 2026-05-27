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

    for (auto i = 0; i < values.extent(0); ++i) {
        for (auto j = 0; j < values.extent(1); ++j) {
            // We are writing the matrix in column-major order for readability
            std::cout << std::setw(4) << values[j, i] << " ";
        }
        std::cout << std::endl;
    }
}

template<typename Extents>
auto print_matrix(std::mdspan<const double, Extents> values, MPI_Comm comm) -> void {
    int rank{};
    MPI_Comm_rank(comm, &rank);
    execute_sequentially([values, rank]() { print_matrix(values, rank); }, comm);
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
        std::cerr << "ERROR: This demo requires 4 MPI ranks" << std::endl;
        MPI_Abort(comm, 1);
    }

    // ***************************************
    // GLOBAL MATRIX: 8x8
    // ***************************************
    constexpr auto global_rows = 8;
    constexpr auto global_cols = 8;

    // ***************************************
    // INITIAL CONTEXT: 2x2 grid, block size 2
    // ***************************************
    constexpr auto initial_num_proc_rows = 2;
    constexpr auto initial_num_proc_cols = 2;
    constexpr auto initial_block_size = 2;

    int initial_context{};
    Cblacs_get(0, 0, &initial_context);
    Cblacs_gridinit(&initial_context, "Row-major", initial_num_proc_rows, initial_num_proc_cols);

    auto initial_rank_coords = std::array<int, 2>{};
    int dummy_nprow{}, dummy_npcol{};
    Cblacs_gridinfo(initial_context, &dummy_nprow, &dummy_npcol, &initial_rank_coords[0],
                    &initial_rank_coords[1]);

    const auto initial_local_rows = numroc_(&global_rows, &initial_block_size,
                                            &initial_rank_coords[0], &zero, &initial_num_proc_rows);
    const auto initial_local_cols = numroc_(&global_cols, &initial_block_size,
                                            &initial_rank_coords[1], &zero, &initial_num_proc_cols);

    // Column-major storage for ScaLAPACK
    auto initial_local_data = std::vector<double>(initial_local_rows * initial_local_cols);

    // Fill with global indices (column-major storage)
    for (int local_i = 0; local_i < initial_local_rows; ++local_i) {
        for (int local_j = 0; local_j < initial_local_cols; ++local_j) {
            const int index_local_block_row = local_i / initial_block_size;
            const int index_local_block_col = local_j / initial_block_size;
            const int global_i =
                    index_local_block_row * initial_num_proc_rows * initial_block_size +
                    initial_rank_coords[0] * initial_block_size + (local_i % initial_block_size);
            const int global_j =
                    index_local_block_col * initial_num_proc_cols * initial_block_size +
                    initial_rank_coords[1] * initial_block_size + (local_j % initial_block_size);
            // Column-major: index = row + col * num_rows
            initial_local_data[local_i + local_j * initial_local_rows] =
                    global_i * global_cols + global_j;
        }
    }

    if (rank == 0) {
        std::cout << "------- Before redistribution (block_size=2, 2x2 grid) ----------\n\n";
    }
    print_matrix(std::mdspan{std::as_const(initial_local_data).data(), initial_local_rows,
                             initial_local_cols},
                 comm);

    auto initial_descriptor = std::array<int, 9>{};
    int info{};
    // LLD = number of local rows (column-major leading dimension)
    const int initial_lld = std::max(1, initial_local_rows);
    descinit_(initial_descriptor.data(), &global_rows, &global_cols, &initial_block_size,
              &initial_block_size, &zero, &zero, &initial_context, &initial_lld, &info);

    // ***************************************
    // FINAL CONTEXT: 2x2 grid, block size 4
    // ***************************************
    constexpr auto final_num_proc_rows = 2;
    constexpr auto final_num_proc_cols = 2;
    constexpr auto final_block_size = 4;

    int final_context{};
    Cblacs_get(0, 0, &final_context);
    Cblacs_gridinit(&final_context, "Row-major", final_num_proc_rows, final_num_proc_cols);

    auto final_rank_coords = std::array<int, 2>{};
    Cblacs_gridinfo(final_context, &dummy_nprow, &dummy_npcol, &final_rank_coords[0],
                    &final_rank_coords[1]);

    const auto final_local_rows = numroc_(&global_rows, &final_block_size, &final_rank_coords[0],
                                          &zero, &final_num_proc_rows);
    const auto final_local_cols = numroc_(&global_cols, &final_block_size, &final_rank_coords[1],
                                          &zero, &final_num_proc_cols);

    auto final_local_data = std::vector<double>(final_local_rows * final_local_cols, 0.0);

    auto final_descriptor = std::array<int, 9>{};
    // LLD = number of local rows
    const int final_lld = std::max(1, final_local_rows);
    descinit_(final_descriptor.data(), &global_rows, &global_cols, &final_block_size,
              &final_block_size, &zero, &zero, &final_context, &final_lld, &info);

    // ***************************************
    // REDISTRIBUTE: block_size 2 -> block_size 4
    // ***************************************
    Cpdgemr2d(global_rows, global_cols, initial_local_data.data(), 1, 1, initial_descriptor.data(),
              final_local_data.data(), 1, 1, final_descriptor.data(), final_context);

    if (rank == 0) {
        std::cout << "\n\n------- After redistribution (block_size=4, 2x2 grid) ----------\n\n";
    }
    print_matrix(
            std::mdspan{std::as_const(final_local_data).data(), final_local_rows, final_local_cols},
            comm);

    // Cleanup
    Cblacs_gridexit(final_context);
    Cblacs_gridexit(initial_context);

    MPI_Finalize();
    return 0;
}