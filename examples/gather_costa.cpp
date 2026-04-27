#include <functional>
#include <iostream>
#include <mdspan>

#include <costa/grid2grid/transform.hpp>
#include <costa/layout.hpp>

#include <mpi.h>

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
        for (auto j = 0; j < values.extent(1); ++j) { std::cout << values[i, j] << " "; }
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

int main() {

    MPI_Init(nullptr, nullptr);

    const auto comm = MPI_COMM_WORLD;

    int num_procs{};
    int rank{};

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &rank);

    if (num_procs != 4) {
        std::cerr << "ERROR: Expected 4 processes, got " << num_procs << std::endl;
        MPI_Abort(comm, 1);
    }

    // ***************************************
    // DESCRIBING THE GLOBAL MATRIX GRID
    // ***************************************
    constexpr auto block_size = 2;

    constexpr auto global_num_rows = 4;

    constexpr auto global_num_cols = 4;

    constexpr auto submatrix_start = std::array{1, 1};// (1-based, required by costa)
    constexpr auto coordinates_initial_rank = std::array{0, 0};

    // ***************************************
    // CREATING THE INITIAL LAYOUT OBJECT
    // ***************************************
    constexpr auto initial_num_processors_per_row = 2;
    constexpr auto initial_num_processors_per_col = 2;

    constexpr auto initial_local_values_per_row = global_num_rows / initial_num_processors_per_row;
    constexpr auto initial_local_values_per_col = global_num_cols / initial_num_processors_per_col;

    auto initial_local_data =
            std::vector(initial_local_values_per_row * initial_local_values_per_col, 0.0);

    constexpr auto initial_processor_grid_ordering = 'R';// row-major ordering

    constexpr auto initial_local_data_ordering = 'R';// column-major ordering


    auto initial_layout = costa::block_cyclic_layout(
            global_num_rows, global_num_cols, block_size, block_size, submatrix_start[0],
            submatrix_start[1], global_num_rows, global_num_cols, initial_num_processors_per_row,
            initial_num_processors_per_col, initial_processor_grid_ordering,
            coordinates_initial_rank[0], coordinates_initial_rank[1], &initial_local_data[0],
            block_size, initial_local_data_ordering, rank);

    // ***************************************
    // INITIALIZE THE INITIAL MATRIX
    // ***************************************
    auto init_function = [](const int i, const int j) -> double { return i * global_num_cols + j; };
    initial_layout.initialize(init_function);

    print_matrix(std::mdspan{std::as_const(initial_local_data).data(), initial_local_values_per_row,
                             initial_local_values_per_col},
                 comm);


    // ***************************************
    // CREATING THE FINAL LAYOUT OBJECT
    // ***************************************
    constexpr auto final_num_processors_per_row = 1;
    constexpr auto final_num_processors_per_col = 1;

    const auto final_local_values_per_row =
            rank == 0 ? global_num_rows / final_num_processors_per_row : 0;
    const auto final_local_values_per_col =
            rank == 0 ? global_num_cols / final_num_processors_per_col : 0;

    auto final_local_data =
            std::vector(final_local_values_per_row * final_local_values_per_col, 0.0);


    constexpr auto final_processor_grid_ordering = 'R';

    constexpr auto final_local_data_ordering = 'R';


    // Note: the block size (and lld) here is the same as the global matrix dimensions. I tried to
    // use the same one as in the initial layout, but it resulted in incorrect results. This is probably
    // a bug in COSTA.
    auto final_layout = costa::block_cyclic_layout(
            global_num_rows, global_num_cols, global_num_rows, global_num_cols, submatrix_start[0],
            submatrix_start[1], global_num_rows, global_num_cols, final_num_processors_per_row,
            final_num_processors_per_col, final_processor_grid_ordering,
            coordinates_initial_rank[0], coordinates_initial_rank[1], &final_local_data[0],
            global_num_cols, final_local_data_ordering, rank);

    // ***************************************
    // TRANSFORMING: INITIAL->FINAL
    // ***************************************
    costa::transform<double>(initial_layout, final_layout, comm);

    print_matrix(std::mdspan{std::as_const(final_local_data).data(), final_local_values_per_row,
                             final_local_values_per_col},
                 comm);

    if (const auto ok = final_layout.validate(init_function, 1e-12); not ok) {
        std::cerr << "[ERROR] Result incorrect on rank " << rank << std::endl;
        MPI_Abort(comm, 1);
    }

    MPI_Barrier(comm);

    // if MPI was not aborted, results are correct
    if (rank == 0) { std::cout << "[PASSED] Results are correct!" << std::endl; }

    MPI_Finalize();

    return 0;
}