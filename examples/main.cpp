#include <iostream>

#include <costa/grid2grid/transform.hpp>
#include <costa/layout.hpp>

#include <mpi.h>

int main() {

    MPI_Init(nullptr, nullptr);

    std::cout << "Hello World!" << std::endl;

    MPI_Finalize();

    return 0;
}