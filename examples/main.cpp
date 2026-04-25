#include <iostream>

#include <mpi.h>

int main() {

    MPI_Init(nullptr, nullptr);

    std::cout << "Hello World!" << std::endl;

    MPI_Finalize();

    return 0;
}