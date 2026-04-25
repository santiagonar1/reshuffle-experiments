add_executable(app.out main.cpp)
target_compile_features(app.out PUBLIC cxx_std_23)

target_link_libraries(app.out
        PRIVATE MPI::MPI_CXX
        PRIVATE costa::costa
)
