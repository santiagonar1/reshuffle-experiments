include(FetchContent)

if (COSTA_FOUND)
    message(STATUS "COSTA library found")
    return()
endif ()

message(STATUS "COSTA library NOT found, will download and build")

FetchContent_Declare(
        costa
        GIT_REPOSITORY https://github.com/eth-cscs/COSTA.git
        GIT_TAG v2.3.2
)

set(COSTA_SCALAPACK "OFF" CACHE STRING "ScaLAPACK backend for COSTA (MKL, CRAY_LIBSCI, CUSTOM)")

FetchContent_MakeAvailable(costa)
