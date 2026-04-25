include(FetchContent)

if (COSTA_FOUND)
    message(STATUS "COSTA library found")
    return()
endif ()

message(STATUS "COSTA library NOT found, will download and build")

FetchContent_Declare(
        costa
        URL https://github.com/eth-cscs/COSTA/archive/refs/heads/master.zip
)

set(COSTA_SCALAPACK "CUSTOM" CACHE STRING "ScaLAPACK backend for COSTA (MKL, CRAY_LIBSCI, CUSTOM)")

FetchContent_MakeAvailable(costa)
