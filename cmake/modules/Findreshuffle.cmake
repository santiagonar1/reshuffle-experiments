include(FetchContent)

if (RESHUFFLE_FOUND)
    message(STATUS "Reshuffle library found")
    return()
endif ()

message(STATUS "Reshuffle library NOT found, will download and build")

# TODO: Change this for a specific release
FetchContent_Declare(
        reshuffle
        URL https://github.com/santiagonar1/reshuffle/archive/refs/heads/main.zip
)

set(RESHUFFLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RESHUFFLE_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(RESHUFFLE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(reshuffle)