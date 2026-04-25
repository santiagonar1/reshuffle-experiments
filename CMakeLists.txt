cmake_minimum_required(VERSION 3.28)
project(reshuffle_experiments)

# Set compiler to use Xcode SDK
if (APPLE)
    execute_process(
            COMMAND xcrun --show-sdk-path
            OUTPUT_VARIABLE SDKROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_OSX_SYSROOT "${SDKROOT}" CACHE PATH "macOS SDK path" FORCE)
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Where to find our CMake modules. Needs to be called before any add_subdirectory for them to have access to the variable
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(MPI REQUIRED)
find_package(costa REQUIRED)

add_subdirectory(examples)


