include(FindPackageHandleStandardArgs)

# Try pkg-config first for ScaLAPACK
find_package(PkgConfig QUIET)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(SCALAPACK QUIET scalapack)
endif ()

# If pkg-config didn't work, try direct path
if (NOT SCALAPACK_FOUND)
    find_library(SCALAPACK_LIBRARIES
            NAMES scalapack scalapack-openmpi
            HINTS
            ENV SCALAPACK_ROOT
            ENV SCALAPACK_PREFIX
            ENV SCALAPACK_DIR
            PATH_SUFFIXES lib lib64
    )
endif ()

find_package_handle_standard_args(SCALAPACK
        REQUIRED_VARS SCALAPACK_LIBRARIES
)

# Create imported target for modern CMake usage
if (SCALAPACK_FOUND AND NOT TARGET SCALAPACK::SCALAPACK)
    add_library(SCALAPACK::SCALAPACK UNKNOWN IMPORTED)
    set_target_properties(SCALAPACK::SCALAPACK PROPERTIES
            IMPORTED_LOCATION "${SCALAPACK_LIBRARIES}"
    )
    # If pkg-config found include dirs, add them
    if (SCALAPACK_INCLUDE_DIRS)
        set_target_properties(SCALAPACK::SCALAPACK PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${SCALAPACK_INCLUDE_DIRS}"
        )
    endif ()
endif ()