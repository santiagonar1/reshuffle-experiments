include(FindPackageHandleStandardArgs)

# Try pkg-config first for ScaLAPACK
find_package(PkgConfig QUIET)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(SCALAPACK QUIET scalapack)
endif ()

# If pkg-config didn't work, try direct path
if (NOT SCALAPACK_FOUND)
    find_library(SCALAPACK_LIBRARIES
            NAMES scalapack scalapack-openmpi scalapack-mpich
            HINTS
            ENV SCALAPACK_ROOT
            ENV SCALAPACK_PREFIX
            ENV SCALAPACK_DIR
            PATH_SUFFIXES lib lib64
    )
endif ()

# Some distros separate BLACS from ScaLAPACK, so we make this logic explicit if needed
find_library(BLACS_LIBRARIES
        NAMES blacs-openmpi blacs-mpi blacs
        HINTS
        ENV BLACS_ROOT
        ENV BLACS_DIR
        PATH_SUFFIXES lib lib64
)

if (BLACS_LIBRARIES)
    list(APPEND SCALAPACK_LIBRARIES ${BLACS_LIBRARIES})
endif ()

find_package_handle_standard_args(SCALAPACK
        REQUIRED_VARS SCALAPACK_LIBRARIES
        FOUND_VAR SCALAPACK_FOUND
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

    if (BLACS_LIBRARIES)
        set_target_properties(SCALAPACK::SCALAPACK PROPERTIES
                INTERFACE_LINK_LIBRARIES "${BLACS_LIBRARIES}"
        )
    endif ()
endif ()

if (SCALAPACK_FOUND)
    message(STATUS "Found ScaLAPACK: ${SCALAPACK_LIBRARIES}")
else ()
    message(WARNING "ScaLAPACK not found. Please install libscalapack-mpi-dev "
            "or set SCALAPACK_ROOT to the installation prefix.")
endif ()
