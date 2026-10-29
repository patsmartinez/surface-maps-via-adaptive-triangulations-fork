cmake_minimum_required(VERSION 3.9)

# CMake 4.x refuses to configure projects declaring cmake_minimum_required < 3.5, and
# several vendored libs (OpenMesh, glfw, ...) still do. This keeps them configurable
# without editing the submodules. Harmless on older CMake.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# ===============================================
# Project name

# Set project name if not build within another project
if("${PROJECT_NAME}" STREQUAL "")
    project(SurfaceMaps)
endif()

# ===============================================
# Build configuration

# The viewer needs OpenGL 4.2-4.5 (glTexStorage, glClipControl, compute shaders),
# but macOS caps desktop OpenGL at 4.1, so it cannot be built there. With this OFF
# we build SurfaceMapsCore + the headless apps only: same algorithm, same metrics,
# CSV/OBJ/PLY output, no window. Linux is unaffected (defaults to ON).
if(APPLE)
    option(ISM_BUILD_VIEWER "Build the glow/GLFW viewer and the apps that use it" OFF)
else()
    option(ISM_BUILD_VIEWER "Build the glow/GLFW viewer and the apps that use it" ON)
endif()
message(STATUS "ISM_BUILD_VIEWER: ${ISM_BUILD_VIEWER}")

# ===============================================
# Global settings

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Add -fPIC flag (required to build predicates.c)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Disable some compiler warnings
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

# Turn some compiler warnings into errors
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Werror=return-type")

# Set glow bin dir
set(GLOW_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin/glow)

# Set output dir for meshes, screenshots, etc...
set(ISM_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/output)

# Disable glfw additionals
option(GLFW_BUILD_EXAMPLES "" OFF)
option(GLFW_BUILD_TESTS "" OFF)
option(GLFW_BUILD_DOCS "" OFF)
option(GLFW_INSTALL "" OFF)

# ===============================================
# Find packages

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Apple clang does not ship the OpenMP runtime, so FindOpenMP cannot detect it on its
# own. Homebrew's libomp supplies it; hand FindOpenMP the flags it needs. No effect on
# Linux, where OpenMP is found normally.
if (APPLE)
    find_path(ISM_LIBOMP_INCLUDE omp.h
        HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
    find_library(ISM_LIBOMP_LIB omp
        HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    if (ISM_LIBOMP_INCLUDE AND ISM_LIBOMP_LIB)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${ISM_LIBOMP_INCLUDE}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${ISM_LIBOMP_INCLUDE}")
        set(OpenMP_C_LIB_NAMES omp)
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY ${ISM_LIBOMP_LIB})
        message(STATUS "Using Homebrew libomp: ${ISM_LIBOMP_LIB}")
    else()
        message(FATAL_ERROR "libomp not found. Install it with: brew install libomp")
    endif()
endif()

find_package(OpenMP REQUIRED)

# ===============================================
# Add submodules

add_subdirectory(libs/OpenMesh)

# Supress an OpenMesh warning
if (${QT_QMAKE_EXECUTABLE})
endif()

# The graphics stack. Skipped entirely when the viewer is off: besides being
# unneeded, glow adds -march=native as a PUBLIC compile option, which does not
# exist on arm64, and clean-core/typed-geometry include x86 intrinsics headers
# unconditionally. Nothing in the headless source set reaches any of them.
if (ISM_BUILD_VIEWER)
    if (NOT TARGET glfw)
        add_subdirectory(libs/glfw)
    endif()

    if (NOT TARGET clean-core)
        add_subdirectory(libs/clean-core)
    endif()

    if (NOT TARGET typed-geometry)
        add_subdirectory(libs/typed-geometry)
    endif()

    if (NOT TARGET polymesh)
        add_subdirectory(libs/polymesh)
    endif()

    if (NOT TARGET glow)
        add_subdirectory(libs/glow)
    endif()

    if (NOT TARGET imgui)
        add_subdirectory(libs/imgui-lean)
    endif()

    if (NOT TARGET glow-extras)
        add_subdirectory(libs/glow-extras)
    endif()
endif()

if (NOT TARGET Eigen3::Eigen)
    add_subdirectory(libs/eigen-lean)
endif()

add_subdirectory(libs/TinyAD)

# ===============================================
# Target SurfaceMaps

file(GLOB_RECURSE ISM_HEADERS src/*.hh libs/ACG_BSP/*.hh libs/predicates/*.h)
file(GLOB_RECURSE ISM_SOURCES src/*.cc libs/ACG_BSP/*.cc libs/predicates/*.c)

# Without the viewer, drop the translation units that need glow/polymesh. These are
# the only ones: everything else in src/ is Eigen + OpenMesh + TinyAD. IO.hh itself
# merely forward-declares glow and polymesh, so it still compiles -- the declarations
# of read_texture/to_polymesh/... stay visible, just undefined, which is fine as long
# as no headless app calls them.
if (NOT ISM_BUILD_VIEWER)
    # Note Colors.hh, HeatmapColors and ColorGenerator are deliberately KEPT: they are
    # pure Vec4f math with no glow in them, and linear_colors() is what feeds
    # write_colored_ply(), which is how a scalar field gets inspected without a window.
    list(FILTER ISM_SOURCES EXCLUDE REGEX
        "src/SurfaceMaps/Viewer/(MeshView|GlowDraw|Picking|LandmarkEditor|ConstantCurvature)\\.cc$")
    list(FILTER ISM_HEADERS EXCLUDE REGEX
        "src/SurfaceMaps/Viewer/(MeshView|GlowDraw|Picking|LandmarkEditor|ConstantCurvature)\\.hh$")
    list(FILTER ISM_SOURCES EXCLUDE REGEX "src/SurfaceMaps/Utils/IO_Graphics\\.cc$")
    list(FILTER ISM_SOURCES EXCLUDE REGEX
        "src/SurfaceMaps/AdaptiveTriangulations/(Visualization|VisualizationDraw|TextureTransfer|WeightPainter)\\.cc$")
    list(FILTER ISM_HEADERS EXCLUDE REGEX
        "src/SurfaceMaps/AdaptiveTriangulations/(Visualization|TextureTransfer)\\.hh$")
endif()

add_library(SurfaceMaps ${ISM_SOURCES} ${ISM_HEADERS})
target_compile_features(SurfaceMaps PUBLIC cxx_std_17)
target_compile_definitions(SurfaceMaps PUBLIC SOURCE_PATH_STR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(SurfaceMaps PUBLIC DATA_PATH_STR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_compile_definitions(SurfaceMaps PUBLIC OUTPUT_PATH_STR="${ISM_OUTPUT_DIR}")
if (ISM_BUILD_VIEWER)
    target_compile_definitions(SurfaceMaps PUBLIC ISM_HAS_VIEWER=1)
else()
    target_compile_definitions(SurfaceMaps PUBLIC ISM_HAS_VIEWER=0)
endif()
target_include_directories(SurfaceMaps PUBLIC
    libs/libigl/include
    ${OPENMESH_INCLUDE_DIRS}
    libs/predicates
    libs/ACG_BSP
    src
)
target_link_libraries(SurfaceMaps PUBLIC
    eigen
    OpenMP::OpenMP_CXX
    OpenMeshCore
    TinyAD
)

if (ISM_BUILD_VIEWER)
    target_link_libraries(SurfaceMaps PUBLIC
        clean-core
        typed-geometry
        polymesh
        glfw
        glow
        imgui
        glow-extras
    )
endif()

# stdc++fs was the separate shim library for <experimental/filesystem>. We now use
# the standard <filesystem>, which libc++ and modern libstdc++ provide directly;
# only older GCC still needs the shim.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(SurfaceMaps PUBLIC stdc++fs)
endif()

# ===============================================
# Target apps (Executables)

file(GLOB_RECURSE ISM_APPS apps/*.cc)

# Apps whose viewer code is guarded by #if ISM_HAS_VIEWER and which therefore build
# both ways. Every other app opens a window unconditionally, so it is skipped when
# the viewer is off. Add a name here once its viewer usage is guarded.
set(ISM_HEADLESS_CAPABLE_APPS
    baseline_comparison
    mobius_probe
    lookup_distortion_probe
)

if (NOT ISM_BUILD_VIEWER)
    set(ISM_APPS_KEPT "")
    foreach(main ${ISM_APPS})
        get_filename_component(app_name ${main} NAME_WE)
        if (app_name IN_LIST ISM_HEADLESS_CAPABLE_APPS)
            list(APPEND ISM_APPS_KEPT ${main})
        endif()
    endforeach()
    set(ISM_APPS ${ISM_APPS_KEPT})
endif()

foreach(main ${ISM_APPS})
    get_filename_component(name ${main} NAME_WE)
    get_filename_component(dir ${main} DIRECTORY)
    get_filename_component(dir_name ${dir} NAME)
    message(STATUS "Adding app ${name}")

    add_executable(${name} ${main})
    target_compile_definitions(${name} PRIVATE APP_EXECUTABLES)
    add_dependencies(${name} SurfaceMaps)
    target_include_directories(${name} PUBLIC
        "src"
    )
    target_link_libraries(${name} PUBLIC
        SurfaceMaps
    )
endforeach()
