cmake_minimum_required(VERSION 3.5)
project(retarget)

# Setting up
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdeprecated-declarations -fpermissive -std=c++11")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


# TODO Add support for macosx
# TODO Switch x86/x64
if (WIN32)
    add_definitions(-DWIN_ENV)
else(WIN32)
    add_definitions(-DUNIX_ENV)
endif(WIN32)
add_definitions(-DUNICODE -DWIN32=1 -D_WINDOWS=1 -DWIN_ENV=1 -DHAVE_EXPAT_CONFIG_H=1 -DXML_STATIC=1 -DXMP_StaticBuild=1 -DXMP_WinBuild=1)

# Gathering files
set(LIB_SOURCE_FILES "retarget-lib/src/retarget.cpp"
        "retarget-lib/src/cvxgen_solver25x25/solver_25x25.cpp"
        "retarget-lib/src/cvxgen_solver25x25/ldl_25x25.cpp"
        "retarget-lib/src/cvxgen_solver25x25/matrix_support_25x25.cpp")
set(EXE_SOURCE_FILES retarget-make/retarget-make.cpp)
file(GLOB_RECURSE XMP_SRCS external/xmp-toolkit/src *.cpp)
file(GLOB_RECURSE XMP_EXPAT_SRCS external/xmp-toolkit/third-party/expat *.c)
file(GLOB_RECURSE XMP_ZLIB_SRCS external/xmp-toolkit/third-party/zlib *.c)
file(GLOB_RECURSE XMP_MD5_SRCS external/xmp-toolkit/third-party/MD5 *.cpp)

set(XMP_LIB_SOURCE_FILES ${XMP_SRCS} ${XMP_EXPAT_SRCS} ${XMP_ZLIB_SRCS} ${XMP_MD5_SRCS}) # Merging

# Creating projects
add_library(retarget-lib STATIC ${LIB_SOURCE_FILES})
add_library(xmp-lib STATIC ${XMP_LIB_SOURCE_FILES})
add_executable(retarget-make ${EXE_SOURCE_FILES})

# Include directories
include_directories(retarget-lib/include)
target_include_directories(retarget-make PUBLIC retarget-make)
target_include_directories(xmp-lib PUBLIC external/xmp-toolkit/include)
target_include_directories(xmp-lib PUBLIC external/xmp-toolkit/src/XMPFiles)
target_include_directories(xmp-lib PUBLIC external/xmp-toolkit/src/XMPFiles/FileHandlers)
target_include_directories(xmp-lib PUBLIC external/xmp-toolkit/src/XMPFiles/FormatSupport)
target_include_directories(xmp-lib PUBLIC external/xmp-toolkit/src/XMPCore)
target_include_directories(xmp-lib PUBLIC external/xmp-toolkit/src/common)
target_include_directories(xmp-lib PUBLIC external/xmp-toolkit/third-party/expat)
target_include_directories(xmp-lib PUBLIC external/xmp-toolkit/third-party/MD5)
target_include_directories(xmp-lib PUBLIC external/xmp-toolkit/third-party/zlib)

# Linking
target_link_libraries(retarget-lib xmp-lib)
target_link_libraries(retarget-make retarget-lib)
