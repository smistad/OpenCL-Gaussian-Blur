cmake_minimum_required(VERSION 2.8)
project( OCL-Gaussian-Blur )

add_subdirectory(SIPL)
include_directories(${SIPL_INCLUDE_DIRS})
add_subdirectory(OpenCLUtilities)
include_directories(${OPENCL_INCLUDE_DIRS})
add_executable(blur main.cpp)
target_link_libraries(blur OCL-Utilities SIPL)
