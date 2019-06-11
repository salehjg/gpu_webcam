cmake_minimum_required(VERSION 3.12)
project(gpu_webcam_repo)
set(CMAKE_CXX_STANDARD 11)

set(REPO_DIR_CMAKE  ${PROJECT_SOURCE_DIR})
set(BIN_DIR_CMAKE   ${CMAKE_BINARY_DIR})


SET(CUDA_INCLUDE_DIRS /usr/include)
SET(CUDA_LIBRARIES cudart cublas)
SET(CUDA_LIBRARY_DIRS /usr/lib/X86_64-linux-gnu)

find_package(CUDA REQUIRED)

SET(CUDA_HOST_COMPILER /usr/bin/gcc-6)
set(CUDA_PROPAGATE_HOST_FLAGS False)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options '-fPIC'")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

set(SOURCE_FILES_CUDA
        main.cpp
        )

# Find all cuda kernels
#(https://stackoverflow.com/questions/29308787/cmake-file-for-cpp-cu-files)
file( GLOB  KERNEL_SOURCES  ${PROJECT_SOURCE_DIR}/cuda-kernels/*.cu)
message("Found these kernels: ")
message("${KERNEL_SOURCES}")

find_package(OpenCV REQUIRED)
cuda_add_executable(gpu_webcam_repo
        ${KERNEL_SOURCES}
        ${SOURCE_FILES_CUDA}
        )
target_link_libraries(gpu_webcam_repo ${OpenCV_LIBS} )