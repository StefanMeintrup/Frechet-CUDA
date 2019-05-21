cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(frechet_cuda LANGUAGES CUDA CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive") #supress error in older gcc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_50,code=sm_50;-rdc=true;-use_fast_math")

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Include header files from directory 'include'
include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost 1.63 COMPONENTS system chrono program_options python3 numpy3 REQUIRED)
find_package(CUDA  REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${install_dir}/include)
link_libraries(${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${DLIB_LIBRARIES} ${CUDA_LIBRARIES})
add_definitions(-D_GLIBCXX_PARALLEL)

PYTHON_ADD_MODULE(frechet_cuda
			src/compute_distance_wrapper.cpp 
			src/curve.cpp 
			src/geometry_basics.cpp
			src/frechet.cpp
			src/intersection_algorithm_in_parallel.cu
)
