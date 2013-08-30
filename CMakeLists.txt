cmake_minimum_required(VERSION 2.4.6)
#include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
FIND_PACKAGE(CUDA)
#FIND_PACKAGE( OpenMP )

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

SET(PROJECT_NAME superresolution)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)


include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("$ENV{CUDADIR}/include")
include_directories("$ENV{CUDASDKDIR}/common/inc")
include_directories("/usr/include/nvidia-current/cuda")
include_directories("/usr/include/opencv-2.3.1")
link_directories("$ENV{CUDASDKDIR}/lib")
link_directories("$ENV{CUDASDKDIR}/../shared/lib")
link_directories("$ENV{CUDADRVDIR}")
link_directories("/usr/lib/nvidia-current")

include_directories("/work/sdks/opencv-2.4.2/include")
link_directories("/work/sdks/opencv-2.4.2/lib")

#ADD_DEFINITIONS(-fopenmp)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -O0 -g")
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Wall -O0 -g")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -O2")
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Wall -O2")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -O3")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Wall -O3")

set(CPU_LIBRARY_TYPE SHARED)
set(GPU_LIBRARY_TYPE SHARED)

set(CUDA_ARCH_FLAGS 
-arch sm_11
# -D$ENV{CUDA_CAPABILITY} -arch $ENV{CUDA_ARCH}
)

set(CUDAOPTIONS 
--ptxas-options=-v -DUNIX 
${CUDA_ARCH_FLAGS}
)




add_library(linearoperations ${CPU_LIBRARY_TYPE}
src/linearoperations/linearoperations.cpp
)
cuda_add_library(linearoperationsGPU ${GPU_LIBRARY_TYPE}
src/linearoperations/linearoperations.cu
OPTIONS ${CUDAOPTIONS}
)

add_library(imagepyramid ${CPU_LIBRARY_TYPE}
src/imagepyramid/imagepyramidcpu.cpp
)
cuda_add_library(imagepyramidGPU ${GPU_LIBRARY_TYPE}
src/imagepyramid/imagepyramidgpu.cpp
OPTIONS ${CUDAOPTIONS}
)

add_library(flowlib ${CPU_LIBRARY_TYPE}
src/flowlib/flowio.cpp
src/flowlib/flowlib.cpp
src/flowlib/flowlib_cpu_sor.cpp
)
cuda_add_library(flowlibGPU ${GPU_LIBRARY_TYPE}
src/flowlib/flowlib_gpu_sor.cu
src/flowlib/flowlib.cu
OPTIONS ${CUDAOPTIONS}
)

add_library(filesystem ${CPU_LIBRARY_TYPE}
src/filesystem/filesystem.cpp
)

add_library(auxiliaryCPU ${CPU_LIBRARY_TYPE}
src/auxiliary/debug.cpp
)
cuda_add_library(auxiliaryGPU ${GPU_LIBRARY_TYPE}
src/auxiliary/cuda_basic.cu
OPTIONS ${CUDAOPTIONS}
)

add_library(superresolutionlib ${CPU_LIBRARY_TYPE}
src/superresolution/superresolution.cpp
)
cuda_add_library(superresolutionlibGPU ${GPU_LIBRARY_TYPE}
src/superresolution/superresolution.cu
OPTIONS ${CUDAOPTIONS}
)


add_executable(superresolution
src/superresolution_main.cpp
)

target_link_libraries(superresolution
superresolutionlib superresolutionlibGPU
flowlib flowlibGPU
imagepyramid imagepyramidGPU
linearoperations linearoperationsGPU
filesystem
auxiliaryCPU auxiliaryGPU
opencv_core opencv_highgui opencv_imgproc cuda cudart
)

add_executable(flow
src/flow_main.cpp
)
target_link_libraries(flow
flowlib flowlibGPU
imagepyramid imagepyramidGPU
linearoperations linearoperationsGPU
auxiliaryCPU auxiliaryGPU
opencv_core opencv_highgui opencv_imgproc cuda cudart
)


