cmake_minimum_required(VERSION 3.20)
project(MyCudaProject LANGUAGES CXX CUDA)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")


find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)


add_library(cuda_wrapper SHARED library.cu)


set_target_properties(cuda_wrapper PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON

        CUDA_ARCHITECTURES "75;80;86;86-virtual"
)


target_include_directories(cuda_wrapper PRIVATE ${OpenCV_INCLUDE_DIRS})


target_link_libraries(cuda_wrapper PRIVATE
        ${OpenCV_LIBS}
        CUDA::cudart
)