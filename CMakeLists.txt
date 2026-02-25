cmake_minimum_required(VERSION 4.1)
project(cource)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_DIR "D:/opencv/mybuild/install/x64/mingw/lib")
find_package(OpenCV REQUIRED)

add_library(cource SHARED library.cpp)

target_include_directories(cource PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cource PRIVATE ${OpenCV_LIBS})