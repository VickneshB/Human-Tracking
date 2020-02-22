cmake_minimum_required(VERSION 3.15)
project(Test)

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_FOUND 1)
set(OpenCV_DIR "C:/opencv/build")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

add_executable(Test main.cpp)
target_link_libraries(Test ${OpenCV_LIBS})