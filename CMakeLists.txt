cmake_minimum_required(VERSION 2.8.11)

project(DualfishMC)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++11 -pedantic -Wall -Wextra -Wconversion -lm")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

find_package(OpenCV 3.1.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SRCS src/*.cpp)
add_executable(fisheye-mc ${SRCS})
target_link_libraries(fisheye-mc ${OpenCV_LIBS})
