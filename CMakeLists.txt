cmake_minimum_required(VERSION 2.8.3)
#project(VehicleDetector)

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(tree_detection src/main.cpp )

target_link_libraries(tree_detection ${OpenCV_LIBS})
