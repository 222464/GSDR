cmake_minimum_required(VERSION 2.8)

project(GSDR)
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")

file(GLOB_RECURSE SOURCES
    "source/*.h"
    "source/*.cpp"
)

add_executable(GSDR ${SOURCES})
include_directories("${PROJECT_SOURCE_DIR}/source")

find_package(SFML 2 REQUIRED system window graphics)
include_directories(${SFML_INCLUDE_DIR})
target_link_libraries(GSDR ${SFML_LIBRARIES})