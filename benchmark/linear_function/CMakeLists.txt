cmake_minimum_required (VERSION 3.0)
project (linear_function)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(SADDLEPOINT QUIET)

if (NOT SADDLEPOINT_FOUND)
   message(FATAL_ERROR "SaddlePoint not found --- You can download it in https://github.com/avaxman/SaddlePoint.git")
endif()

message("SaddlePoint includes: ${SADDLEPOINT_INCLUDE_DIRS}")

# Prepare the build environment

include_directories(${SADDLEPOINT_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add your project files
FILE(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME}_bin ${SRCFILES})
target_link_libraries(${PROJECT_NAME}_bin Eigen3::Eigen)