cmake_minimum_required (VERSION 3.0)
project (seamless_integration)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         OFF)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           OFF)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"  	  OFF)
option(LIBIGL_WITH_MATLAB            "Use MATLAB"  	  ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use IMGUI"  	  OFF)

find_package(LIBIGL REQUIRED QUIET)
include(Directional)
find_package(SADDLEPOINT QUIET)
find_package(MATLAB REQUIRED)

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
target_include_directories(${PROJECT_NAME}_bin PRIVATE ${MATLAB_ROOT}/extern/include)
target_link_libraries(${PROJECT_NAME}_bin  igl::core igl::matlab ${MATLAB_ROOT}/extern/lib/maci64 ${MATLAB_MX_LIBRARY} ${MATLAB_MEX_LIBRARY} ${MATLAB_MAT_LIBRARY} ${MATLAB_ENG_LIBRARY})