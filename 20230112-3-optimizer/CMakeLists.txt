cmake_minimum_required(VERSION 3.24)
project(simple_test)

#find_package(OpenMP REQUIRED)
#if(OpenMP_FOUND)
#    message(STATUS "found openmp")
#    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS}, ${OPENMP_C_FLAGS})
#    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}, ${OPENMP_CXX_FLAGS})
#else()
#    message(FATAL_ERROR "openmp not found!")
#endif()
include_directories(/opt/homebrew/include)
#link_libraries(/opt/homebrew/lib)
link_directories(/opt/homebrew/lib)
include_directories(include)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 23)
add_executable(simple_test main.cpp include/TimerClock.hpp include/resources/ThreadPool.hpp
        dataset.hpp include/compute/LineaerRegression.hpp include/resources/MemoryPool.hpp include/MyException.hpp include/compute/Matrix.hpp include/DEFINES.h include/SegmentVectorWise.hpp include/DynamicElementWiseSegment.h)




