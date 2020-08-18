cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
project(ObjectTracking VERSION 1.0)

add_executable(${PROJECT_NAME} ObjectTracking.cpp)
configure_file(ObjectTrackingConfig.h.in ObjectTrackingConfig.h)

target_include_directories(ObjectTracking PUBLIC
			   ${PROJECT_BINARY_DIR}
			  )
