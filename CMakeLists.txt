cmake_minimum_required(VERSION 3.30)
set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
project(orb_template)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV using vcpkg
find_package(OpenCV REQUIRED)

# Add OpenCV include directories to the project
include_directories(${OpenCV_INCLUDE_DIRS})

# Add OpenCV libraries to the project
link_directories(${OpenCV_LIBRARY_DIRS})

# Define the executable
add_executable(${PROJECT_NAME} orb_template.cpp)

# Link OpenCV libraries to the executable
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Install the executable
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# Install the OpenCV libraries
install(DIRECTORY ${OpenCV_LIBRARY_DIRS} DESTINATION lib)