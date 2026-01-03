cmake_minimum_required(VERSION 3.16)

project(image_converter VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME}
	src/main.cpp

	src/matrix.h
	src/svd.h
	src/bidiagonal.h
)

if(MSVC)
	target_compile_options(${PROJECT_NAME}
	PRIVATE /W4 /permissive-)
else()
	target_compile_options(${PROJECT_NAME}
	PRIVATE -Wall -Wextra -Wpedantic)
endif()