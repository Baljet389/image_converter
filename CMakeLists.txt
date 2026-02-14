cmake_minimum_required(VERSION 3.16)

project(image_converter VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)

add_executable(${PROJECT_NAME}
	src/main.cpp
	src/stb_impl.cpp

	src/matrix.h
	src/svd.h
	src/bidiagonal.h
	src/testSVD.h
	src/qr.h
	src/image.h
	src/utility.h)

target_include_directories(${PROJECT_NAME} SYSTEM BEFORE PRIVATE lib)

if(MSVC)
	target_compile_options(${PROJECT_NAME}
	PRIVATE /W4 
			/permissive- 
			"/FI${CMAKE_CURRENT_SOURCE_DIR}/src/macros.h"
			$<$<CONFIG:Release>:/O2 
			/arch:AVX2 
			/fp:fast>
			)
else()
	target_compile_options(${PROJECT_NAME}
	PRIVATE -Wall         
			-Wextra        
			-Wpedantic
			-march=native 
			-ffast-math 
			-O3 
			#-fsanitize=address -g
			#-Rpass-analysis=loop-vectorize
			-include "${CMAKE_CURRENT_SOURCE_DIR}/src/macros.h" 
	)
	if(OpenMP_CXX_FOUND)
		target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
	endif()
	# target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
endif()
 
