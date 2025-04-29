cmake_minimum_required(VERSION 3.10)
project(RealPoseSense)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 프로젝트 소스 디렉토리 지정
include_directories(${CMAKE_SOURCE_DIR})

# OpenCV 패키지 찾기
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# RealSense 패키지 찾기
find_package(realsense2 REQUIRED)
include_directories(${realsense2_INCLUDE_DIRS})

# CUDA 패키지 찾기
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# CUDA 라이브러리 직접 찾기
find_library(CUDART_LIBRARY cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/local/cuda/lib64)
if(NOT CUDART_LIBRARY)
    message(STATUS "cudart library not found in standard locations, trying additional paths")
    find_library(CUDART_LIBRARY cudart HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda-*/lib64)
    if(NOT CUDART_LIBRARY)
        message(WARNING "cudart library still not found, will try with -lcudart directly")
        set(CUDART_LIBRARY cudart)
    endif()
endif()
message(STATUS "CUDART_LIBRARY: ${CUDART_LIBRARY}")

# TensorRT 경로 설정
set(TENSORRT_ROOT /usr/local/TensorRT)
if(NOT EXISTS ${TENSORRT_ROOT})
    set(TENSORRT_ROOT /usr/lib/aarch64-linux-gnu)
endif()
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

# TensorRT 라이브러리 직접 찾기
find_library(NVINFER_LIBRARY nvinfer HINTS ${TENSORRT_ROOT}/lib /usr/lib/aarch64-linux-gnu)
if(NOT NVINFER_LIBRARY)
    message(WARNING "nvinfer library not found, will try with -lnvinfer directly")
    set(NVINFER_LIBRARY nvinfer)
endif()
message(STATUS "NVINFER_LIBRARY: ${NVINFER_LIBRARY}")

# 소스 파일 추가
set(SOURCES
    main.cpp
    ConfigManager.cpp
    DepthProcessor.cpp
    RealSenseCamera.cpp
    utils/FileUtils.cpp
    utils/FPSCounter.cpp
    utils/KeyboardHandler.cpp
    utils/ImageSaver.cpp
    PoseEstimator.cpp
)

# 실행 파일 생성
add_executable(${PROJECT_NAME} ${SOURCES})

# 라이브러리 링크
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
    ${CUDA_LIBRARIES}
    ${NVINFER_LIBRARY}
    ${CUDART_LIBRARY}
)

# ONNX Runtime 헤더 경로 포함
target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})

# ONNX Runtime 라이브러리 링크
target_link_libraries(${PROJECT_NAME} PRIVATE ${onnxruntime_LIBRARIES}) 