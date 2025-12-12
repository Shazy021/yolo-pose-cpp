cmake_minimum_required(VERSION 3.16)
project(PoseEstimation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# ONNX Runtime root detection
# -----------------------------
# Priority:
# 1) -DONNXRUNTIME_DIR=...
# 2) env ONNXRUNTIME_DIR
# 3) third_party/onnxruntime (local folder)
if(NOT DEFINED ONNXRUNTIME_DIR)
    if(DEFINED ENV{ONNXRUNTIME_DIR})
        set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
    else()
        set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
    endif()
endif()

message(STATUS "ONNXRUNTIME_DIR = ${ONNXRUNTIME_DIR}")

set(ONNXRT_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRT_LIB_DIR     "${ONNXRUNTIME_DIR}/lib")

# Look for onnxruntime library (name is the same, extension is platform-dependent)
find_library(ONNXRT_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRT_LIB_DIR}" "${ONNXRUNTIME_DIR}/lib64"
    REQUIRED
)

message(STATUS "ONNX Runtime library: ${ONNXRT_LIBRARY}")

# -----------------------------
# OpenCV
# -----------------------------
if(WIN32)
    # Adjust this path if you installed OpenCV differently
    set(OpenCV_DIR "C:/opencv/opencv/build")
endif()

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

# -----------------------------
# Executable target
# -----------------------------
add_executable(PoseEstimation
    src/main.cpp
    src/onnx_engine.cpp
    src/preprocess_letterbox.cpp
    src/yolo_pose_postprocess.cpp
    src/visualize_results.cpp
    src/input_handler.cpp
    src/pipeline.cpp
)

# Include paths
target_include_directories(PoseEstimation
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRT_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(PoseEstimation
    PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRT_LIBRARY}
)

# -----------------------------
# Post-build steps (Windows)
# -----------------------------
if(WIN32)
    # Copy models directory next to the executable
    add_custom_command(TARGET PoseEstimation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/models"
                "$<TARGET_FILE_DIR:PoseEstimation>/models"
    )

    # Copy data directory next to the executable
    add_custom_command(TARGET PoseEstimation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/data"
                "$<TARGET_FILE_DIR:PoseEstimation>/data"
    )

    # Copy onnxruntime.dll next to the executable
    add_custom_command(TARGET PoseEstimation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "${ONNXRT_LIB_DIR}/onnxruntime.dll"
                "$<TARGET_FILE_DIR:PoseEstimation>/onnxruntime.dll"
    )
endif()