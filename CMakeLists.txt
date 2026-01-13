cmake_minimum_required(VERSION 3.16)
project(PoseEstimation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ==============================================================================
# Dependencies
# ==============================================================================

# OpenCV: Required for image processing and visualization
# Path is provided via CMakePresets.json (OpenCV_DIR variable)
find_package(OpenCV REQUIRED)

# ONNX Runtime: Required for model inference
# ONNXRUNTIME_ROOT is provided via CMakePresets.json
find_library(ONNXRT_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib"
    REQUIRED
)

set(ONNXRT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

# Log dependency paths for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV directory: ${OpenCV_DIR}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "ONNX Runtime library: ${ONNXRT_LIBRARY}")

# ==============================================================================
# Executable Target
# ==============================================================================
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
target_include_directories(PoseEstimation PRIVATE
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

# ==============================================================================
# Platform-Specific Configuration (Windows)
# ==============================================================================
if(WIN32)
    # Copy data assets (models, test images) to output directory
    file(COPY "${CMAKE_SOURCE_DIR}/models" DESTINATION "${CMAKE_BINARY_DIR}")
    file(COPY "${CMAKE_SOURCE_DIR}/data" DESTINATION "${CMAKE_BINARY_DIR}")

    # Copy ONNX Runtime DLLs to output directory
    # without modifying system PATH
    add_custom_command(TARGET PoseEstimation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:PoseEstimation>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
            "$<TARGET_FILE_DIR:PoseEstimation>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:PoseEstimation>"
        COMMENT "Copying ONNX Runtime DLLs to output directory"
    )

    # Configure Visual Studio debugger environment
    # Adds OpenCV and cuDNN paths to PATH when running from VS (F5)
    # CUDNN_BIN_DIR is provided via CMakePresets.json
    set_target_properties(PoseEstimation PROPERTIES 
        VS_DEBUGGER_ENVIRONMENT "PATH=${ONNXRUNTIME_ROOT}/lib;${CUDNN_BIN_DIR};$ENV{PATH}"
    )
endif()