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
        # Fallback to HARDCODED paths for local Windows development
        set(CUDNN_DIR "C:/Program Files/NVIDIA/CUDNN/v9.17")
        set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-gpu-1.23.2")
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
    set(OpenCV_DIR "C:/opencv/build_cuda/install")
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

    # Copy CUDA execution provider DLLs if available
    if(EXISTS "${ONNXRT_LIB_DIR}/onnxruntime_providers_cuda.dll")
        add_custom_command(TARGET PoseEstimation POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRT_LIB_DIR}/onnxruntime_providers_cuda.dll"
                    "$<TARGET_FILE_DIR:PoseEstimation>/onnxruntime_providers_cuda.dll"
        )
        add_custom_command(TARGET PoseEstimation POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRT_LIB_DIR}/onnxruntime_providers_shared.dll"
                    "$<TARGET_FILE_DIR:PoseEstimation>/onnxruntime_providers_shared.dll"
        )
        message(STATUS "ONNX Runtime CUDA DLLs will be copied")
    endif()

    # copy ALL cuDNN 9 DLLs
    set(CUDNN_BIN_DIR "${CUDNN_DIR}/bin/12.9")
    
    if(EXISTS "${CUDNN_BIN_DIR}/cudnn64_9.dll")
        file(GLOB CUDNN_DLLS "${CUDNN_BIN_DIR}/cudnn*64_9.dll")
        
        # Copy each found DLL to the executable directory
        foreach(dll ${CUDNN_DLLS})
            get_filename_component(dll_name ${dll} NAME)
            add_custom_command(TARGET PoseEstimation POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dll}"
                        "$<TARGET_FILE_DIR:PoseEstimation>/${dll_name}"
            )
        endforeach()
        
        message(STATUS "cuDNN 9 DLLs will be copied from ${CUDNN_BIN_DIR}")
        message(STATUS "Found cuDNN DLLs: ${CUDNN_DLLS}")
    else()
        message(WARNING "cuDNN DLLs not found at ${CUDNN_BIN_DIR}")
    endif()

endif()