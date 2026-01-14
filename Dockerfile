# ==============================================================================
# STAGE 1: Builder - Compile OpenCV with CUDA and build application
# ==============================================================================
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies: compilers, build tools, OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends\
    build-essential cmake ninja-build git wget unzip pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libtbb-dev gcc-13 g++-13 \
    && rm -rf /var/lib/apt/lists/*

# Build OpenCV 4.12.0 with CUDA support for RTX 5080 (compute capability 8.9 & 9.0)
RUN git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv_contrib.git && \
    mkdir -p opencv/build && cd opencv/build && \
    cmake -G "Ninja" \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          # CUDA configuration for RTX 5080 (compute 8.9, 9.0) with fallback to 40xx series
          BUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,dnn,cudev,cudaimgproc,cudawarping \
          -D WITH_CUDA=ON \
          -D CUDA_ARCH_BIN="8.9;9.0" \
          -D CUDA_ARCH_PTX="9.0" \
           -D ENABLE_FAST_MATH=ON \
          -D CUDA_FAST_MATH=ON \
          -D WITH_CUBLAS=ON \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          # Build optimizations: disable unnecessary modules
          -D BUILD_opencv_apps=OFF \
          -D BUILD_opencv_python3=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_JAVA=OFF \
          -D WITH_GTK=OFF \
          -D WITH_QT=OFF \
          -D WITH_OPENGL=OFF \
          -D WITH_VTK=OFF \
          .. && \
    ninja -j$(nproc) install && \
    ldconfig

# Download ONNX Runtime GPU (v1.23.2 with CUDA 12.x support)
WORKDIR /opt
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-1.23.2.tgz && \
    mv onnxruntime-linux-x64-gpu-1.23.2 onnxruntime

# Build C++ application with CMake
WORKDIR /app_build
COPY . .
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
RUN cmake --preset linux-release && \
    cmake --build --preset linux-release

# ==============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ==============================================================================
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

# Install runtime libraries only (no dev packages to minimize image size)
RUN apt-get update && apt-get install -y --no-install-recommends\
    # Video/image codecs (FFmpeg provides libavcodec, libavformat, libswscale)
    ffmpeg \
    
    # Image format libraries (Ubuntu 24.04 package names)
    libjpeg-turbo8 \
    libpng16-16 \
    libtiff6 \
    libwebp7 \
    libwebpdemux2 \
    libwebpmux3 \
    libopenexr-3.1-30 \
    
    # GStreamer for video stream support
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    
    # Threading library (OpenCV TBB backend)
    libtbb12 \
    
    && rm -rf /var/lib/apt/lists/*

# Copy compiled OpenCV libraries and headers from builder
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/include/opencv4 /usr/local/include/opencv4

# Copy ONNX Runtime GPU libraries
COPY --from=builder /opt/onnxruntime/lib /opt/onnxruntime/lib

# Copy built application binary
WORKDIR /app
COPY --from=builder /app_build/out/build/linux-release/PoseEstimation ./PoseEstimation

# Configure library paths for runtime linking
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/onnxruntime/lib:$LD_LIBRARY_PATH
ENV NO_DISPLAY=1

# Update shared library cache
RUN ldconfig

ENTRYPOINT ["/app/PoseEstimation"]