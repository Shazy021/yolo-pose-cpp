FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies: build tools, OpenCV, Python (for optional model export)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libopencv-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===== ONNX Runtime CPU =====
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz && \
    tar -xzf onnxruntime-linux-x64-1.19.2.tgz && \
    mv onnxruntime-linux-x64-1.19.2 /opt/onnxruntime && \
    rm onnxruntime-linux-x64-1.19.2.tgz

ENV ONNXRUNTIME_DIR=/opt/onnxruntime
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

# Copy C++ source code and build configuration
COPY include/ /app/include/
COPY src/ /app/src/
COPY CMakeLists.txt /app/

# Create folders for models / input data / output
RUN mkdir -p /app/models /app/data /app/output

# Build C++ application
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . --config Release && \
    cp PoseEstimation /app/

# ============================================
# Optional: export YOLO pose model to ONNX inside Docker
# RECOMMENDED: run scripts/export_yolo_pose_onnx.py on the HOST
# to avoid pulling heavy torch/ultralytics into the image.
#
# If you really need to export inside Docker, uncomment:
#
# COPY scripts/export_yolo_pose_onnx.py /app/scripts/
# RUN pip3 install ultralytics onnx
# RUN python3 /app/scripts/export_yolo_pose_onnx.py \
#        --model yolov8n-pose.pt \
#        --output-dir /app/models
# ============================================

# Default entrypoint: show CLI help (can be overridden by docker run ...)
ENTRYPOINT ["/app/PoseEstimation"]
CMD ["--help"]