# 🚀 YOLO Pose Estimation (C++/ONNX Runtime/CPU/Docker)

**Human pose estimation** using **YOLOv8/YOLO11 pose models** with **native C++**, **ONNX Runtime**, and **Docker deployment**.

## 🖼️ Example before / after

| Input | Pose Estimation |
|:-----:|:---------------:|
| <p align="center"><img src="./data/test_1.jpg" width="600"/></p> | <p align="center"><img src="./output/res_1.jpg" width="600"/></p> |
| <p align="center"><img src="./data/test_2.jpg" width="600"/></p> | <p align="center"><img src="./output/res_2.jpg" width="600"/></p> |

## 🎥 Demos
>  ⚠️ **Note**: GIFs may take time to load (~20 MB each). If not loading, check [`./data/`](./data/) folder.
> >📌 **Legacy footage**: These demos show the original CPU-only implementation. Current GPU-accelerated version achieves 3-5x higher FPS (see [`benchmark.md`](./BENCHMARKS.md)).

<p align="center">
  <img src="./data/out_1_small.gif" width="70%" />
  <img src="./data/out_2_small.gif" width="70%" />
  <img src="./data/out_3_small.gif" width="70%" />
</p>

## ✨ Features
- **Multi-model support**: YOLOv8n/YOLO11n pose ONNX models
- **Batch Inference**: Process multiple frames simultaneously for higher throughput (1-32 frames)
- **Dynamic input size**: Configurable at runtime (480×480, 640×640, 1280×640, etc.)
- **Full pose pipeline**: letterbox → GPU inference → NMS → 17 COCO keypoints + skeleton
- **Production CLI**: images/videos/webcam + output saving
- **Multi-stage Docker build**: (CUDA 12.9 + OpenCV 4.12.0 + ORT gpu v1.23.2) **image 10GB**
- **Cross-platform**: Windows (Visual Studio) + Linux (Docker)

**Tested on**: Intel Xeon E5-2680 v4 @ 2.4GHz, 32GB RAM, Windows 10/11, **Tested on:** NVIDIA RTX 5080, CUDA 12.9, Ubuntu 24.04 Docker

## 🎯 Quick Start

### **Prerequisites**

- Docker with NVIDIA GPU support
- NVIDIA drivers (for CUDA 12.9)
- [Models](./models/) (yolov8n-pose.onnx included)


### **1. Clone repository**

```bash
git clone https://github.com/Shazy021/yolo-pose-cpp.git
cd yolo-pose-cpp
```


### **2. Docker GPU**

```bash
# Build GPU image (10GB)
docker compose build

# Image → image
docker compose run --rm pose -i /app/data/test_1.jpg -o /app/output/res_1.jpg

# High-res inference
docker compose run --rm pose -i /app/data/test_2.jpg -o /app/output/res_hd.jpg -W 1280 -H 1280

# Video → video (batch=4)
docker compose run --rm pose -i /app/data/test_benchmark.mp4 -o /app/output/res.mp4 -b 4

# YOLO11 model
docker compose run --rm pose -m /app/models/yolo11n-pose.onnx -i /app/data/test_1.jpg -o /app/output/res_y11.jpg
```


### **3. Local Windows (Visual Studio)**

```cmd
# Open CMakePresets.json in VS → Build → Release x64
# Or from build dir:
PoseEstimation.exe -i data\test_1.jpg -o output\res_1.jpg -b 4
```


## ⚙️ CLI Parameters

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `-i, --input` | `str` | **required** | Input path: image/video or `0` (webcam) |
| `-o, --output` | `str` | *none* | Output path (saves result if specified) |
| `-m, --model` | `str` | `yolov8n-pose.onnx` | Path to ONNX model |
| `-W, --width` | `int` | `640` | Inference width (**multiple of 32**) |
| `-H, --height` | `int` | `640` | Inference height (**multiple of 32**) |
| `-b, --batch` | `int` | `1` | Batch size (1-32) **for video only** |
| `-h, --help` | `flag` | - | Show help |

**Input dimensions must be multiples of 32** (YOLO stride requirement).

## 📦 Models

```
models/
├── yolov8n-pose.onnx # default model
└── yolo11n-pose.onnx # YOLO11 pose
```

You can either use these models or export your own from Ultralytics checkpoints.

### Export your own models

#### Export script parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | `str` | **required** | Path to YOLO pose `.pt` model or model name (e.g., `yolov8n-pose.pt`, `yolo11n-pose.pt`) |
| `--output-dir` | `str` | `models` | Directory to save exported ONNX model |
| `--imgsz` | `int [int]` | `640` | Image size for export (single value or H W pair) |
| `--opset` | `int` | `17` | ONNX opset version (use 17+ for ONNX Runtime 1.16+) |
| `--dynamic` | `flag` | `False` | Enable dynamic input shapes (allows runtime size changes) |
| `--simplify` | `flag` | `False` | Run ONNX simplifier to optimize graph |

#### Installation
```bash
pip install ultralytics onnx
```
YOLOv8 pose → ONNX
```bash
python scripts/export_yolo_pose_onnx.py
--model yolov8n-pose.pt
--output-dir models
```

YOLO11 pose → ONNX
```bash
python scripts/export_yolo_pose_onnx.py
--model yolo11n-pose.pt
--output-dir models
--dynamic
--simplify
```
---

## 🏗️ Architecture

```
[Image/Video/Webcam] → input_handler → pipeline →
├── preprocess_letterbox (BGR→NCHW , letterbox scale/pad)
├── OnnxEngine (ONNX Runtime CPU, opset 17+, dynamic input)
├── yolo_pose_postprocess ( → Person structs + NMS)
└── visualize_results (bbox(green)+keypoints(red)+skeleton(blue))
```
---

## ⚙️ Performance Tips

### Key Improvements:

- **GPU Preprocessing**: 57% faster vs CPU
- **Zero-Copy Pipeline**: Eliminated CPU↔GPU transfer overhead (-23% latency)
- **Optimal Batching**: 3-5x FPS improvement with batch=8-16

**See full analysis:** [`benchmark.md`](./BENCHMARKS.md)

---

## ⚠️ Limitations

- Input dimensions multiples of 32 (YOLO architecture)
- COCO 17-keypoint pose only
- Input dimensions must be multiples of 32 (YOLO architecture requirement)
- Batch size must be between 1 and 32

## 📄 License

MIT License
---

**Pose estimation**