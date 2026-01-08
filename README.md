# 🚀 YOLO Pose Estimation (C++/ONNX Runtime/CPU/Docker)

**Human pose estimation** using **YOLOv8/YOLO11 pose models** with **native C++**, **ONNX Runtime**, and **Docker deployment**.

## 🖼️ Example before / after

| Input | Pose Estimation |
|:-----:|:---------------:|
| <p align="center"><img src="./data/test_1.jpg" width="600"/></p> | <p align="center"><img src="./output/res_1.jpg" width="600"/></p> |
| <p align="center"><img src="./data/test_2.jpg" width="600"/></p> | <p align="center"><img src="./output/res_2.jpg" width="600"/></p> |

## 🎥 Demos
> ⚠️ **Note**: GIFs may take time to load (~20 MB each). If not loading, check [`./data/`](./data/) folder.

<p align="center">
  <img src="./data/out_1_small.gif" width="70%" />
  <img src="./data/out_2_small.gif" width="70%" />
  <img src="./data/out_3_small.gif" width="70%" />
</p>

## ✨ Features
- **Multi-model support**: YOLOv8n/YOLO11n pose ONNX models
- **Batch Inference**: Process multiple frames simultaneously for higher throughput (1-32 frames)
- **Dynamic input size**: Configurable at runtime (480×480, 640×640, 1280×640, etc.)
- **Full pose pipeline**: letterbox → ONNX inference → NMS → 17 COCO keypoints + skeleton
- **Production CLI**: images/videos/webcam + output saving
- **Docker**: **2GB** (optimized, no PyTorch) / **23GB** (with ultralytics export)
- **Cross-platform**: Windows (Visual Studio) + Linux (Docker)

Tested on: Intel Xeon E5-2680 v4 @ 2.4GHz, 32GB RAM, Windows 10/11

## 🎯 Quick Start

### 1. Clone repository

```bash
git clone https://github.com/Shazy021/yolo-pose-cpp.git
cd yolo-pose-cpp
```

### 2. CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-i, --input` | `str` | **required** | Input file path (image/video) or `0` for webcam |
| `-o, --output` | `str` | *none* | Output file path (optional, saves result if specified) |
| `-m, --model` | `str` | Platform-dependent* | Path to ONNX model file |
| `-W, --width` | `int` | `640` | Input width for inference (must be multiple of 32) |
| `-H, --height` | `int` | `640` | Input height for inference (must be multiple of 32) |
| `-b, --batch`  | `int` | `1`| Batch size for video inference (range: 1-32)|
| `-h, --help` | `flag` | - | Show help message |

**Default model paths:**
- Windows: `models/yolov8n-pose.onnx`
- Docker: `/app/models/yolov8n-pose.onnx`

> **Note**: Input dimensions must be **multiples of 32** (YOLO stride requirement). Valid examples: 320, 480, 640, 960, 1280.

### Docker (Recommended, 2GB image)


| Task | Command |
|------|---------|
| Build image | `docker compose build` |
| Image → image | `docker compose run --rm pose -i /app/data/test_1.jpg -o /app/output/res_1.jpg` |
| Image (inference 1280×1280) | `docker compose run --rm pose -i /app/data/test_2.jpg -o /app/output/res_hd.jpg -W 1280 -H 1280` |
| Image (inference 480×1280) | `docker compose run --rm pose -i /app/data/test_2.jpg -o /app/output/res_custom_res.jpg -W 480 -H 1280` |
| Video → video | `docker compose run --rm pose -i /app/data/test_vid.mp4 -o /app/output/res_vid.mp4` |
| Use YOLO11 | `docker compose run --rm pose -m /app/models/yolo11n-pose.onnx -i /app/data/test_2.jpg -o /app/output/res_y11.jpg` |

### Local (Windows)

| Task | Command (example) |
|------|-------------------|
| Image → image | `PoseEstimation.exe -i data\test_1.jpg -o output\res_1.jpg` |
| Image (custom inference size) | `PoseEstimation.exe -i data\test_1.jpg -o output\res_480.jpg -W 480 -H 480` |
| Video → video | `PoseEstimation.exe -i data\Test_vid.mp4 -o output\res_vid.mp4` |
| Video (high-res) | `PoseEstimation.exe -i data\Test_vid.mp4 -o output\res_hd.mp4 -W 1280 -H 1280` |
| YOLO11 model | `PoseEstimation.exe -m models\yolo11n-pose.onnx -i data\test_2.jpg -o output\res_2.jpg` |
| Webcam | `PoseEstimation.exe -i 0` |

## 🎮 Experimental GPU Support (Windows)

> ⚠️ **Experimental Feature**: GPU acceleration is currently tested only on Windows with NVIDIA GPUs. Docker GPU support is not tested YET.

### Requirements


- CUDA Toolkit 12.x
- cuDNN 9.x
- ONNX Runtime GPU 1.23.2+ (included in project)


### Automatic GPU Detection

The application automatically detects and uses GPU if available:

1. Attempts **TensorRT** provider (best performance, not yet fully tested)
2. Falls back to **CUDA** provider (tested and working)
3. Falls back to **CPU** if GPU unavailable

Check console output on startup to see which provider is active:

```
[ONNX] Available providers: TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider
[ONNX] Using CUDA ExecutionProvider (GPU)
```

## 📦 Models

```
models/
├── yolov8n-pose.onnx # default model
└── yolo11n-pose.onnx # YOLO11 pose
```

You can either use these models or export your own from Ultralytics checkpoints.

### Export your own models (host, optional)

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

Then:
```bash
docker compose run --rm pose
-m /app/models/your_model.onnx
-i /app/data/test_1.jpg
-o /app/output/res_custom.jpg
-W 960 -H 960
```
---

## 🧰 Optional: export models inside Docker (Ultralytics)

By default the image is lightweight (~2 GB) and does **not** include Ultralytics / PyTorch.  
If you need to export ONNX models **inside Docker**, there are two options.

### 1) Enable built-in export in Dockerfile (heavy image, ~23 GB)

In `Dockerfile` there is an optional block you can enable:

```
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
```
After uncommenting and rebuilding, the image will:

- install `ultralytics` + `onnx` inside the container  
- export `yolov8n-pose.pt` → `yolov8n-pose.onnx` into `/app/models`

…but the image size will grow to roughly **23 GB**.

### 2) One-off export using an existing container

If you already have a container with Python and Ultralytics available, you can run the export script **once**.

Example (from project root, using the existing image):

Windows (CMD):
```bash
docker run --rm ^
  --entrypoint python3 ^
  -v "%cd%/models:/app/models" ^
  pose-estimation-cpu ^
  /app/scripts/export_yolo_pose_onnx.py ^
  --model yolov8n-pose.pt ^
  --output-dir /app/models ^
  --dynamic ^
  --simplify
```

Linux/macOS:
```bash
docker run --rm \
  --entrypoint python3 \
  -v "$(pwd)/models:/app/models" \
  pose-estimation-cpu \
  /app/scripts/export_yolo_pose_onnx.py \
  --model yolov8n-pose.pt \
  --output-dir /app/models \
  --dynamic \
  --simplify
```


The script will:

- call `YOLO("yolov8n-pose.pt")` inside the container  
- Ultralytics will download the checkpoint by name (if needed)  
- export it to `yolov8n-pose.onnx` and move it into `/app/models` (mapped to your local `./models`)  

You can replace `--model` with any other Ultralytics pose model name, e.g.:
```
--model yolo11n-pose.pt
```



For most users the recommended approach is still:

- export `.onnx` on the **host**, and  
- keep it in the main Docker image.
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

### Input Size vs Speed Trade-off

| Input Size | Speed | Use Case |
|------------|----------------|----------|
| 320×320 | faster | Fast tracking, low-res |
| 480×480 | faster | Balanced speed/accuracy |
| 480×960 | faster | Portrait/vertical videos |
| 640×640 | **Default** | Standard accuracy |
| 960×960 | slower | High-res images |
| 1280×1280 | slower | Maximum accuracy, distant persons |


---

## ⚠️ Limitations

- GPU support is experimental (Windows only, Docker GPU not tested YET)
- COCO 17-keypoint pose only
- Input dimensions must be multiples of 32 (YOLO architecture requirement)
- Batch size must be between 1 and 32

## 📄 License

MIT License
---

**Pose estimation**