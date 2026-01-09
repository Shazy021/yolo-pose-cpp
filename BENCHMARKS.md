# 📊 Performance Benchmarks

> **Hardware**: Intel Xeon E5-2680 v4 @ 2.4GHz, NVIDIA GPU (CUDA), 32GB RAM  
> **Software**: Windows 11, CUDA 12.x, cuDNN 9.x, ONNX Runtime 1.23.2  
> **Model**: `yolo11n-pose.onnx` (FP32)  
> **Input**: 1080p Video (1920x1080, ~30s)

## 1. Baseline (GPU infer CPU Preprocessing)
*Date: 2026-01-08*  
*State: Original implementation with OpenCV CPU resizing.*

| Resolution | Batch Size | Preprocessing (ms) | Inference (ms/frame) | Postprocessing (ms) | **FPS** | Bottleneck |
|:----------:|:----------:|:------------------:|:--------------------:|:-------------------:|:-------:|:----------:|
| **640×640** | 1 | ~7 | 18.0 | ~1 | **37** | Inference |
| **640×640** | 4 | ~27 | 5.2 | ~3 | **75** | Balanced |
| **640×640** | 8 | ~53 | **3.0** | ~7 | **95** | **Preprocessing (63%)** |
| **640×640** | 16 | ~107 | **2.1** | ~15 | **102** | **Preprocessing (68%)** |
| **1280×1280** | 1 | ~21 | 22.0 | ~3 | **21** | Balanced |
| **1280×1280** | 4 | ~88 | 8.5 | ~14 | **29** | **Preprocessing (64%)** |
| **1280×1280** | 8 | ~172 | 9.0 | ~29 | **29** | **Preprocessing (65%)** |

---

### 2️⃣ GPU Zero-Copy Pipeline
**Date**: 2026-01-09  
**Implementation**: Full GPU preprocessing (`cv::cuda`) + Zero-copy inference (no CPU↔GPU transfers)

**Changes**:
- ✅ GPU-accelerated resize (`cv::cuda::resize`)
- ✅ GPU-accelerated padding (`cv::cuda::copyMakeBorder`)
- ✅ GPU normalization + HWC→NCHW conversion (custom CUDA kernel)
- ✅ Direct GPU memory binding to ONNX Runtime (`Ort::MemoryInfo::CreateCuda`)

| Resolution | Batch | Prep (ms) | Infer (ms/frame) | Post (ms) | **FPS** | **Speedup** |
|:----------:|:-----:|:---------:|:----------------:|:---------:|:-------:|:-----------:|
| 640×640 | 1 | 6 | 18.0 | 1 | **39** | **+5%** |
| 640×640 | 4 | 19 | 4.5 | 4 | **95** | **+26%** |
| 640×640 | 8 | 37 | 2.3 | 7 | **126** | **+32%** |
| 640×640 | 16 | 71 | 1.4 | 15 | **148** | **+45%** |
| 1280×1280 | 1 | 11 | 18.0 | 3 | **30** | **+42%** |
| 1280×1280 | 4 | 38 | 5.2 | 14 | **54** | **+86%** |
| 1280×1280 | 8 | 74 | 4.5 | 29 | **58** | **+100%** 🚀|

**Impact**:
- Preprocessing time: **-57%** (172ms → 74ms @ 1280×1280, b=8)
- Inference latency: **-23%** (eliminated CPU→GPU copy overhead)
- **1280×1280 doubled FPS** from 29 to 58 at batch=8

---
## 📈 Performance Analysis

### Speedup by Resolution

640×640: 37 FPS → 148 FPS (+300% with optimal batching)

1280×1280: 21 FPS → 58 FPS (+176% with optimal batching)

## 🚀 Future Optimizations

### Planned
- [ ] **TensorRT Engine** (graph optimization, kernel fusion) — *Expected: +20-30% FPS*
- [ ] **Postprocessing on GPU** (CUDA NMS kernel) — *Expected: -15ms latency*

### Experimental
- [ ] **NVIDIA Video Codec SDK** (GPU video decoding) — *Could eliminate 40-70ms decode time*
- [ ] **Multi-stream processing** — *Better GPU utilization*
---

**Last Updated**: 2026-01-09