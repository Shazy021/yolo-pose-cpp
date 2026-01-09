#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

namespace cv { namespace cuda { class GpuMat; } }

#ifdef HAVE_OPENCV_CUDAWARPING
#include <opencv2/core/cuda.hpp>
#endif

/// Letterbox transformation parameters for post-processing.
struct LetterboxParams {
    float scale = 1.0f; ///< Resize scale factor applied to original image.
    int pad_w = 0; ///< Horizontal padding (pixels) added on each side.
    int pad_h = 0; ///< Vertical padding (pixels) added on each side.
    int input_w = 0; ///< Final input tensor width.
    int input_h = 0; ///< Final input tensor height.
    int orig_w = 0; ///< Original image width.
    int orig_h = 0; ///< Original image height.
};

/// Preprocessing result for single frame.
/**
 * Contains both CPU and GPU data paths to support heterogeneous execution.
 *
 * **Usage**:
 * - If `use_gpu_buffer == true`: Use `gpu_nchw_buffer` for zero-copy GPU inference.
 * - If `use_gpu_buffer == false`: Use `input_tensor` (CPU path fallback).
 *
 * ONNX engine automatically selects the appropriate buffer based on `use_gpu_buffer`.
 */
struct PreprocessResult {
    // CPU Data (Fallback)
	std::vector<float> input_tensor; ///< Flattened input tensor [1,3,H,W] NCHW format.
    
    // GPU Data (Primary)
    cv::cuda::GpuMat gpu_nchw_buffer; // NCHW Float32 buffer
    bool use_gpu_buffer = false;

	std::vector<std::int64_t> input_shape; ///< Tensor shape {1, 3, H, W}.
	LetterboxParams params; ///< Letterbox transformation parameters.
};

/// Preprocessing result for batch of frames.
/**
 * Batch version of PreprocessResult. All frames share the same target dimensions
 * but may have different original sizes, so each gets its own LetterboxParams.
 *
 * Memory layout (NCHW format):
 * ```
 * [frame0_R, frame0_G, frame0_B, frame1_R, frame1_G, frame1_B, ...]
 * where each channel is flattened H×W image
 */
struct BatchPreprocessResult {
    // CPU Data (Fallback)
    std::vector<float> input_tensor; ///< Flattened batch tensor [N,3,H,W] NCHW format.
    
    // GPU Data (Primary)
    cv::cuda::GpuMat gpu_nchw_buffer; ///< Contiguous GPU buffer for entire batch [N,C,H,W].
    bool use_gpu_buffer = false; ///< True if GPU buffer is active.

    std::vector<int64_t> input_shape;  ///< Tensor shape {batch_size, 3, H, W}.
    std::vector<LetterboxParams> params;  ///< Transformation parameters for each frame in batch.
};

/// Apply letterbox resize + padding + normalization for YOLO input.
/**
 * Resizes image preserving aspect ratio, pads to target size with gray (114,114,114),
 * converts to NCHW float tensor [0,1] range.
 *
 * \param frame    Input image/frame in BGR format.
 * \param target_w Target network input width.
 * \param target_h Target network input height.
 * \param use_gpu  Enable GPU-accelerated preprocessing if available (default: false).
 * \return Preprocessed tensor and letterbox parameters for post-processing.
 */
PreprocessResult preprocess_letterbox(
	const cv::Mat& frame,
	int target_w,
	int target_h,
    bool use_gpu = false
);

/// Apply letterbox preprocessing to multiple frames simultaneously.
/**
 * Batch version of preprocess_letterbox that processes multiple frames efficiently.
 * Each frame is independently resized, padded, and normalized.
 * Results are concatenated into a single batch tensor for inference.
 *
 * GPU acceleration (if enabled) processes each frame's resize operation on GPU
 * before combining into the final batch tensor.
 *
 * \param frames   Vector of input images/frames in BGR format.
 * \param target_w Target network input width.
 * \param target_h Target network input height.
 * \param use_gpu  Enable GPU-accelerated preprocessing if available (default: false).
 * \return Batch tensor [N,3,H,W] and parameters for each frame.
 */
BatchPreprocessResult preprocess_letterbox_batch(
    const std::vector<cv::Mat>& frames,
    int target_w,
    int target_h,
    bool use_gpu = false
);
