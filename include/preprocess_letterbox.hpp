#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

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

/// Result of letterbox preprocessing: tensor and transformation parameters.
struct PreprocessResult {
	std::vector<float> input_tensor; ///< Flattened input tensor [1,3,H,W] NCHW format.
	std::vector<std::int64_t> input_shape; ///< Tensor shape {1, 3, H, W}.
	LetterboxParams params; ///< Letterbox transformation parameters.
};

/// Apply letterbox resize + padding + normalization for YOLO input.
/**
 * Resizes image preserving aspect ratio, pads to target size with gray (114,114,114),
 * converts to NCHW float tensor [0,1] range.
 *
 * \param frame    Input image/frame in BGR format.
 * \param target_w Target network input width.
 * \param target_h Target network input height.
 * \return Preprocessed tensor and letterbox parameters for post-processing.
 */
PreprocessResult preprocess_letterbox(
	const cv::Mat& frame,
	int target_w,
	int target_h
);
