#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "model_types.hpp"
#include "preprocess_letterbox.hpp"

/// Decode YOLO pose ONNX output into high-level Person structures.
/**
 * Expected model output shape: [1, 56, num_proposals] or [1, num_proposals, 56],
 * where 56 = 4 box coords + 1 objectness + 1 class logit + 17 * 3 keypoints.
 *
 * \param output_data   Pointer to raw output tensor data (float).
 * \param output_shape  Output tensor shape, e.g. {1, 56, 8400}.
 * \param params        Letterbox parameters used during preprocessing.
 * \param persons       Output vector of detected persons with bboxes and keypoints.
 * \param conf_thresh   Confidence threshold for filtering detections.
 * \param nms_thresh    IoU threshold for Non-Maximum Suppression (NMS).
 *
 * \throws std::runtime_error on unexpected output shapes.
 */
void yolo_pose_postprocess(
    const float* output_data,
    const std::vector<int64_t>& output_shape,
    const LetterboxParams& params,
    std::vector<Person>& persons,
    float conf_thresh = 0.3f,
    float nms_thresh = 0.3f
);
