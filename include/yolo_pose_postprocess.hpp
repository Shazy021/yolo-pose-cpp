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
 * Processing steps:
 * 1. Transpose output to [num_proposals, 56] format
 * 2. Filter detections by confidence threshold
 * 3. Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
 * 4. Transform coordinates from letterbox space back to original image space
 * 5. Extract and transform keypoints for each person
 *
 * \param output_data   Pointer to raw output tensor data (float).
 * \param output_shape  Output tensor shape, e.g. {1, 56, 8400}.
 * \param params        Letterbox parameters used during preprocessing.
 * \param persons       Output vector of detected persons with bboxes and keypoints.
 * \param conf_thresh   Confidence threshold for filtering detections (default: 0.45).
 * \param nms_thresh    IoU threshold for Non-Maximum Suppression (NMS) (default: 0.5).
 *
 * \throws std::runtime_error on unexpected output shapes.
 */
void yolo_pose_postprocess(
    const float* output_data,
    const std::vector<int64_t>& output_shape,
    const LetterboxParams& params,
    std::vector<Person>& persons,
    float conf_thresh = 0.45f,
    float nms_thresh = 0.5f
);

/// Decode batch YOLO pose ONNX output into Person structures for multiple frames.
/**
 * Batch version of yolo_pose_postprocess that handles multiple frames simultaneously.
 * Expected output shape: [batch_size, 56, num_proposals] or [batch_size, num_proposals, 56].
 *
 * \param output_data      Pointer to raw batch output tensor data (float).
 * \param output_shape     Output tensor shape, e.g. {4, 56, 8400} for batch_size=4.
 * \param params           Vector of letterbox parameters, one per frame in batch.
 * \param batch_persons    Output: vector of Person vectors (one vector per frame).
 * \param CONF_THRESHOLD   Confidence threshold for filtering detections (default: 0.45).
 * \param NMS_THRESHOLD    IoU threshold for Non-Maximum Suppression (default: 0.5).
 *
 * \throws std::runtime_error on unexpected output shapes.
 */
void yolo_pose_postprocess_batch(
    const float* output_data,
    const std::vector<int64_t>& output_shape,
    const std::vector<LetterboxParams>& params,
    std::vector<std::vector<Person>>& batch_persons,
    float CONF_THRESHOLD = 0.45f,
    float NMS_THRESHOLD = 0.5f
);
