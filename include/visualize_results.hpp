#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "model_types.hpp"

/// Draw pose estimation results (bounding boxes, keypoints, skeleton, FPS) on an image.
/**
 * \param image              Source image (BGR).
 * \param persons            List of detected persons with keypoints.
 * \param keypoint_threshold Minimum confidence for a keypoint to be drawn.
 * \param fps                Optional FPS value to display in the top-left corner (0.0 = do not draw).
 * \return New image with visualization overlays.
 */
cv::Mat visualize_pose_results(
    const cv::Mat& image,
    const std::vector<Person>& persons,
    float keypoint_threshold = 0.2f,
    double fps = 0.0
);
