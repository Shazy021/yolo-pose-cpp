#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/// Single keypoint with 2D coordinates and confidence score.
struct KeyPoint {
	cv::Point2f point; ///< 2D coordinates (x, y) in image space.
	float confidence; ///< Detection confidence in range [0, 1].
};

/// Detected person with bounding box and pose keypoints.
struct Person {
    cv::Rect bbox; ///< Bounding box around the person.
    float confidence; ///< Overall detection confidence in range [0, 1].
    std::vector<KeyPoint> keypoints; ///< Pose keypoints (typically 17 for COCO format).
};