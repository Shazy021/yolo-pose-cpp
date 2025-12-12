#include "visualize_results.hpp"

namespace {

    /// Skeleton connections for COCO-style 17-keypoint layout.
    /**
     * Each pair defines a line between two keypoint indices.
     */
    static const std::vector<std::pair<int, int>> SKELETON = {
        {0,1},{0,2},{1,3},{2,4},
        {5,6},{5,7},{7,9},
        {6,8},{8,10},
        {5,11},{6,12},{11,12},
        {11,13},{13,15},
        {12,14},{14,16}
    };

} // namespace

cv::Mat visualize_pose_results(
    const cv::Mat& image,
    const std::vector<Person>& persons,
    float keypoint_threshold,
    double fps)
{
    // Work on a copy to keep original image intact
    cv::Mat result = image.clone();

    for (const auto& person : persons) {
        // Draw bb
        cv::rectangle(result, person.bbox, cv::Scalar(0, 255, 0), 2);

        std::string label = "Person " +
            std::to_string(static_cast<int>(person.confidence * 100)) + "%";
        cv::putText(result, label,
            cv::Point(person.bbox.x, person.bbox.y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 255, 0), 2);

        // Draw keypoints as small red circles
        for (const auto& kp : person.keypoints) {
            if (kp.confidence > keypoint_threshold) {
                cv::circle(result, kp.point, 3,
                    cv::Scalar(0, 0, 255), -1);
            }
        }

        // Draw skeleton as blue lines between connected keypoints
        for (const auto& [start, end] : SKELETON) {
            const auto& kp1 = person.keypoints[start];
            const auto& kp2 = person.keypoints[end];

            if (kp1.confidence > keypoint_threshold &&
                kp2.confidence > keypoint_threshold) {
                cv::line(result, kp1.point, kp2.point,
                    cv::Scalar(255, 0, 0), 2);
            }
        }
    }

    // Draw FPS in the top-left corner if provided.
    if (fps > 0.0) {
        cv::putText(result, "FPS: " + std::to_string(static_cast<int>(fps)),
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 1.0,
            cv::Scalar(0, 255, 0), 2);
    }

    return result;
}
