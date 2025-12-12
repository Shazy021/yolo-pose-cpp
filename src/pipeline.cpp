#include "pipeline.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "preprocess_letterbox.hpp"
#include "yolo_pose_postprocess.hpp"
#include "visualize_results.hpp"

namespace {

    /// Ensure that parent directory for the given file path exists
    void ensure_parent_dir_exists(const std::string& path) {
        std::filesystem::path p(path);
        auto parent = p.parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                std::cerr << "Warning: failed to create directory " << parent
                    << " : " << ec.message() << std::endl;
            }
        }
    }

} // namespace

void process_image(const std::string& input_path,
    const std::string& output_path,
    OnnxEngine& engine,
    int target_w,
    int target_h)
{
    // Decide whether to show OpenCV window (disabled in Docker mode)
    const char* no_display_env = std::getenv("NO_DISPLAY");
    bool show_window = (no_display_env == nullptr);

    // Load input image.
    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }

    std::vector<Person> persons;

    // Preprocess image to match network input size and aspect ratio
    auto prep = preprocess_letterbox(image, target_w, target_h);

    // Run inference
    OnnxOutput out = engine.run(prep.input_tensor, prep.input_shape);

    // Decode model output into pose detections
    yolo_pose_postprocess(out.data, out.shape, prep.params, persons);

    // Visualize poses on the original image
    cv::Mat result = visualize_pose_results(image, persons, 0.2f, 0.0);

    std::cout << "Detected " << persons.size() << " person(s)" << std::endl;

    // Optionally save result to file
    if (!output_path.empty()) {
        cv::imwrite(output_path, result);
        std::cout << "Result saved to: " << output_path << std::endl;
    }

    // Optionally show result in a window (disabled in Docker)
    if (show_window) {
        cv::imshow("YOLO Pose Estimation", result);
        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

void process_video(const std::string& input_path,
    const std::string& output_path,
    OnnxEngine& engine,
    bool is_webcam,
    int target_w,
    int target_h)
{
    // Decide whether to show OpenCV window (disabled in Docker mode)
    const char* no_display_env = std::getenv("NO_DISPLAY");
    bool show_window = (no_display_env == nullptr);
    cv::VideoCapture cap;

    // Open video source: file or webcam
    if (is_webcam) {
        cap.open(0);
    }
    else {
        cap.open(input_path);
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open video source!" << std::endl;
        return;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps_input = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer;
    if (!output_path.empty() && !is_webcam) {
        ensure_parent_dir_exists(output_path);

        // MP4 container with MPEG-4 codec
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

        writer.open(output_path,
            fourcc,
            fps_input,
            cv::Size(frame_width, frame_height));

        if (writer.isOpened()) {
            std::cout << "Saving output to: " << output_path << std::endl;
        }
        else {
            std::cerr << "Failed to open VideoWriter for: " << output_path
                << " (w=" << frame_width << ", h=" << frame_height
                << ", fps=" << fps_input << ")" << std::endl;
        }
    }

    cv::Mat frame, result;
    std::vector<Person> persons;
    cv::TickMeter tm;
    int frame_count = 0;

    std::cout << "Processing... (Press ESC to exit)" << std::endl;

    while (cap.read(frame)) {
        tm.start();

        // Preprocess frame and run inference
        auto prep = preprocess_letterbox(frame, target_w, target_h);
        OnnxOutput out = engine.run(prep.input_tensor, prep.input_shape);
        yolo_pose_postprocess(out.data, out.shape, prep.params, persons);

        tm.stop();
        double fps = 1000.0 / tm.getTimeMilli();
        tm.reset();

        // Draw poses and FPS on the frame
        result = visualize_pose_results(frame, persons, 0.2f, fps);

        // Optionally write processed frame to output video
        if (writer.isOpened()) {
            writer.write(result);
        }

        // Optionally show live preview
        if (show_window) {
            cv::imshow("YOLO Pose Estimation", result);
            if (cv::waitKey(1) == 27) break;
        }

        frame_count++;
        if (frame_count % 30 == 0) {
            std::cout << "\rProcessed " << frame_count
                << " frames | FPS: " << static_cast<int>(fps)
                << std::flush;
        }
    }

    std::cout << "\n\nProcessing complete!" << std::endl;

    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    if (show_window) {
        cv::destroyAllWindows();
    }
}
