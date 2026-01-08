#include "pipeline.hpp"
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
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
    auto prep = preprocess_letterbox(image, target_w, target_h, engine.isUsingGPU());

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
    int target_h,
    int batch_size)
{
    // Check if display is available (headless mode in Docker sets NO_DISPLAY)
    const char* no_display_env = std::getenv("NO_DISPLAY");
    bool show_window = (no_display_env == nullptr);
    cv::VideoCapture cap;

    // Open video source: webcam or file
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

    // Retrieve video properties for output writer configuration
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps_input = cap.get(cv::CAP_PROP_FPS);

    // Initialize video writer if output path is specified
    cv::VideoWriter writer;
    if (!output_path.empty() && !is_webcam) {
        ensure_parent_dir_exists(output_path);
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(output_path, fourcc, fps_input, cv::Size(frame_width, frame_height));

        if (writer.isOpened()) {
            std::cout << "Saving output to: " << output_path << std::endl;
        }
        else {
            std::cerr << "Failed to open VideoWriter for: " << output_path << std::endl;
        }
    }

    cv::Mat result;
    cv::TickMeter tm;
    int frame_count = 0;

    // Frame buffer for batch processing
    std::vector<cv::Mat> frame_buffer;
    frame_buffer.reserve(batch_size);

    std::cout << "Processing with batch size: " << batch_size << "..." << std::endl;

    // Main processing loop: read frames in batches
    while (true) {
        // Accumulate batch_size frames into buffer
        frame_buffer.clear();
        for (int i = 0; i < batch_size; ++i) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                break;  // End of video stream
            }
            frame_buffer.push_back(frame.clone());
        }

        // Exit if no frames were read (end of video)
        if (frame_buffer.empty()) break;

        BatchPreprocessResult prep;

        try {
            tm.start();

            // Measure preprocessing time for performance analysis
            auto t1 = std::chrono::high_resolution_clock::now();
            // Preprocess entire batch: resize, pad, normalize, and convert to tensor
            prep = preprocess_letterbox_batch(frame_buffer, target_w, target_h, engine.isUsingGPU());


            auto t2 = std::chrono::high_resolution_clock::now();
            // Run batch inference: process all frames in a single forward pass
            OnnxOutput out = engine.run(prep.input_tensor, prep.input_shape);

            auto t3 = std::chrono::high_resolution_clock::now();
            // Decode batch output: extract keypoints for all persons in all frames
            std::vector<std::vector<Person>> batch_persons;
            yolo_pose_postprocess_batch(out.data, out.shape, prep.params, batch_persons);

            auto t4 = std::chrono::high_resolution_clock::now();

            // Calculate effective FPS across the batch
            tm.stop();
            double fps = 1000.0 / tm.getTimeMilli() * frame_buffer.size();
            tm.reset();

            // Visualize and output each frame from the batch
            for (size_t i = 0; i < frame_buffer.size(); ++i) {
                result = visualize_pose_results(frame_buffer[i], batch_persons[i], 0.2f, fps);

                // Write to output video if writer is active
                if (writer.isOpened()) {
                    writer.write(result);
                }

                // Display in window if available (not in headless mode)
                if (show_window) {
                    cv::imshow("YOLO Pose Estimation", result);
                    if (cv::waitKey(1) == 27) goto exit_loop;
                }
            }

            frame_count += frame_buffer.size();

            // Print detailed timing information every 30 frames
            if (frame_count % 30 == 0) {
                auto prep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                auto inf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
                auto post_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

                std::cout << "\n[BATCH TIMING] Prep: " << prep_ms
                    << "ms | Inf: " << inf_ms
                    << "ms (" << (float)inf_ms / batch_size << "ms/frame)"
                    << " | Post: " << post_ms << "ms" << std::endl;
                std::cout << "Processed " << frame_count << " frames | FPS: " << (int)fps << std::flush;
            }

        }
        // Comprehensive exception handling for robust operation
        // Provides detailed error messages to help diagnose issues
        catch (const cv::Exception& e) {
            std::cerr << "\n[ERROR] OpenCV exception: " << e.what() << std::endl;
            break;
        }
        catch (const Ort::Exception& e) {
            std::cerr << "\n[ERROR] ONNX Runtime exception: " << e.what() << std::endl;
            std::cerr << "[DEBUG] Batch size: " << frame_buffer.size() << std::endl;
            if (!prep.input_shape.empty()) {
                std::cerr << "[DEBUG] Tensor shape: [" << prep.input_shape[0] << ", "
                    << prep.input_shape[1] << ", " << prep.input_shape[2] << ", "
                    << prep.input_shape[3] << "]" << std::endl;
            }
            break;
        }
        catch (const std::exception& e) {
            std::cerr << "\n[ERROR] Standard exception: " << e.what() << std::endl;
            break;
        }
        catch (...) {
            std::cerr << "\n[ERROR] Unknown exception!" << std::endl;
            break;
        }
    }

exit_loop:
    std::cout << "\n\nProcessing complete!" << std::endl;

    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    if (show_window) {
        cv::destroyAllWindows();
    }
}
