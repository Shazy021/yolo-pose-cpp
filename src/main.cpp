#include <iostream>
#include <opencv2/opencv.hpp>

#include "onnx_engine.hpp"
#include "input_handler.hpp"
#include "pipeline.hpp"

/// Target network input dimensions (width, height)
const int TARGET_W = 640;
const int TARGET_H = 640;

int main(int argc, char** argv) {
#ifdef _WIN32
    // Suppress verbose OpenCV logging on Windows
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
#endif

    // Parse command-line arguments
    InputConfig config = parse_arguments(argc, argv);

    // Load ONNX model
    std::cout << "Loading: " << config.model_path << std::endl;
    OnnxEngine engine(config.model_path);
    std::cout << "Model loaded successfully!\n" << std::endl;

    // Process input based on detected type
    switch (config.type) {
    case InputType::IMAGE:
        std::cout << "Processing image: " << config.path << std::endl;
        process_image(config.path, config.output_path, engine, TARGET_W, TARGET_H);
        break;

    case InputType::VIDEO:
        std::cout << "Processing video: " << config.path << std::endl;
        process_video(config.path, config.output_path, engine, false, TARGET_W, TARGET_H);
        break;

    case InputType::WEBCAM:
        std::cout << "Starting webcam..." << std::endl;
        process_video("0", config.output_path, engine, true, TARGET_W, TARGET_H);
        break;

    default:
        std::cerr << "Unknown input type!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
