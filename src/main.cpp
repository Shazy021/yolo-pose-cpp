#include <iostream>
#include <opencv2/opencv.hpp>

#include "onnx_engine.hpp"
#include "input_handler.hpp"
#include "pipeline.hpp"


int main(int argc, char** argv) {
#ifdef _WIN32
    // Suppress verbose OpenCV logging on Windows
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
#endif

    // Parse command-line arguments
    InputConfig config = parse_arguments(argc, argv);

    // Display parsed configuration for debugging and verification
    // Helps users confirm the application is using correct settings
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Input: " << config.path << std::endl;
    std::cout << "Model: " << config.model_path << std::endl;
    std::cout << "Inference resolution: " << config.input_width << "x" << config.input_height << std::endl;
    std::cout << "=====================\n" << std::endl;

    // Load ONNX model
    std::cout << "Loading: " << config.model_path << std::endl;
    OnnxEngine engine(config.model_path);
    std::cout << "Model loaded successfully!\n" << std::endl;

    // Process input based on detected type
    switch (config.type) {
    case InputType::IMAGE:
        // Process single image: load, infer, visualize, optionally save
        std::cout << "Processing image: " << config.path << std::endl;
        process_image(config.path, config.output_path, engine, config.input_width, config.input_height);
        break;

    case InputType::VIDEO:
        // Process video file with configurable batch size for throughput optimization
        std::cout << "Processing video: " << config.path << std::endl;
        process_video(config.path, config.output_path, engine, false, config.input_width, config.input_height, config.batch_size);
        break;

    case InputType::WEBCAM:
        // Process live webcam stream with batch_size=1 for minimal latency
        std::cout << "Starting webcam..." << std::endl;
        process_video("0", config.output_path, engine, true, config.input_width, config.input_height, 1);
        break;

    default:
        // Should never reach here due to validation in parse_arguments
        std::cerr << "Unknown input type!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
