#pragma once

#include <string>
#include <opencv2/opencv.hpp>

/// Type of input source.
enum class InputType {
    IMAGE,   ///< Image file input.
    VIDEO,   ///< Video file input.
    WEBCAM,  ///< Webcam stream input.
    UNKNOWN  ///< Unknown or unsupported input type.
};

/// Configuration parsed from command-line arguments.
struct InputConfig {
    InputType type;          ///< Detected input type.
    std::string path;        ///< Input path (image, video file or webcam id).
    std::string output_path; ///< Optional output file path.
    bool save_output;        ///< Whether to save processed output.
    std::string model_path;  ///< Path to ONNX model.
};

/// Detect input type based on path or special values like "0" / "webcam".
InputType detect_input_type(const std::string& path);

/// Parse command-line arguments into InputConfig.
/// Exits the application on invalid arguments.
InputConfig parse_arguments(int argc, char** argv);

/// Print CLI usage/help message to stdout.
void print_usage(const char* program_name);
