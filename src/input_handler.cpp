#include "input_handler.hpp"

#include <iostream>
#include <algorithm>

InputType detect_input_type(const std::string& path) {
    // Special values for webcam input
    if (path == "0" || path == "webcam") {
        return InputType::WEBCAM;
    }

    // Extract file extension
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Image formats
    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
        return InputType::IMAGE;
    }
    // Video formats
    if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv") {
        return InputType::VIDEO;
    }

    return InputType::UNKNOWN;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  -i, --input <path>     Input file (image/video) or '0' for webcam\n"
        << "  -o, --output <path>    Output file (optional, for saving results)\n"
        << "  -m, --model <path>     ONNX model path (optional, default: yolo11n-pose.onnx)\n"
        << "  -h, --help             Show this help message\n\n"
        << "Examples:\n"
        << "  " << program_name << " -i data/image.jpg -m models/yolo11n-pose.onnx\n"
        << "  " << program_name << " -i data/video.mp4 -o output/output.mp4\n"
        << "  " << program_name << " -i 0 -m models/yolo11n-pose.onnx\n";
}

InputConfig parse_arguments(int argc, char** argv) {
    InputConfig config;
    config.type = InputType::UNKNOWN;
    config.save_output = false;

#ifdef _WIN32
    // Default model path for local Windows runs.
    config.model_path = "models/yolov8n-pose.onnx";
#else
    // Default model path inside Docker container.
    config.model_path = "/app/models/yolov8n-pose.onnx";
#endif

    if (argc < 2) {
        print_usage(argv[0]);
        exit(1);
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
        else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                config.path = argv[++i];
                config.type = detect_input_type(config.path);
            }
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                config.output_path = argv[++i];
                config.save_output = true;
            }
        }
        else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                config.model_path = argv[++i];
            }
        }
    }

    if (config.type == InputType::UNKNOWN) {
        std::cerr << "Error: Invalid or missing input file!\n\n";
        print_usage(argv[0]);
        exit(1);
    }

    return config;
}
