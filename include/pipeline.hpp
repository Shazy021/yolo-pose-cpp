#pragma once
#include <string>
#include "onnx_engine.hpp"

/// Run pose estimation on a single image.
/**
 * \param input_path  Path to the input image.
 * \param output_path Optional path to save the processed image. Empty string means "do not save".
 * \param engine      Initialized ONNX Runtime engine.
 * \param target_w    Network input width (default: 640).
 * \param target_h    Network input height (default: 640).
 */
void process_image(const std::string& input_path,
    const std::string& output_path,
    OnnxEngine& engine, 
    int target_w = 640,
    int target_h = 640);

/// Run pose estimation on a video file or webcam stream.
/**
 * \param input_path  Path to the input video file. Ignored when is_webcam is true.
 * \param output_path Optional path to save the processed video. Empty string means "do not save".
 * \param engine      Initialized ONNX Runtime engine.
 * \param is_webcam   If true, capture frames from default webcam instead of a file.
 * \param target_w    Network input width (default: 640).
 * \param target_h    Network input height (default: 640).
 */
void process_video(const std::string& input_path,
    const std::string& output_path,
    OnnxEngine& engine,
    bool is_webcam = false,
    int target_w = 640,
    int target_h = 640);
