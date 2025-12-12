#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

/// Output from ONNX inference: raw data buffer and tensor shape.
struct OnnxOutput {
	float* data = nullptr; ///< Pointer to raw float data.
	std::vector<int64_t> shape; ///< Tensor shape (dimensions).
};

/// ONNX Runtime inference engine for YOLO pose models.
/**
 * Supports single input/output models with dynamic batch size.
 * Automatically detects input/output names and model input shape.
 */
class OnnxEngine {
public:
	/// Create engine and load ONNX model from disk.
	/**
	 * \param model_path Path to ONNX model file (.onnx).
	 * \throws std::runtime_error if model file not found or invalid.
	 */
	explicit OnnxEngine(const std::string& model_path);

	/// Run inference on preprocessed input tensor.
	/**
	 * \param input       Input data as 1D vector (NHWC or NCHW format).
	 * \param input_shape Expected input tensor shape [N,C,H,W].
	 * \return Output tensor data and shape.
	 * \throws std::runtime_error if input size doesn't match shape.
	 */
	OnnxOutput run(const std::vector<float>& input,
		const std::vector<int64_t>& input_shape);

	/// Get model input shape [N,C,H,W] from loaded ONNX model.
	const std::vector<int64_t>& modelInputShape() const { return model_input_shape_; }

private:
	Ort::Env env_;
	Ort::SessionOptions session_options_;
	Ort::Session session_;
	Ort::MemoryInfo memory_info_;

	std::string input_name_;
	std::string output_name_;
	std::vector<const char*> input_names_;
	std::vector<const char*> output_names_;

	std::vector<int64_t> model_input_shape_;
};
