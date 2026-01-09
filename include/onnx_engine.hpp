#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <optional>

#include "preprocess_letterbox.hpp"

/// Output from ONNX inference: raw data buffer and tensor shape.
struct OnnxOutput {
	float* data = nullptr; ///< Pointer to raw float data.
	std::vector<int64_t> shape; ///< Tensor shape (dimensions).
};

/// ONNX Runtime inference engine for YOLO pose models.
/**
 * Supports single input/output models with dynamic batch size.
 * Automatically detects input/output names and model input shape.
 * 
 * Execution provider priority (best to worst):
 * 1. TensorRT - Optimized for NVIDIA GPUs with graph optimization
 * 2. CUDA - Standard NVIDIA GPU acceleration
 * 3. CPU - Fallback for systems without GPU support
 * 
 * Zero-Copy GPU Pipeline** (when CUDA is available):
 * - Accepts preprocessed data already in GPU memory
 * - Binds GPU buffer directly to inference session
 */
class OnnxEngine {
public:
	/// Create engine and load ONNX model from disk.
	/**
	 * Automatically selects the best available execution provider:
	 * - TensorRT: Maximum performance with JIT optimization
	 * - CUDA: High performance GPU acceleration
	 * - CPU: Multi-threaded fallback
	 * 
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
	
	/// Run inference on preprocessed single frame with automatic GPU/CPU handling.
	/**
	 * Automatically selects optimal data path:
	 * - If GPU preprocessing was used: Zero-copy inference (no transfers)
	 * - Otherwise: Standard CPU tensor path
	 *
	 * \param preprocessed Preprocessed frame from preprocess_letterbox().
	 * \return Output tensor data and shape.
	 * \throws std::runtime_error if tensor creation fails.
	 */
	OnnxOutput run(const PreprocessResult& preprocessed);

	/// Run inference on preprocessed batch with automatic GPU/CPU handling.
	/**
	 * Batch version of run(). Handles multiple frames in a single inference pass.
	 *
	 * \param preprocessed Preprocessed batch from preprocess_letterbox_batch().
	 * \return Output tensor data and shape for entire batch.
	 * \throws std::runtime_error if tensor creation fails.
	 */
	OnnxOutput run(const BatchPreprocessResult& preprocessed);

	/// Get model input shape [N,C,H,W] from loaded ONNX model.
	const std::vector<int64_t>& modelInputShape() const { return model_input_shape_; }

	/// Check if GPU execution provider is active.
	/**
	 * Returns true if using TensorRT or CUDA, false if using CPU.
	 * Used by preprocessing to determine optimal memory layout.
	 *
	 * \return true if GPU-accelerated, false otherwise.
	 */
	bool isUsingGPU() const { return using_gpu_; }

private:
	Ort::Env env_;
	Ort::SessionOptions session_options_;
	Ort::Session session_;
	Ort::MemoryInfo memory_info_; ///< CPU memory allocator

	/// GPU memory info for Zero-Copy inference (only if CUDA is available).
	std::optional<Ort::MemoryInfo> cuda_memory_info_;

	std::string input_name_;
	std::string output_name_;
	std::vector<const char*> input_names_;
	std::vector<const char*> output_names_;

	std::vector<int64_t> model_input_shape_;
	bool using_gpu_ = false; ///< True if TensorRT or CUDA is active.
};
