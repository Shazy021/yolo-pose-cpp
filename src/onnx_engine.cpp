#include "onnx_engine.hpp"

#include <stdexcept>
#include <opencv2/core.hpp>  // cv::getNumberOfCPUs()

OnnxEngine::OnnxEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxEngine"),
    session_options_(),
    session_(nullptr),
    memory_info_{ Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault) }
{
    // Configure session options for optimal CPU performance
    session_options_.SetIntraOpNumThreads(cv::getNumberOfCPUs());
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load model (wide char path on Windows)
#ifdef _WIN32
    std::wstring wpath(model_path.begin(), model_path.end());
    session_ = Ort::Session(env_, wpath.c_str(), session_options_);
#else
    session_ = Ort::Session(env_, model_path.c_str(), session_options_);
#endif

    // Get input/output names dynamically
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
    input_names_ = { input_name_.c_str() };
    output_names_ = { output_name_.c_str() };

    // Extract model input shape [N,C,H,W]
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    model_input_shape_ = tensor_info.GetShape();

    // CPU memory allocator for input tensors
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
}

OnnxOutput OnnxEngine::run(const std::vector<float>& input,
    const std::vector<int64_t>& input_shape)
{
    // Validate input size matches expected shape
    size_t need = 1;
    for (auto v : input_shape) {
        need *= static_cast<size_t>(v);
    }
    if (need != input.size()) {
        throw std::runtime_error("OnnxEngine::run: input size mismatch");
    }

    // Create input tensor from float vector
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input.data()),
        need,
        input_shape.data(),
        input_shape.size()
    );

    // Execute inference
    auto output_tensors = session_.Run(
        Ort::RunOptions{ nullptr },
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), 1
    );

    // Extract output tensor data and shape
    OnnxOutput out;
    out.data = output_tensors[0].GetTensorMutableData<float>();
    out.shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    return out;
}
