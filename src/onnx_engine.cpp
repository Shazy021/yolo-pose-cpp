#include "onnx_engine.hpp"

#include <stdexcept>
#include <iostream>
#include <opencv2/core.hpp>  // cv::getNumberOfCPUs()

OnnxEngine::OnnxEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxEngine"),
    session_options_(),
    session_(nullptr),
    memory_info_{ Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault) }
{
    // This list depends on how ONNX Runtime was compiled (with/without GPU support)
    std::vector<std::string> available_providers = Ort::GetAvailableProviders();

    std::cout << "[ONNX] Available providers: ";
    for (const auto& p : available_providers) {
        std::cout << p << " ";
    }
    std::cout << std::endl;

    bool gpu_enabled = false;

    // Priority 1: TensorRT ExecutionProvider
    // Provides best performance through kernel fusion and FP16/INT8 optimization
    // Performs Just-In-Time (JIT) compilation for target GPU architecture
    for (const auto& provider : available_providers) {
        if (provider == "TensorrtExecutionProvider") {
            try {
                OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;
                Ort::GetApi().CreateTensorRTProviderOptions(&tensorrt_options);

                // Configure TensorRT settings
                // device_id: GPU to use (0 = first GPU)
                // trt_max_workspace_size: Memory limit for TensorRT optimization (2GB)
                std::vector<const char*> keys = { "device_id", "trt_max_workspace_size" };
                std::vector<const char*> values = { "0", "2147483648" }; // 2GB workspace

                Ort::GetApi().UpdateTensorRTProviderOptions(tensorrt_options, keys.data(), values.data(), 2);

                session_options_.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
                Ort::GetApi().ReleaseTensorRTProviderOptions(tensorrt_options);

                gpu_enabled = true;
                using_gpu_ = true;
                std::cout << "[ONNX] Using TensorRT ExecutionProvider (GPU)" << std::endl;
                break;
            }
            catch (const Ort::Exception& e) {
                // TensorRT may fail if driver version is incompatible or CUDA libraries are missing
                std::cerr << "[ONNX] TensorRT failed: " << e.what() << std::endl;
            }
        }
    }

    // Priority 2: CUDA ExecutionProvider
    // Standard NVIDIA GPU acceleration (cuda+cudnn)
    // More reliable fallback if TensorRT fails or is unavailable
    if (!gpu_enabled) {
        for (const auto& provider : available_providers) {
            if (provider == "CUDAExecutionProvider") {
                try {
                    OrtCUDAProviderOptions cuda_options;
                    cuda_options.device_id = 0;
                    cuda_options.arena_extend_strategy = 0; // Default memory strategy

                    // Use exhaustive search for best convolution algorithm
                    // Increases first-run time but improves inference speed
                    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                    cuda_options.do_copy_in_default_stream = 1;

                    session_options_.AppendExecutionProvider_CUDA(cuda_options);
                    gpu_enabled = true;
                    using_gpu_ = true;
                    std::cout << "[ONNX] Using CUDA ExecutionProvider (GPU)" << std::endl;
                    break;
                }
                catch (const Ort::Exception& e) {
                    // CUDA may fail if CUDA toolkit or cuDNN is not properly installed
                    std::cerr << "[ONNX] CUDA failed: " << e.what() << std::endl;
                }
            }
        }
    }

    // Priority 3: CPU ExecutionProvider (Fallback)
    // Used when no GPU is available or GPU providers failed to initialize
    // Still provides multi-threaded execution for reasonable performance
    if (!gpu_enabled) {
        std::cout << "[ONNX] Using CPU ExecutionProvider" << std::endl;
        using_gpu_ = false;
    }

    // Configure CPU execution settings (always enabled as fallback for GPU)
    // Even when GPU is primary, some operations may fall back to CPU
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
