#include "preprocess_letterbox.hpp"
#include <string>
#include <iostream>

// Conditional compilation: check if OpenCV was built with CUDA support
#ifdef __has_include
#  if __has_include(<opencv2/cudawarping.hpp>)
#    include <opencv2/cudawarping.hpp>
#    include <opencv2/cudaimgproc.hpp>
#    include <opencv2/cudaarithm.hpp>
#    define OPENCV_CUDA_AVAILABLE
#  endif
#endif

namespace {

    /// CPU-based letterbox preprocessing
    /**
     * Performs resize and padding on CPU using standard OpenCV functions.
     * Used as fallback when GPU is unavailable or disabled.
     */
    static cv::Mat letterbox_cpu(const cv::Mat& image,
        int target_w, int target_h,
        float& scale, int& pad_w, int& pad_h)
    {
        int img_w = image.cols;
        int img_h = image.rows;

        // Calculate scale fit image within target size while preserving aspect ratio
        scale = std::min(static_cast<float>(target_w) / img_w,
            static_cast<float>(target_h) / img_h);

        int new_w = static_cast<int>(img_w * scale);
        int new_h = static_cast<int>(img_h * scale);

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        // Calculate padding to center the resized image
        pad_w = (target_w - new_w) / 2;
        pad_h = (target_h - new_h) / 2;

        // Add gray padding to reach target dimensions
        // Gray value (114,114,114) is YOLO standard background color
        cv::Mat padded;
        cv::copyMakeBorder(resized, padded,
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        return padded;
    }

#ifdef OPENCV_CUDA_AVAILABLE
    /// GPU-accelerated letterbox preprocessing
    /**
     * Performs resize operation on GPU for maximum performance.
     * Padding is still done on CPU
     */
    static cv::cuda::GpuMat letterbox_gpu(const cv::Mat& image,
        int target_w, int target_h,
        float& scale, int& pad_w, int& pad_h)
    {
        int img_w = image.cols;
        int img_h = image.rows;

        // Calculate scale factor (same as CPU version)
        scale = std::min(static_cast<float>(target_w) / img_w,
            static_cast<float>(target_h) / img_h);

        int new_w = static_cast<int>(img_w * scale);
        int new_h = static_cast<int>(img_h * scale);

        // Upload image to GPU memory
        cv::cuda::GpuMat gpu_image;
        gpu_image.upload(image);

        // Perform resize on GPU (most computationally intensive operation)
        cv::cuda::GpuMat gpu_resized;
        cv::cuda::resize(gpu_image, gpu_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        // Perform padding
        pad_w = (target_w - new_w) / 2;
        pad_h = (target_h - new_h) / 2;

        cv::cuda::GpuMat padded;
        cv::cuda::copyMakeBorder(gpu_resized, padded,
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        return padded;
    }
    static void blobFromImage_gpu_nchw(
        const std::vector<cv::cuda::GpuMat>& batch_frames,
        cv::cuda::GpuMat& output_buffer,
        float scale_factor = 1.0f / 255.0f)
    {
        if (batch_frames.empty()) return;

        int batch_size = batch_frames.size();
        int height = batch_frames[0].rows;
        int width = batch_frames[0].cols;
        int channels = 3;

        // Выделяем память под ВЕСЬ тензор (как 2D матрица для OpenCV)
        // Rows = Batch * Channels (например 8*3=24 строки)
        // Cols = Height * Width (плоская картинка)
        output_buffer.create(batch_size * channels, height * width, CV_32F);

        std::vector<cv::cuda::GpuMat> ch_split;

        for (int i = 0; i < batch_size; ++i) {
            // Convert to Float + Normalize
            cv::cuda::GpuMat float_frame;
            batch_frames[i].convertTo(float_frame, CV_32F, scale_factor);

            // Split channels (B, G, R)
            cv::cuda::split(float_frame, ch_split);

            // Копируем каналы в правильные места (HWC -> NCHW)
            // YOLO хочет RGB, OpenCV дает BGR.
            // Поэтому берем ch_split[2] (R), [1] (G), [0] (B)

            int offset = i * channels;

            // Red <- BGR[2]
            ch_split[2].reshape(1, 1).copyTo(output_buffer.row(offset + 0));
            // Green <- BGR[1]
            ch_split[1].reshape(1, 1).copyTo(output_buffer.row(offset + 1));
            // Blue <- BGR[0]
            ch_split[0].reshape(1, 1).copyTo(output_buffer.row(offset + 2));
        }
    }
#endif

} // namespace



PreprocessResult preprocess_letterbox(
    const cv::Mat& frame,
    int target_w,
    int target_h,
    bool use_gpu)
{
    PreprocessResult res;
    res.params.input_w = target_w;
    res.params.input_h = target_h;
    res.params.orig_w = frame.cols;
    res.params.orig_h = frame.rows;

    cv::Mat padded;
    bool gpu_used = false;

#ifdef OPENCV_CUDA_AVAILABLE
    // Attempt GPU acceleration if requested and available
    if (use_gpu) {
        try {
            // Verify that at least one CUDA-capable GPU is present
            int gpu_count = cv::cuda::getCudaEnabledDeviceCount();
            if (gpu_count > 0) {
                cv::cuda::GpuMat gpu_padded = letterbox_gpu(
                    frame, target_w, target_h,
                    res.params.scale, res.params.pad_w, res.params.pad_h
                );
                gpu_padded.download(padded);
                gpu_used = true;
            }
        }
        catch (const cv::Exception& e) {
            // Gracefully fall back to CPU if GPU processing fails
            // Common failures: out of memory, driver issues, CUDA context errors
            std::cerr << "[Preprocess] GPU failed: " << e.what()
                << ", falling back to CPU" << std::endl;
        }
    }
#else
    // Warn user once if GPU was requested but OpenCV lacks CUDA support
    // This helps diagnose configuration issues without spamming console
    static bool warned = false;
    if (use_gpu && !warned) {
        std::cerr << "[Preprocess] OpenCV CUDA not available, using CPU" << std::endl;
        warned = true;
    }
#endif

    // CPU fallback path
    // Used when: GPU disabled, GPU unavailable, or GPU processing failed
    if (!gpu_used) {
        padded = letterbox_cpu(
            frame, target_w, target_h,
            res.params.scale, res.params.pad_w, res.params.pad_h
        );
    }

    // Convert BGR image to normalized NCHW float tensor
    // - NCHW format: [batch, channels, height, width] required by ONNX
    // - Normalization: pixel values scaled from [0,255] to [0,1]
    // - swapRB=true: converts BGR to RGB (YOLO expects RGB input)
    cv::Mat blob;
    cv::dnn::blobFromImage(
        padded, blob, 1.0 / 255.0, cv::Size(),
        cv::Scalar(0, 0, 0), true, false, CV_32F
    );

    // Prepare tensor data and shape for ONNX Runtime
    res.input_shape = { 1, 3, target_h, target_w };
    size_t total = static_cast<size_t>(1) * 3 * target_h * target_w;
    res.input_tensor.resize(total);

    // Copy blob data to output tensor
    CV_Assert(blob.total() == total);
    std::memcpy(res.input_tensor.data(), blob.ptr<float>(),
        total * sizeof(float));

    return res;
}

BatchPreprocessResult preprocess_letterbox_batch(
    const std::vector<cv::Mat>& frames,
    int target_w,
    int target_h,
    bool use_gpu)
{
    BatchPreprocessResult res;
    int batch_size = frames.size();

    // Allocate batch tensor: [batch_size, 3, height, width]
    // All frames are concatenated along batch dimension for parallel inference
    res.input_shape = { batch_size, 3, target_h, target_w };
    size_t single_frame_size = 3 * target_h * target_w;
    size_t total_size = single_frame_size * batch_size;
    res.input_tensor.resize(total_size);
    res.params.reserve(batch_size);

    // Process each frame independently
    // Each frame gets its own letterbox parameters since original sizes may differ
    for (int i = 0; i < batch_size; ++i) {
        LetterboxParams params;
        params.input_w = target_w;
        params.input_h = target_h;
        params.orig_w = frames[i].cols;
        params.orig_h = frames[i].rows;

        cv::Mat padded;
        bool gpu_used = false;

#ifdef OPENCV_CUDA_AVAILABLE
        // GPU acceleration path for each frame
        if (use_gpu) {
            try {
                int gpu_count = cv::cuda::getCudaEnabledDeviceCount();
                if (gpu_count > 0) {

                    // === [NEW] Full GPU Pipeline ===
                    std::vector<cv::cuda::GpuMat> processed_frames_gpu;
                    processed_frames_gpu.reserve(batch_size);

                    for (int i = 0; i < batch_size; ++i) {
                        LetterboxParams params;
                        params.input_w = target_w;
                        params.input_h = target_h;
                        params.orig_w = frames[i].cols;
                        params.orig_h = frames[i].rows;

                        // Вызываем обновленную letterbox_gpu (возвращает GpuMat)
                        // БЕЗ download!
                        cv::cuda::GpuMat gpu_padded = letterbox_gpu(
                            frames[i], target_w, target_h,
                            params.scale, params.pad_w, params.pad_h
                        );

                        processed_frames_gpu.push_back(gpu_padded);
                        res.params.push_back(params);
                    }

                    // Превращаем вектор картинок в NCHW тензор (на GPU)
                    blobFromImage_gpu_nchw(processed_frames_gpu, res.gpu_nchw_buffer);

                    res.use_gpu_buffer = true; // Сигнал для OnnxEngine!
                    res.input_shape = { batch_size, 3, target_h, target_w };

                    return res; // <--- УСПЕХ! Возвращаем результат, пропуская CPU код
                }
            }
            catch (const cv::Exception& e) {
                // Silent fallback to CPU for this frame
                // Error already reported in single-frame function
            }
        }
#endif

        if (!gpu_used) {
            // CPU fallback path for this frame
            int img_w = frames[i].cols;
            int img_h = frames[i].rows;
            float scale = std::min((float)target_w / img_w, (float)target_h / img_h);
            params.scale = scale;

            int new_w = (int)(img_w * scale);
            int new_h = (int)(img_h * scale);

            // CPU resize
            cv::Mat resized;
            cv::resize(frames[i], resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

            // CPU padding
            params.pad_w = (target_w - new_w) / 2;
            params.pad_h = (target_h - new_h) / 2;

            cv::copyMakeBorder(resized, padded,
                params.pad_h, target_h - new_h - params.pad_h,
                params.pad_w, target_w - new_w - params.pad_w,
                cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        }

        // Convert to normalized NCHW tensor (same as single-frame version)
        cv::Mat blob;
        cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0, cv::Size(),
            cv::Scalar(0, 0, 0), true, false, CV_32F);

        // Copy frame data into batch tensor at appropriate offset
        // Frames are stored sequentially in memory: [frame0][frame1][frame2]...
        std::memcpy(res.input_tensor.data() + i * single_frame_size,
            blob.ptr<float>(),
            single_frame_size * sizeof(float));

        res.params.push_back(params);
    }

    return res;
}
