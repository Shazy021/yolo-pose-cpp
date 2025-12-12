#include"preprocess_letterbox.hpp"
#include<string>

namespace {

	/// Apply letterbox resize + padding to target dimensions.
	/**
	 * Preserves aspect ratio, pads with gray (114,114,114) to square target size.
	 *
	 * \param image    Input image.
	 * \param target_w Target width.
	 * \param target_h Target height.
	 * \param scale    Output: computed resize scale factor.
	 * \param pad_w    Output: horizontal padding (each side).
	 * \param pad_h    Output: vertical padding (each side).
	 * \return Padded image matching target dimensions.
	 */
	static cv::Mat letterbox(const cv::Mat& image,
			int target_w, int target_h,
			float& scale, int& pad_w, int& pad_h)
	{
		int img_w = image.cols;
		int img_h = image.rows;

		scale = std::min(static_cast<float>(target_w) / img_w,
			static_cast<float>(target_h) / img_h);

		int new_w = static_cast<int>(img_w * scale);
		int new_h = static_cast<int>(img_h * scale);

		cv::Mat resized;
		cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

		pad_w = (target_w - new_w) / 2;
		pad_h = (target_h - new_h) / 2;

		cv::Mat padded;
		cv::copyMakeBorder(resized, padded,
			pad_h, target_h - new_h - pad_h,
			pad_w, target_w - new_w - pad_w,
			cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114)
		);

		return padded;
	}

} // namespace

PreprocessResult preprocess_letterbox(
	const cv::Mat& frame,
	int target_w,
	int target_h)
{
	PreprocessResult res;
	res.params.input_w = target_w;
	res.params.input_h = target_h;
	res.params.orig_w = frame.cols;
	res.params.orig_h = frame.rows;

	// Apply letterbox resize + padding.
	cv::Mat padded = letterbox(
		frame, target_w, target_h,
		res.params.scale, res.params.pad_w, res.params.pad_h
	);

	// Convert to NCHW float tensor [0,1] range.
	cv::Mat blob;
	cv::dnn::blobFromImage(
		padded, blob, 1.0 / 255.0, cv::Size(),
		cv::Scalar(0, 0, 0), true, false, CV_32F
	);
	
	// Prepare tensor data and shape for ONNX Runtime.
	res.input_shape = { 1, 3, target_h, target_w };
	size_t total = static_cast<size_t>(1) * 3 * target_h * target_w;
	res.input_tensor.resize(total);

	CV_Assert(blob.total() == total);
	std::memcpy(res.input_tensor.data(), blob.ptr<float>(),
		total * sizeof(float));

	return res;
}
