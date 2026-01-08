#include "yolo_pose_postprocess.hpp"

void yolo_pose_postprocess(
    const float* output_data,
    const std::vector<int64_t>& output_shape,
    const LetterboxParams& params,
    std::vector<Person>& persons,
    float CONF_THRESHOLD,
    float NMS_THRESHOLD)
{
    // Validate output shape: must be 3D tensor with batch size 1
    // Shape examples: [1, 56, 8400] or [1, 8400, 56]
    if (output_shape.size() != 3 || output_shape[0] != 1) {
        throw std::runtime_error("Unexpected output shape");
    }

    int64_t d1 = output_shape[1];
    int64_t d2 = output_shape[2];

    int rows = 0, cols = 0;

    // Handle both layouts: [1, 56, N] or [1, N, 56]
    if (d1 == 56) {
        rows = static_cast<int>(d1);
        cols = static_cast<int>(d2);
    }
    else if (d2 == 56) {
        rows = static_cast<int>(d2);
        cols = static_cast<int>(d1);
    }
    else {
        throw std::runtime_error("Unexpected pose dims");
    }

    // Create a temporary matrix view of the raw data
    cv::Mat output(rows, cols, CV_32F, const_cast<float*>(output_data));

    // Transpose so that each row corresponds to one detection:
    // after transpose: [num_proposals, 56]
    cv::Mat output_transposed;
    cv::transpose(output, output_transposed);
    float* data = reinterpret_cast<float*>(output_transposed.data);

    std::vector<cv::Rect > boxes;
    std::vector<float>    confidences;
    std::vector<int>      class_ids;
    std::vector<int>      indices; // map NMS index -> original row index

    boxes.reserve(output_transposed.rows);
    confidences.reserve(output_transposed.rows);
    class_ids.reserve(output_transposed.rows);

    // Iterate over all proposals.
    for (int i = 0; i < output_transposed.rows; ++i) {
        float* row = data + i * output_transposed.cols;

        // YOLO pose layout:
        // 0: x_center, 1: y_center, 2: w, 3: h, 4: objectness score,
        // 5..: keypoints (17 * 3 values)
        float conf = row[4];
        if (conf < CONF_THRESHOLD) continue;

        // Extract bounding box in letterbox coordinate space
        float x_center = row[0];
        float y_center = row[1];
        float w_box = row[2];
        float h_box = row[3];

        // Convert from letterbox coordinates back to original image space
        int x = static_cast<int>(((x_center - w_box / 2.f) - params.pad_w) / params.scale);
        int y = static_cast<int>(((y_center - h_box / 2.f) - params.pad_h) / params.scale);
        int w = static_cast<int>(w_box / params.scale);
        int h = static_cast<int>(h_box / params.scale);

        boxes.emplace_back(x, y, w, h);
        confidences.push_back(conf);
        class_ids.push_back(i); // Store original row index for keypoint extraction
    }

    // Apply Non-Maximum Suppression to filter overlapping boxes
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    persons.clear();
    persons.reserve(indices.size());

    // Second pass: extract keypoints for NMS-filtered detections
    for (int idx : indices) {
        Person person;
        person.bbox = boxes[idx];
        person.confidence = confidences[idx];

        int original_idx = class_ids[idx];
        float* row = data + original_idx * output_transposed.cols;

        person.keypoints.clear();
        person.keypoints.reserve(17); // COCO-style 17 keypoints

        // Extract 17 keypoints (COCO pose format)
        // Keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
        for (int kp_idx = 0; kp_idx < 17; ++kp_idx) {
            // Keypoint data starts at index 5, with 3 values per keypoint
            int base = 5 + kp_idx * 3;

            float kp_x = row[base + 0];
            float kp_y = row[base + 1];
            float kp_conf = row[base + 2];

            // Transform keypoint from letterbox space back to original image space
            // Apply same inverse transformation as bounding box
            float x = (kp_x - params.pad_w) / params.scale;
            float y = (kp_y - params.pad_h) / params.scale;

            KeyPoint kp;
            kp.point = cv::Point2f(x, y);
            kp.confidence = kp_conf;

            person.keypoints.push_back(kp);
        }

        persons.push_back(std::move(person));
    }
}

void yolo_pose_postprocess_batch(
    const float* output_data,
    const std::vector<int64_t>& output_shape,
    const std::vector<LetterboxParams>& params,
    std::vector<std::vector<Person>>& batch_persons,
    float CONF_THRESHOLD,
    float NMS_THRESHOLD)
{
    // Validate batch output shape: must be 3D tensor
    // Expected shapes: [batch_size, 56, num_proposals] or [batch_size, num_proposals, 56]
    if (output_shape.size() != 3) {
        throw std::runtime_error("Unexpected batch output shape");
    }

    int batch_size = static_cast<int>(output_shape[0]);
    int64_t d1 = output_shape[1];
    int64_t d2 = output_shape[2];

    // Determine dimensions (handle both possible layouts)
    int rows = 0, cols = 0;
    if (d1 == 56) {
        rows = static_cast<int>(d1);
        cols = static_cast<int>(d2);
    }
    else if (d2 == 56) {
        rows = static_cast<int>(d2);
        cols = static_cast<int>(d1);
    }
    else {
        throw std::runtime_error("Unexpected pose dimensions in batch");
    }

    // Calculate size of output for a single frame
    // Batch tensor layout: [frame0_data][frame1_data][frame2_data]...
    size_t single_output_size = rows * cols;
    batch_persons.resize(batch_size);

    // Process each frame independently
    // We reuse the single-frame postprocessing function for each frame in the batch
    for (int b = 0; b < batch_size; ++b) {
        // Calculate pointer to this frame's data within the batch tensor
        const float* batch_data = output_data + b * single_output_size;

        // Create single-frame shape for compatibility with existing function
        // The single-frame function expects [1, 56, N] or [1, N, 56]
        std::vector<int64_t> single_shape = { 1, d1, d2 };

        // Process this frame using single-frame postprocessing
        // Results are stored in batch_persons[b]
        yolo_pose_postprocess(batch_data, single_shape, params[b],
            batch_persons[b], CONF_THRESHOLD, NMS_THRESHOLD);
    }
}