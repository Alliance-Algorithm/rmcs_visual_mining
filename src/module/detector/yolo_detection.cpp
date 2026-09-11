#include "yolo_detection.hpp"

#include "module/detector/models/tech_core_yolo.hpp"
#include "utility/model/common_model.hpp"
#include "utility/serializable.hpp"

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/exception.hpp>

using namespace rmcs::detector;

struct YoloDetection::Impl {
    struct Config : util::Serializable {
        bool enable = false;

        std::string model_location;
        std::string infer_device;

        int input_rows;
        int input_cols;

        float min_confidence;
        float score_threshold;
        float nms_threshold;

        constexpr static std::tuple metas {
            // clang-format off
            &Config::enable,               "enable",
            &Config::model_location,       "model_location",
            &Config::infer_device,         "infer_device",
            &Config::input_rows,           "input_rows",
            &Config::input_cols,           "input_cols",
            &Config::min_confidence,       "min_confidence",
            &Config::score_threshold,      "score_threshold",
            &Config::nms_threshold,        "nms_threshold",
            // clang-format on
        };
    } config;

    bool initialized = false;

    ov::Core openvino_core;
    ov::CompiledModel openvino_model;

    TensorLayout input_layout = TensorLayout::from<"NHWC">();
    Dimensions input_dimensions { .W = 640, .H = 640 };

    float adapt_scaling = 1.f;

    auto initialize(const YAML::Node& yaml) noexcept -> std::expected<void, std::string> {
        auto result = config.serialize(yaml);
        if (!result.has_value()) {
            return std::unexpected { result.error() };
        }

        if (!config.enable) return { };

        try {
            auto model = TechCoreYolo { };
            if (!config.infer_device.empty()) {
                model.infer_device = config.infer_device;
            }
            if (config.input_cols > 0) {
                model.dimensions.W = config.input_cols;
            }
            if (config.input_rows > 0) {
                model.dimensions.H = config.input_rows;
            }

            openvino_model   = model.compile(openvino_core, config.model_location);
            input_layout     = model.input_layout;
            input_dimensions = model.dimensions;

            initialized = true;
            return { };
        } catch (const std::runtime_error& e) {
            return std::unexpected { std::string { "Failed to load yolo model | " } + e.what() };
        } catch (...) {
            return std::unexpected { "Failed to load yolo model caused by unknown exception" };
        }
    }

    auto sync_detect(const cv::Mat& origin_mat) noexcept -> Detections {
        if (!initialized) return { };
        if (origin_mat.empty()) return { };

        const auto rows = static_cast<int>(input_dimensions.H);
        const auto cols = static_cast<int>(input_dimensions.W);

        auto input_tensor = ov::Tensor {
            ov::element::u8,
            input_layout.shape(input_dimensions),
        };

        adapt_scaling = std::min(static_cast<float>(1. * cols / origin_mat.cols),
            static_cast<float>(1. * rows / origin_mat.rows));

        const auto scaled_w = static_cast<int>(1. * origin_mat.cols * adapt_scaling);
        const auto scaled_h = static_cast<int>(1. * origin_mat.rows * adapt_scaling);

        auto input_mat = cv::Mat { rows, cols, CV_8UC3, input_tensor.data() };
        input_mat.setTo(cv::Scalar::all(0));

        auto input_roi = cv::Rect2i { 0, 0, scaled_w, scaled_h };
        cv::resize(origin_mat, input_mat(input_roi), { scaled_w, scaled_h });

        auto request = openvino_model.create_infer_request();
        request.set_input_tensor(input_tensor);
        request.infer();

        using result_type    = TechCoreYolo::Result;
        using precision_type = result_type::precision_type;

        auto tensor = request.get_output_tensor();
        auto& shape = tensor.get_shape();

        const auto det_rows = static_cast<std::size_t>(shape.at(1));
        const auto det_cols = static_cast<std::size_t>(shape.at(2));
        if (det_cols != result_type::length()) {
            return { };
        }

        auto parsed = std::vector<result_type> { };
        auto scores = std::vector<float> { };
        auto boxes  = std::vector<cv::Rect> { };

        const auto* data = tensor.data<precision_type>();
        for (std::size_t row = 0; row < det_rows; row++) {
            auto line = result_type { };
            line.unsafe_from(std::span { data + row * det_cols, det_cols });

            const auto confidence = line.confidence();
            if (confidence > config.min_confidence) {
                parsed.push_back(line);
                scores.push_back(confidence);
                boxes.push_back(line.xyxy());
            }
        }

        auto kept_points = std::vector<int> { };
        cv::dnn::NMSBoxes(boxes, scores, config.score_threshold, config.nms_threshold, kept_points);

        const auto scale = 1.f / adapt_scaling;
        auto result_boxes = Detections { };
        result_boxes.reserve(kept_points.size());

        for (auto idx : kept_points) {
            const auto& line = parsed[static_cast<std::size_t>(idx)];
            auto rect        = line.xyxy();
            result_boxes.push_back(Detection {
                .rect = cv::Rect2i {
                    static_cast<int>(rect.x * scale),
                    static_cast<int>(rect.y * scale),
                    static_cast<int>(rect.width * scale),
                    static_cast<int>(rect.height * scale),
                },
                .confidence = line.confidence(),
                .class_id   = static_cast<int>(line.class_id()),
            });
        }

        return result_boxes;
    }
};

auto YoloDetection::initialize(const YAML::Node& yaml) noexcept
    -> std::expected<void, std::string> {
    return pimpl->initialize(yaml);
}

auto YoloDetection::sync_detect(const cv::Mat& image) noexcept -> Detections {
    return pimpl->sync_detect(image);
}

YoloDetection::YoloDetection() noexcept
    : pimpl { std::make_unique<Impl>() } { }

YoloDetection::~YoloDetection() noexcept = default;
