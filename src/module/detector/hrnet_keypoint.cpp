#include "hrnet_keypoint.hpp"

#include "module/detector/models/tech_core_hrnet.hpp"
#include "utility/model/common_model.hpp"
#include "utility/serializable.hpp"

#include <array>
#include <cstring>
#include <utility>
#include <vector>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/exception.hpp>

using namespace rmcs::detector;

namespace {

auto sign(float value) noexcept -> float {
    return static_cast<float>((value > 0.f) - (value < 0.f));
}

auto third_point(const cv::Point2f& a, const cv::Point2f& b) noexcept -> cv::Point2f {
    const auto direction = a - b;
    return b + cv::Point2f { -direction.y, direction.x };
}

auto bbox_to_square_affine(const cv::Rect2i& bbox, const cv::Size2i& input_size,
    float padding) noexcept -> std::pair<cv::Mat, cv::Mat> {
    const auto x1 = static_cast<float>(bbox.x);
    const auto y1 = static_cast<float>(bbox.y);
    const auto x2 = static_cast<float>(bbox.x + bbox.width);
    const auto y2 = static_cast<float>(bbox.y + bbox.height);

    auto scale = cv::Point2f { std::max(x2 - x1, 1.0f), std::max(y2 - y1, 1.0f) };
    scale *= padding;

    const auto input_w = static_cast<float>(input_size.width);
    const auto input_h = static_cast<float>(input_size.height);
    const auto aspect  = input_w / input_h;
    if (scale.x > scale.y * aspect) {
        scale.y = scale.x / aspect;
    } else {
        scale.x = scale.y * aspect;
    }

    const auto center = cv::Point2f { (x1 + x2) * 0.5f, (y1 + y2) * 0.5f };

    auto src = std::array<cv::Point2f, 3> {
        center,
        center + cv::Point2f { -0.5f * scale.x, 0.0f },
        cv::Point2f { },
    };
    src[2] = third_point(src[0], src[1]);

    const auto dst_center = cv::Point2f { input_w * 0.5f, input_h * 0.5f };
    auto dst              = std::array<cv::Point2f, 3> {
        dst_center,
        dst_center + cv::Point2f { -0.5f * input_w, 0.0f },
        cv::Point2f { },
    };
    dst[2] = third_point(dst[0], dst[1]);

    return {
        cv::getAffineTransform(src.data(), dst.data()),
        cv::getAffineTransform(dst.data(), src.data()),
    };
}

auto apply_affine(const cv::Mat& affine, cv::Point2f point) noexcept -> cv::Point2f {
    const auto* m = affine.ptr<double>();
    return {
        static_cast<float>(m[0] * point.x + m[1] * point.y + m[2]),
        static_cast<float>(m[3] * point.x + m[4] * point.y + m[5]),
    };
}

} // namespace

struct HrnetKeypoint::Impl {
    struct Config : util::Serializable {
        bool enable = false;

        std::string model_location;
        std::string infer_device;

        int input_rows;
        int input_cols;

        std::array<float, 3> mean;
        std::array<float, 3> std_dev;

        float bbox_padding;
        int bbox_class_id;

        float visibility_score_threshold;
        int min_visible_keypoints;

        constexpr static std::tuple metas {
            // clang-format off
            &Config::enable,                     "enable",
            &Config::model_location,             "model_location",
            &Config::infer_device,               "infer_device",
            &Config::input_rows,                 "input_rows",
            &Config::input_cols,                 "input_cols",
            &Config::mean,                       "mean",
            &Config::std_dev,                    "std",
            &Config::bbox_padding,               "bbox_padding",
            &Config::bbox_class_id,              "bbox_class_id",
            &Config::visibility_score_threshold, "visibility_score_threshold",
            &Config::min_visible_keypoints,      "min_visible_keypoints",
            // clang-format on
        };
    } config;

    bool initialized = false;

    ov::Core openvino_core;
    ov::CompiledModel openvino_model;

    TensorLayout input_layout = TensorLayout::from<"NCHW">();
    Dimensions input_dimensions { .W = 256, .H = 256 };

    auto initialize(const YAML::Node& yaml) noexcept -> std::expected<void, std::string> {
        auto result = config.serialize(yaml);
        if (!result.has_value()) {
            return std::unexpected { result.error() };
        }

        if (!config.enable) return { };

        try {
            auto model = TechCoreHrnet18 { };
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
            return std::unexpected { std::string { "Failed to load hrnet model | " } + e.what() };
        } catch (...) {
            return std::unexpected { "Failed to load hrnet model caused by unknown exception" };
        }
    }

    auto sync_detect(const cv::Mat& image,
        std::span<const YoloDetection::Detection> detections) noexcept -> KeypointResults {
        if (!initialized) return { };
        if (image.empty()) return { };

        const auto input_w = static_cast<int>(input_dimensions.W);
        const auto input_h = static_cast<int>(input_dimensions.H);

        auto results = KeypointResults { };
        results.reserve(detections.size());

        for (const auto& detection : detections) {
            if (config.bbox_class_id != 2 && detection.class_id != config.bbox_class_id) continue;

            const auto [affine, inverse] =
                bbox_to_square_affine(detection.rect, { input_w, input_h }, config.bbox_padding);

            cv::Mat crop;
            cv::warpAffine(image, crop, affine, { input_w, input_h }, cv::INTER_LINEAR,
                cv::BORDER_CONSTANT, cv::Scalar { 0 });

            cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
            crop.convertTo(crop, CV_32F);

            auto channels = std::vector<cv::Mat> { };
            cv::split(crop, channels);
            for (int channel = 0; channel < 3; ++channel) {
                channels[channel] =
                    (channels[channel] - config.mean[channel]) / config.std_dev[channel];
            }
            cv::merge(channels, crop);

            auto blob = cv::dnn::blobFromImage(crop, 1.0, { }, { }, false);

            auto input_tensor =
                ov::Tensor { ov::element::f32, input_layout.shape(input_dimensions) };
            std::memcpy(input_tensor.data(), blob.ptr(), blob.total() * sizeof(float));

            auto request = openvino_model.create_infer_request();
            request.set_input_tensor(input_tensor);
            request.infer();

            auto output_tensor = request.get_output_tensor();
            const auto* data   = output_tensor.data<float>();

            const auto heatmap_w = TechCoreHrnet18::kHeatmapWidth;
            const auto heatmap_h = TechCoreHrnet18::kHeatmapHeight;

            auto keypoints = std::vector<Keypoint> { };
            keypoints.reserve(TechCoreHrnet18::kKeypoints);

            for (int index = 0; index < TechCoreHrnet18::kKeypoints; ++index) {
                const auto* heatmap =
                    data + static_cast<std::size_t>(index) * heatmap_w * heatmap_h;

                auto max_index = 0;
                auto max_value = heatmap[0];
                for (int i = 1; i < heatmap_w * heatmap_h; ++i) {
                    if (heatmap[i] > max_value) {
                        max_value = heatmap[i];
                        max_index = i;
                    }
                }

                if (max_value < config.visibility_score_threshold) continue;

                const auto y = max_index / heatmap_w;
                const auto x = max_index % heatmap_w;

                auto px = static_cast<float>(x);
                auto py = static_cast<float>(y);
                if (1 <= x && x < heatmap_w - 1 && 1 <= y && y < heatmap_h - 1) {
                    px += sign(heatmap[y * heatmap_w + x + 1] - heatmap[y * heatmap_w + x - 1])
                        * 0.25f;
                    py += sign(heatmap[(y + 1) * heatmap_w + x] - heatmap[(y - 1) * heatmap_w + x])
                        * 0.25f;
                }

                const auto input_point = cv::Point2f {
                    (px + 0.5f) * static_cast<float>(input_w) / heatmap_w,
                    (py + 0.5f) * static_cast<float>(input_h) / heatmap_h,
                };

                keypoints.push_back(Keypoint {
                    .point = apply_affine(inverse, input_point),
                    .score = max_value,
                    .index = static_cast<std::size_t>(index),
                });
            }

            if (static_cast<int>(keypoints.size()) < config.min_visible_keypoints) continue;

            results.push_back(KeypointResult {
                .bbox      = detection.rect,
                .class_id  = detection.class_id,
                .keypoints = std::move(keypoints),
            });
        }

        return results;
    }
};

auto HrnetKeypoint::initialize(const YAML::Node& yaml) noexcept
    -> std::expected<void, std::string> {
    return pimpl->initialize(yaml);
}

auto HrnetKeypoint::sync_detect(const cv::Mat& image,
    std::span<const YoloDetection::Detection> detections) noexcept -> KeypointResults {
    return pimpl->sync_detect(image, detections);
}

HrnetKeypoint::HrnetKeypoint() noexcept
    : pimpl { std::make_unique<Impl>() } { }

HrnetKeypoint::~HrnetKeypoint() noexcept = default;
