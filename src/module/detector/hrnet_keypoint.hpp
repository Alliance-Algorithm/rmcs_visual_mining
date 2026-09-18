#pragma once

#include "module/detector/yolo_detection.hpp"
#include "utility/pimpl.hpp"

#include <array>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

#include <opencv2/core/types.hpp>
#include <yaml-cpp/yaml.h>

namespace rmcs::detector {

class HrnetKeypoint {
    RMCS_PIMPL_DEFINITION(HrnetKeypoint)

public:
    static constexpr std::array<std::string_view, 12> kKeypointNames = {
        "TL",
        "TR",
        "BL",
        "BR",
        "ring",
        "light_BR",
        "light_TR",
        "shell_R",
        "shell_M",
        "shell_L",
        "light_TL",
        "light_BL",
    };

    struct Keypoint {
        cv::Point2f point;
        float score;
        std::size_t index;
    };
    struct KeypointResult {
        cv::Rect2i bbox;
        int class_id = -1;
        std::vector<Keypoint> keypoints;
    };
    using KeypointResults = std::vector<KeypointResult>;

    auto initialize(const YAML::Node&) noexcept -> std::expected<void, std::string>;
    auto sync_detect(const cv::Mat& image,
        std::span<const YoloDetection::Detection> detections) noexcept -> KeypointResults;
};

}
