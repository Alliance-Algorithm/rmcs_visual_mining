#pragma once

#include "utility/pimpl.hpp"

#include <array>
#include <expected>
#include <string_view>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <yaml-cpp/yaml.h>

namespace rmcs::detector {

class YoloDetection {
    RMCS_PIMPL_DEFINITION(YoloDetection)

public:
    struct Detection {
        cv::Rect2i rect;
        float confidence;
        int class_id;

        auto name() const noexcept -> std::string_view {
            constexpr std::array names { "pillar", "exchange" };
            if (class_id >= 0 && static_cast<std::size_t>(class_id) < names.size())
                return names[static_cast<std::size_t>(class_id)];
            return "unknown";
        }
    };
    using Detections = std::vector<Detection>;

    auto initialize(const YAML::Node&) noexcept -> std::expected<void, std::string>;
    auto sync_detect(const cv::Mat&) noexcept -> Detections;
};

}
