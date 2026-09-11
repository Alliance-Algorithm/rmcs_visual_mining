#pragma once
#include "utility/model/common_model.hpp"

#include <array>

#include <openvino/runtime/compiled_model.hpp>

namespace rmcs {

struct TechCoreHrnet18 {
    static constexpr auto kLocation = "hrnet_heatmap_18_v1.xml";

    TensorLayout input_layout = TensorLayout::from<"NCHW">();
    Dimensions dimensions     = Dimensions { .C = 3, .W = 256, .H = 256 };
    std::string infer_device  = "AUTO";

    auto compile(ov::Core& core, std::string_view location) const -> ov::CompiledModel {
        auto raw = core.read_model(std::string { location });
        return core.compile_model(raw, infer_device, kRealTimePerformanceMode);
    }

    static constexpr int kKeypoints     = 12;
    static constexpr int kHeatmapWidth  = 128;
    static constexpr int kHeatmapHeight = 128;

    struct ResultData {
        using precision_type = float;

        struct Keypoint {
            precision_type x;
            precision_type y;
            precision_type score;
        };

        std::array<Keypoint, kKeypoints> keypoints;
    };
};

}
