#pragma once
#include "utility/model/common_model.hpp"

#include <cstring>
#include <span>
#include <type_traits>

#include <opencv2/core/types.hpp>

#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/compiled_model.hpp>

namespace rmcs {

struct TechCoreYolo {
    static constexpr auto kLocation = "detect_yolo_v1.xml";

    TensorLayout input_layout = TensorLayout::from<"NHWC">();
    TensorLayout model_layout = TensorLayout::from<"NCHW">();
    Dimensions dimensions     = Dimensions { .W = 640, .H = 640 };
    std::string infer_device  = "AUTO";

    auto compile(ov::Core& core, std::string_view location) const -> ov::CompiledModel {
        auto raw = core.read_model(std::string { location });
        auto ppp = ov::preprocess::PrePostProcessor { raw };
        {
            auto& input = ppp.input();
            input.tensor()
                .set_element_type(ov::element::u8)
                .set_shape(input_layout.partial_shape(dimensions))
                .set_layout(input_layout.layout())
                .set_color_format(ov::preprocess::ColorFormat::BGR);
            input.preprocess()
                .convert_element_type(ov::element::f32)
                .convert_color(ov::preprocess::ColorFormat::RGB)
                .scale(255.0);
            input.model().set_layout(model_layout.layout());
        }
        return core.compile_model(ppp.build(), infer_device, kRealTimePerformanceMode);
    }

    struct ResultData {
        using precision_type = float;

        precision_type x1;
        precision_type y1;
        precision_type x2;
        precision_type y2;

        precision_type confidence;

        precision_type class_id;
    };

    struct Result {
        using precision_type = typename ResultData::precision_type;
        using rectangle_type = cv::Rect_<precision_type>;

        ResultData data;

        auto xyxy() const noexcept -> rectangle_type {
            return { data.x1, data.y1, data.x2 - data.x1, data.y2 - data.y1 };
        }
        auto confidence() const noexcept -> precision_type { return data.confidence; }
        auto class_id() const noexcept -> precision_type { return data.class_id; }

        auto unsafe_from(std::span<const precision_type> raw) noexcept -> void {
            static_assert(std::is_trivially_copyable_v<ResultData>);
            static_assert(std::is_standard_layout_v<ResultData>);
            if (raw.size() < length()) return;
            std::memcpy(&data, raw.data(), sizeof(ResultData));
        }

        constexpr static auto length() noexcept -> std::size_t {
            return sizeof(ResultData) / sizeof(precision_type);
        }
    };
};

}
