#pragma once

#include "utility/pimpl.hpp"
#include "utility/robot/armor.hpp"

#include <array>
#include <cstddef>
#include <expected>
#include <optional>
#include <span>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <yaml-cpp/yaml.h>

namespace rmcs::kernel {

class PoseEstimator {
    RMCS_PIMPL_DEFINITION(PoseEstimator)

public:
    struct Addition {
        Armor3ds origin;
        std::vector<cv::Rect2i> areas;
        Point2d center_2d;
        Point3d center_3d;
        Lightbar2ds predicted_near;
        Lightbar2ds predicted_away;
        Lightbar2ds detected_2d;
        Lightbar3ds detected_3d;

        /// 科技核心总装的 PnP 结果（Odom 坐标系），以及其平均重投影误差（像素）
        std::optional<Transform> tech_core;
        double tech_core_reprojection_error = 0.0;
    };

    auto initialize(const YAML::Node&) noexcept -> std::expected<void, std::string>;

    auto configure_camera(std::array<double, 9>, std::array<double, 5>) -> void;
    auto update_camera_transform(const Transform& transform) -> void;

    auto estimate_armor(const std::vector<Armor2d>&) const -> Armor3ds;
    auto estimate_armor(const std::vector<Armor2d>&, const cv::Mat&) const -> Armor3ds;

    /// 使用 HRNet 关键点求解科技核心总装位姿（Odom 坐标系）
    auto estimate_tech_core(std::span<const std::size_t> indices,
        std::span<const cv::Point2f> points) -> std::optional<Transform>;

    auto make_point2d(const Point3d& point_odom) const -> std::optional<Point2d>;

    auto addition() -> const Addition&;
};

}
