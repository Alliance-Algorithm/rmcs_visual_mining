#pragma once

#include "utility/math/camera.hpp"

#include <cstddef>
#include <span>

#include <opencv2/core/types.hpp>

namespace rmcs::util {

/// @brief 照搬参考仓库 PnPEstimator 的关键点 PnP 管线：
///        RANSAC(SQPnP) -> SQPnP 多解 -> 相机后方点剪枝 -> LM 精化 -> 最小平均重投影误差选优。
struct KeypointPnpSolution {
    struct Input {
        CameraFeature camera;

        /// 关键点索引与图像点一一对应，索引用于在 object_points 中查表
        std::span<const std::size_t> indices;
        std::span<const cv::Point2f> points;
        std::span<const Point3d> object_points;

        bool use_ransac                     = true;
        double ransac_reprojection_error_px = 3.0;
        double ransac_confidence            = 0.99;
        int ransac_iterations               = 100;
    } input;

    struct Result {
        /// solve() 内部已经通过 CameraFeature 外参转换到 Odom 坐标系
        Transform transform;
        double reprojection_error = 0.0;
    } result;

    auto solve() -> bool;
};

}
