#pragma once

#include "utility/math/linear.hpp"

#include <array>

namespace rmcs {

/// @NOTE: 科技核心总装（exchange_12pt schema）在目标系下的 12 个关键点，单位米。
///        顺序必须与 detector::HrnetKeypoint::kKeypointNames 完全一致：
///        TL, TR, BL, BR, ring, light_BR, light_TR, shell_R, shell_M, shell_L, light_TL, light_BL
///        其中 0-4 属于 pillar，5-11 属于 exchange。pillar 可能沿轴相对 exchange 滑动，
///        因此解算时依赖 RANSAC 剔除被滑动污染的点。
inline constexpr std::array<Point3d, 12> kTechCoreObjectPoints {
    Point3d { -0.100000, -0.040000, +0.040000 }, // TL
    Point3d { -0.100000, +0.040000, +0.040000 }, // TR
    Point3d { -0.100000, -0.040000, -0.040000 }, // BL
    Point3d { -0.100000, +0.040000, -0.040000 }, // BR
    Point3d { +0.000000, +0.000000, +0.000000 }, // ring
    Point3d { -0.112000, +0.114870, +0.035953 }, // light_BR
    Point3d { -0.112000, +0.114870, +0.167312 }, // light_TR
    Point3d { -0.112000, +0.084871, +0.276449 }, // shell_R
    Point3d { -0.112000, +0.000000, +0.325449 }, // shell_M
    Point3d { -0.112000, -0.084871, +0.276449 }, // shell_L
    Point3d { -0.112000, -0.114870, +0.167312 }, // light_TL
    Point3d { -0.112000, -0.114870, +0.035953 }, // light_BL
};

}
