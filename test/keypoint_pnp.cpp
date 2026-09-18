#define OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "utility/math/conversion.hpp"
#include "utility/math/solve_pnp/keypoint_pnp.hpp"
#include "utility/robot/tech_core.hpp"

using namespace rmcs;
using namespace rmcs::util;

namespace {

constexpr auto kFx = 1.722231837421459e+03;
constexpr auto kFy = 1.724876404292754e+03;
constexpr auto kCx = 7.013056440882832e+02;
constexpr auto kCy = 5.645821718351237e+02;

auto make_camera() -> CameraFeature {
    auto camera          = CameraFeature { };
    camera.camera_matrix = { {
        { kFx, 0.0, kCx },
        { 0.0, kFy, kCy },
        { 0.0, 0.0, 1.0 },
    } };
    camera.distort_coeff = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    camera.orientation   = Orientation::kIdentity();
    camera.translation   = Translation::kZero();
    return camera;
}

auto object_points_as_cv() -> std::vector<cv::Point3f> {
    auto result = std::vector<cv::Point3f> { };
    result.reserve(kTechCoreObjectPoints.size());
    for (const auto& point : kTechCoreObjectPoints) {
        result.push_back(point.make<cv::Point3f>());
    }
    return result;
}

auto rotation_angle(const Orientation& lhs, const Orientation& rhs) -> double {
    const auto a = lhs.make<Eigen::Quaterniond>().normalized();
    const auto b = rhs.make<Eigen::Quaterniond>().normalized();
    return a.angularDistance(b);
}

} // namespace

TEST(KeypointPnp, RecoversKnownPoseWithoutRansac) {
    const auto camera_matrix = make_camera().intrinsic();
    const auto distortion    = make_camera().distortion();

    const auto rvec = cv::Vec3d { 0.05, -0.10, 0.20 };
    const auto tvec = cv::Vec3d { 0.02, -0.01, 0.80 };

    const auto object_points = object_points_as_cv();
    auto image_points        = std::vector<cv::Point2f> { };
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion, image_points);

    auto indices = std::vector<std::size_t>(kTechCoreObjectPoints.size());
    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;

    auto solver                    = KeypointPnpSolution { };
    solver.input.camera            = make_camera();
    solver.input.indices           = indices;
    solver.input.points            = image_points;
    solver.input.object_points     = kTechCoreObjectPoints;
    solver.input.use_ransac        = false;
    solver.input.ransac_iterations = 100;

    ASSERT_TRUE(solver.solve());

    auto rotation_mat = cv::Mat { };
    cv::Rodrigues(rvec, rotation_mat);
    auto rotation    = Eigen::Matrix3d { };
    auto translation = Eigen::Vector3d { };
    cv::cv2eigen(rotation_mat, rotation);
    cv::cv2eigen(tvec, translation);

    const auto expected_translation = Translation { opencv2ros_position(translation) };
    const auto expected_orientation =
        Orientation { Eigen::Quaterniond { opencv2ros_rotation(rotation) }.normalized() };

    EXPECT_LT(std::abs(solver.result.transform.translation.x - expected_translation.x), 1e-6);
    EXPECT_LT(std::abs(solver.result.transform.translation.y - expected_translation.y), 1e-6);
    EXPECT_LT(std::abs(solver.result.transform.translation.z - expected_translation.z), 1e-6);
    EXPECT_LT(rotation_angle(solver.result.transform.orientation, expected_orientation), 1e-6);
    EXPECT_LT(solver.result.reprojection_error, 1e-3);
}

TEST(KeypointPnp, RecoversExchangeWithRansacUnderPillarSlide) {
    const auto camera_matrix = make_camera().intrinsic();
    const auto distortion    = make_camera().distortion();

    const auto rvec = cv::Vec3d { -0.12, 0.08, 0.30 };
    const auto tvec = cv::Vec3d { -0.03, 0.02, 1.10 };

    const auto object_points = object_points_as_cv();

    // 模拟 pillar（index 0-4）相对 exchange 沿其轴线（object x）滑动一段距离
    auto observed = object_points;
    for (std::size_t i = 0; i < 5; ++i)
        observed[i].x -= 0.045f;

    auto image_points = std::vector<cv::Point2f> { };
    cv::projectPoints(observed, rvec, tvec, camera_matrix, distortion, image_points);

    auto indices = std::vector<std::size_t>(kTechCoreObjectPoints.size());
    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;

    auto solver                               = KeypointPnpSolution { };
    solver.input.camera                       = make_camera();
    solver.input.indices                      = indices;
    solver.input.points                       = image_points;
    solver.input.object_points                = kTechCoreObjectPoints;
    solver.input.use_ransac                   = true;
    solver.input.ransac_reprojection_error_px = 3.0;
    solver.input.ransac_confidence            = 0.99;
    solver.input.ransac_iterations            = 100;

    ASSERT_TRUE(solver.solve());

    auto rotation_mat = cv::Mat { };
    cv::Rodrigues(rvec, rotation_mat);
    auto rotation    = Eigen::Matrix3d { };
    auto translation = Eigen::Vector3d { };
    cv::cv2eigen(rotation_mat, rotation);
    cv::cv2eigen(tvec, translation);

    const auto expected_translation = Translation { opencv2ros_position(translation) };
    const auto expected_orientation =
        Orientation { Eigen::Quaterniond { opencv2ros_rotation(rotation) }.normalized() };

    EXPECT_LT(std::abs(solver.result.transform.translation.x - expected_translation.x), 1e-3);
    EXPECT_LT(std::abs(solver.result.transform.translation.y - expected_translation.y), 1e-3);
    EXPECT_LT(std::abs(solver.result.transform.translation.z - expected_translation.z), 1e-3);
    EXPECT_LT(rotation_angle(solver.result.transform.orientation, expected_orientation), 1e-3);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
