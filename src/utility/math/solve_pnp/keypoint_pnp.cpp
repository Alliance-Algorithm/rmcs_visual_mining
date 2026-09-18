#define OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT
#include "keypoint_pnp.hpp"

#include "utility/math/conversion.hpp"

#include <eigen3/Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <ranges>
#include <vector>

using namespace rmcs;
using namespace rmcs::util;

namespace {

auto min_camera_depth(const std::vector<cv::Point3f>& object_points, const cv::Vec3d& rvec,
    const cv::Vec3d& tvec) -> double {
    auto rotation_mat = cv::Mat { };
    cv::Rodrigues(rvec, rotation_mat);

    // camera = R * object + t，取 z 分量即 depth = R.row(2) · object + t.z
    const auto* rotation = rotation_mat.ptr<double>();

    auto depth = std::numeric_limits<double>::max();
    for (const auto& point : object_points) {
        const auto camera_z =
            rotation[6] * point.x + rotation[7] * point.y + rotation[8] * point.z + tvec[2];
        depth = std::min(depth, camera_z);
    }
    return depth;
}

auto mean_reprojection_error(const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points, const cv::Mat& camera_matrix,
    const cv::Mat& distort_coeff, const cv::Vec3d& rvec, const cv::Vec3d& tvec) -> double {
    auto projected = std::vector<cv::Point2f> { };
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, distort_coeff, projected);

    if (projected.size() != image_points.size() || projected.empty()) {
        return std::numeric_limits<double>::max();
    }

    auto accumulated = 0.0;
    for (const auto& [projected_point, image_point] : std::views::zip(projected, image_points)) {
        accumulated += std::hypot(static_cast<double>(projected_point.x - image_point.x),
            static_cast<double>(projected_point.y - image_point.y));
    }
    return accumulated / static_cast<double>(projected.size());
}

}

auto KeypointPnpSolution::solve() -> bool {
    const auto& indices = input.indices;
    const auto& points  = input.points;
    const auto& table   = input.object_points;

    if (indices.size() != points.size()) return false;
    if (indices.size() < 4) return false;

    const auto camera_matrix = input.camera.intrinsic();
    const auto distort_coeff = input.camera.distortion();

    auto object_points = std::vector<cv::Point3f> { };
    auto image_points  = std::vector<cv::Point2f> { };
    object_points.reserve(indices.size());
    image_points.reserve(points.size());

    for (const auto& [index, point] : std::views::zip(indices, points)) {
        if (index >= table.size()) continue;
        object_points.push_back(table[index].make<cv::Point3f>());
        image_points.push_back(point);
    }
    if (object_points.size() < 4) return false;

    auto inliers = std::vector<int> { };

    if (input.use_ransac) {
        auto rvec     = cv::Vec3d { };
        auto tvec     = cv::Vec3d { };
        auto selected = std::vector<int> { };

        // 用 AP3P 作为 RANSAC 的最小解算器：待定内点（如仅剩共面的 exchange 点）
        // 会让 SQPnP 在随机最小样本上退化，AP3P 更稳健；最终位姿仍由 SQPnP 求解。
        const auto ok = cv::solvePnPRansac(object_points, image_points, camera_matrix,
            distort_coeff, rvec, tvec, false, input.ransac_iterations,
            static_cast<float>(input.ransac_reprojection_error_px), input.ransac_confidence,
            selected, cv::SOLVEPNP_AP3P);

        if (!ok || selected.size() < 4) return false;
        inliers = std::move(selected);
    } else {
        inliers.resize(object_points.size());
        std::iota(inliers.begin(), inliers.end(), 0);
    }

    auto solve_object = std::vector<cv::Point3f> { };
    auto solve_image  = std::vector<cv::Point2f> { };
    solve_object.reserve(inliers.size());
    solve_image.reserve(inliers.size());
    for (const auto index : inliers) {
        if (index < 0 || static_cast<std::size_t>(index) >= object_points.size()) continue;
        solve_object.push_back(object_points[static_cast<std::size_t>(index)]);
        solve_image.push_back(image_points[static_cast<std::size_t>(index)]);
    }
    if (solve_object.size() < 4) return false;

    auto rota_vecs = std::vector<cv::Vec3d> { };
    auto tran_vecs = std::vector<cv::Vec3d> { };

    const auto success = cv::solvePnPGeneric(solve_object, solve_image, camera_matrix,
        distort_coeff, rota_vecs, tran_vecs, false, cv::SOLVEPNP_SQPNP);

    if (!success || rota_vecs.empty() || rota_vecs.size() != tran_vecs.size()) return false;

    auto best_error = std::numeric_limits<double>::max();
    auto best_rvec  = cv::Vec3d { };
    auto best_tvec  = cv::Vec3d { };

    for (const auto& [candidate_rvec, candidate_tvec] : std::views::zip(rota_vecs, tran_vecs)) {
        auto rvec = candidate_rvec;
        auto tvec = candidate_tvec;

        if (min_camera_depth(solve_object, rvec, tvec) <= 0.0) continue;

        cv::solvePnPRefineLM(solve_object, solve_image, camera_matrix, distort_coeff, rvec, tvec);
        if (min_camera_depth(solve_object, rvec, tvec) <= 0.0) continue;

        const auto error = mean_reprojection_error(
            solve_object, solve_image, camera_matrix, distort_coeff, rvec, tvec);
        if (error < best_error) {
            best_error = error;
            best_rvec  = rvec;
            best_tvec  = tvec;
        }
    }

    if (best_error == std::numeric_limits<double>::max()) return false;

    const auto reprojection_error = mean_reprojection_error(
        object_points, image_points, camera_matrix, distort_coeff, best_rvec, best_tvec);

    auto rotation_mat = cv::Mat { };
    cv::Rodrigues(best_rvec, rotation_mat);

    auto rotation_camera_object = Eigen::Matrix3d { };
    cv::cv2eigen(rotation_mat, rotation_camera_object);

    auto translation_camera_object = Eigen::Vector3d { };
    cv::cv2eigen(best_tvec, translation_camera_object);

    const auto q_odom_camera = input.camera.orientation.make<Eigen::Quaterniond>();
    const auto t_odom_camera = input.camera.translation.make<Eigen::Vector3d>();

    const auto q_camera_object =
        Eigen::Quaterniond { opencv2ros_rotation(rotation_camera_object) }.normalized();
    const auto t_camera_object = Eigen::Vector3d { opencv2ros_position(translation_camera_object) };

    const auto q_odom_object = Eigen::Quaterniond { q_odom_camera * q_camera_object }.normalized();
    const auto t_odom_object = Eigen::Vector3d { q_odom_camera * t_camera_object + t_odom_camera };

    result.transform.translation = Translation { t_odom_object };
    result.transform.orientation = Orientation { q_odom_object };
    result.reprojection_error    = reprojection_error;
    return true;
}
