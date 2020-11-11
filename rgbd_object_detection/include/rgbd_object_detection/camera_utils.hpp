#ifndef CAMERA_UTILS_INCLUDE_GUARD_HPP
#define CAMERA_UTILS_INCLUDE_GUARD_HPP
/// \file
/// \brief Camera model and camera lidar fusion

// ROS
#include "ros/ros.h"
// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

namespace disinfection_robot
{

class Camera
{
public:
    float color_cx_;
    float color_cy_;
    float color_fx_;
    float color_fy_; // Camera intrinsics

    Eigen::Matrix3d Rcl_; // extrinsic, from camera to lidar frame transformation
    Eigen::Matrix<double, 3, 1> tcl_;

    Eigen::Matrix<double, 4, 4> Tlm_; // from lidar to map

    Camera();

    /*! \brief construct camera object with instrinsics
    */
    Camera(ros::NodeHandle &nh)
    {
        nh.getParam("/color_cx", color_cx_);
        nh.getParam("/color_cy", color_cy_);
        nh.getParam("/color_fx", color_fx_);
        nh.getParam("/color_fy", color_fy_);

        std::vector<double> Rcl_temp;
        std::vector<double> tcl_temp;
        nh.getParam("/Rcl", Rcl_temp);
        nh.getParam("/tcl", tcl_temp);
        Rcl_ << Rcl_temp[0], Rcl_temp[1], Rcl_temp[2],
            Rcl_temp[3], Rcl_temp[4], Rcl_temp[5],
            Rcl_temp[6], Rcl_temp[7], Rcl_temp[8];
        tcl_ << tcl_temp[0], tcl_temp[1], tcl_temp[2];
    }

    /*! \brief return 3 by 3 intrinsic matrix
    */
    Eigen::Matrix<double, 3, 3> K() const
    {
        Eigen::Matrix<double, 3, 3> k;
        k << color_fx_, 0, color_cx_, 0, color_fy_, color_cy_, 0, 0, 1;
        return k;
    }

    /*! \brief transform a point from lidar coordinate to camera coordinate
    */
    Eigen::Matrix<double, 3, 1> lidar2camera(const Eigen::Matrix<double, 3, 1> &p_l)
    {
        return Rcl_ * p_l + tcl_;
    }

    /*! \brief transform a point from camera coordinate to sensor coordinate
    */
    Eigen::Matrix<int, 2, 1> camera2pixel(const Eigen::Matrix<double, 3, 1> &p_c)
    {
        return Eigen::Matrix<int, 2, 1>(
            color_fx_ * p_c(0, 0) / p_c(2, 0) + color_cx_,
            color_fy_ * p_c(1, 0) / p_c(2, 0) + color_cy_);
    }

    /*! \brief transform a point in the lidar coordinate to pixel position (sensor coordinate)
    */
    Eigen::Matrix<int, 2, 1> lidar2pixel(const Eigen::Matrix<double, 3, 1> &p_l)
    {
        return camera2pixel(lidar2camera(p_l));
    }

    /*! \brief transform a point from lidar coordinate to map coordinate
    */
    Eigen::Matrix<double, 3, 1> lidar2map(const Eigen::Matrix<double, 3, 1> &p_l)
    {
        return (Tlm_ * p_l.colwise().homogeneous()).colwise().hnormalized();
    }

    /*! \brief given a pixel position and a depth, transfer it into a 3D point in camere frame (all in color camera frame)
    */
    Eigen::Matrix<float, 3, 1> depth2camera(const Eigen::Matrix<int, 2, 1> &p_s, unsigned int depth)
    {
        float z = float(depth);
        float x = (p_s[0] - color_cx_) * z / color_fx_;
        float y = (p_s[1] - color_cy_) * z / color_fy_;

        return Eigen::Matrix<float, 3, 1>(x, y, z);
    }
};

} // namespace disinfection_robot
#endif
