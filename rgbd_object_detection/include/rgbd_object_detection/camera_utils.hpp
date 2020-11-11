#ifndef CAMERA_UTILS_INCLUDE_GUARD_HPP
#define CAMERA_UTILS_INCLUDE_GUARD_HPP
/// \file
/// \brief Camera model and camera lidar fusion
///

/*! Definition of extrinsics
 *  map -> ... -> base_link -> velodyne -> camera_link -> camera_color_optical_frame
 *      (obtained from slam)         (extrinsic)    (given by camera)
 *   
 *                    z  x                          / z         
 *                    | /                          /_ _ x                  
 *               y _ _|/                          |                       
 *                                                | y                       
 *               camera_link              camera_color_optical_frame                 
 *  
*/

// ROS
#include "ros/ros.h"
// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

namespace disinfection_robot
{

Eigen::Affine3d create_rotation_matrix(double ax, double ay, double az)
{
    Eigen::Affine3d rx =
        Eigen::Affine3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
    Eigen::Affine3d ry =
        Eigen::Affine3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
    Eigen::Affine3d rz =
        Eigen::Affine3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
    return rz * ry * rx;
}

class Camera
{
public:
    float color_cx_;
    float color_cy_;
    float color_fx_;
    float color_fy_; // Camera intrinsics

    Eigen::Matrix4d Tco2l_; // extrinsic, from camera optical to lidar frame transformation

    Camera();

    /*! \brief construct camera object with instrinsics
    */
    Camera(ros::NodeHandle &nh)
    {
        nh.getParam("/color_cx", color_cx_);
        nh.getParam("/color_cy", color_cy_);
        nh.getParam("/color_fx", color_fx_);
        nh.getParam("/color_fy", color_fy_);

        std::vector<double> rpy_lc;
        std::vector<double> t_lc;
        nh.getParam("/rpy_lc", rpy_lc);
        nh.getParam("/t_lc", t_lc);
        Eigen::Affine3d Rot_lc = create_rotation_matrix(rpy_lc[0], rpy_lc[1], rpy_lc[2]);
        Eigen::Affine3d Trans_lc(Eigen::Translation3d(Eigen::Vector3d(t_lc[0], t_lc[1], t_lc[2])));
        // transformation from camera link to lidar
        Eigen::Affine3d T_cl = (Trans_lc * Rot_lc).inverse();

        std::vector<double> rpy_c2co;
        std::vector<double> t_c2co;
        nh.getParam("/rpy_c2co", rpy_c2co);
        nh.getParam("/t_c2co", t_c2co);
        Eigen::Affine3d Rot_c2co = create_rotation_matrix(rpy_c2co[0], rpy_c2co[1], rpy_c2co[2]);
        Eigen::Affine3d Trans_c2co(Eigen::Translation3d(Eigen::Vector3d(t_c2co[0], t_c2co[1], t_c2co[2])));
        // transformation from camera optical frame to camera link
        Eigen::Affine3d T_co2c = (Trans_c2co * Rot_c2co).inverse();

        Tco2l_ = (T_co2c * T_cl).matrix();

        // validated by comparing to ros tf
        // std::cout << Tco2l_ << std::endl;
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
        return (Tco2l_ * p_l.colwise().homogeneous()).colwise().hnormalized();
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
    *           Given map to lidar transform and 3D point
    */
    Eigen::Matrix<double, 3, 1> lidar2map(const Eigen::Matrix<double, 3, 1> &p_l, const Eigen::Matrix<double, 4, 4> &Tml)
    {
        return (Tml * p_l.colwise().homogeneous()).colwise().hnormalized();
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
