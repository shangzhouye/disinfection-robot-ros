/// \file
/// \brief Saving rgb images

#include <rgbd_object_detection/camera_interface.hpp>
#include "ros/ros.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_saver");
    ros::NodeHandle nh;

    double color_cx;
    double color_cy;
    double color_fx;
    double color_fy;
    double depth_cx;
    double depth_cy;
    double depth_fx;
    double depth_fy;
    nh.getParam("/color_cx", color_cx);
    nh.getParam("/color_cy", color_cy);
    nh.getParam("/color_fx", color_fx);
    nh.getParam("/color_fy", color_fy);
    nh.getParam("/depth_cx", depth_cx);
    nh.getParam("/depth_cy", depth_cy);
    nh.getParam("/depth_fx", depth_fx);
    nh.getParam("/depth_fy", depth_fy);

    disinfection_robot::CameraInterface my_camera(nh,
                                                  color_cx,
                                                  color_cy,
                                                  color_fx,
                                                  color_fy,
                                                  depth_cx,
                                                  depth_cy,
                                                  depth_fx,
                                                  depth_fy,
                                                  true,
                                                  false);

    ros::spin();

    return 0;
}