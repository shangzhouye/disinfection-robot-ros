#ifndef CAMERA_INTERFACE_INCLUDE_GUARD_HPP
#define CAMERA_INTERFACE_INCLUDE_GUARD_HPP
/// \file
/// \brief The module contains help function to interface with the realsense rgb-d camera
///     1. Read color image and depth image
///     2. Turn depth images into PointCloud using camera intrinsics
///     3. Return color image or PointCloud when requested
/// SUBSCRIBERS:
///     camera/color/image_raw (sensor_msgs/Image): rgb images
///     camera/depth/image_rect_raw (sensor_msgs/Image): depth images
/// PUBLISHERS:
///     camera/raw_cloud (sensor_msgs/PointCloud2): Raw point cloud from realsense camera

#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>


namespace disinfection_robot
{

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class CameraInterface
{
public:
    // defines camera intrinsics: focal lengths (fx, fy) and principal point (cx, cy)
    double color_cx_;
    double color_cy_;
    double color_fx_;
    double color_fy_;
    cv::Mat color_K_;

    double depth_cx_;
    double depth_cy_;
    double depth_fx_;
    double depth_fy_;
    cv::Mat depth_K_;

    image_transport::ImageTransport it_;
    image_transport::Subscriber color_sub;
    image_transport::Subscriber depth_sub;

    // store the current frame
    cv::Mat color_;
    cv::Mat depth_;
    PointCloud::Ptr current_frame_;

    ros::Publisher raw_cloud_pub_;
    std::string pc_frame_ = "camera_depth_optical_frame"; // the frame pc is published in

public:
    /*! \brief Store camera intrinsics; create subscribers
    *
    *  \param nh - node handler
    *  \param color_cx - principal point (cx, cy)
    *  \param color_cy
    *  \param color_fx - focal lengths (fx, fy)
    *  \param color_fy
    *  \param depth_cx - principal point (cx, cy)
    *  \param depth_cy
    *  \param depth_fx - focal lengths (fx, fy)
    *  \param depth_fy
    */
    CameraInterface(ros::NodeHandle &nh,
                    double color_cx,
                    double color_cy,
                    double color_fx,
                    double color_fy,
                    double depth_cx,
                    double depth_cy,
                    double depth_fx,
                    double depth_fy)
        : it_(nh), color_cx_(color_cx), color_cy_(color_cy), color_fx_(color_fx), color_fy_(color_fy),
          depth_cx_(depth_cx), depth_cy_(depth_cy), depth_fx_(depth_fx), depth_fy_(depth_fy)
    {
        color_K_ = (cv::Mat_<double>(3, 3) << color_fx_, 0, color_cx_,
                    0, color_fy_, color_cy_,
                    0, 0, 1);

        depth_K_ = (cv::Mat_<double>(3, 3) << depth_fx_, 0, depth_cx_,
                    0, depth_fy_, depth_cy_,
                    0, 0, 1);

        color_sub = it_.subscribe("camera/color/image_raw", 1000,
                                  &CameraInterface::color_callback, this);
        depth_sub = it_.subscribe("camera/depth/image_rect_raw", 1000,
                                  &CameraInterface::depth_callback, this);

        // std::cout << "Color K: " << color_K_ << std::endl;
        // std::cout << "Depth K: " << depth_K_ << std::endl;

        raw_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("camera/raw_cloud", 10);
    }

    // rgb8: CV_8UC3, color image with red-green-blue color order
    void color_callback(const sensor_msgs::ImageConstPtr &msg);

    // mono16: CV_16UC1, 16-bit grayscale image
    void depth_callback(const sensor_msgs::ImageConstPtr &msg);

    /*! \brief Turn the stored depth image into point cloud
    *
    *  \param if_publish - whether publish the pointcloud
    */
    void depth2pc(bool if_publish);
};

} // namespace disinfection_robot

#endif