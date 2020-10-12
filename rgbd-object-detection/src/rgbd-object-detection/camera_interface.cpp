/// \file
/// \brief The module contains help function to interface with the realsense rgb-d camera

#include <rgbd-object-detection/camera_interface.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

namespace disinfection_robot
{

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void CameraInterface::color_callback(const sensor_msgs::ImageConstPtr &msg)
{

    // std::cout << "Color image sequence: " << msg->header.seq << std::endl;

    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    color_ = cv_ptr->image;
}

void CameraInterface::depth_callback(const sensor_msgs::ImageConstPtr &msg)
{
    // std::cout << "Depth image sequence: " << msg->header.seq << std::endl;

    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    depth_ = cv_ptr->image;

    // int v = depth_.rows / 2;
    // int u = depth_.cols / 2;
    // unsigned int center_depth = depth_.ptr<unsigned short>(v)[u];
    // std::cout << "Center depth in mm: " << center_depth << std::endl;

    depth2pc(true);
}

void CameraInterface::depth2pc(bool if_publish)
{
    PointCloud::Ptr current_cloud(new PointCloud);
    for (int v = 0; v < depth_.rows; v++)
    {
        for (int u = 0; u < depth_.cols; u++)
        {
            unsigned int d = depth_.ptr<unsigned short>(v)[u];
            if (d == 0)
            {
                continue;
            }

            PointT p;
            p.z = float(d);
            p.x = (u - depth_cx_) * p.z / depth_fx_;
            p.y = (v - depth_cy_) * p.z / depth_fy_;

            p.z /= 1000.0;
            p.x /= 1000.0;
            p.y /= 1000.0;

            current_cloud->points.push_back(p);
        }
    }

    current_frame_.reset();
    current_frame_ = current_cloud;

    if (if_publish)
    {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*current_cloud, cloud_msg);
        cloud_msg.header.stamp = ros::Time::now();
        cloud_msg.header.frame_id = pc_frame_;
        raw_cloud_pub_.publish(cloud_msg);
    }
}

} // namespace disinfection_robot