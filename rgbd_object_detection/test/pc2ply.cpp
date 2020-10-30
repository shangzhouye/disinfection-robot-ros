/// \file
/// \brief Convert pointcloud to ply files (each frame)
///
/// SUBSCRIBES:
///     camera/depth/color/points (sensor_msgs/PointCloud2): The point cloud realsense is publishing to

#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <string>
#include <iostream>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void pointcloud2ply(const sensor_msgs::PointCloud2 &msg)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(msg, cloud);

    // convert the pointcloud axis
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        float temp = cloud.points[i].x;
        cloud.points[i].x = cloud.points[i].z;
        cloud.points[i].z = -cloud.points[i].y;
        cloud.points[i].y = -temp;
    }

    static int id = 0;
    std::string file_name = std::to_string(id) + ".ply";
    pcl::io::savePLYFileASCII(file_name, cloud);
    std::cout << "Saved ply file id: " << id << std::endl;
    id++;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pc2ply");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("camera/depth/color/points", 1000, pointcloud2ply);

    ros::spin();

    return 0;
}
