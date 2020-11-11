/// \file
/// \brief Utility functions for point cloud manipulation

#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <rgbd_object_detection/pc_utils.hpp>
#include <pcl/filters/voxel_grid.h>

namespace disinfection_robot
{

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void voxel_filter(PointCloud::Ptr in_cloud, double resolution)
{
    // std::cout << "Before filtering - point cloud has " << in_cloud->size() << " points." << std::endl;
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setLeafSize(resolution, resolution, resolution);
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(in_cloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*in_cloud);
    // std::cout << "After filtering - point cloud has " << in_cloud->size() << " points." << std::endl;
}

} // namespace disinfection_robot