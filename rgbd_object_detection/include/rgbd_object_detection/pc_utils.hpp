#ifndef PC_UTILS_INCLUDE_GUARD_HPP
#define PC_UTILS_INCLUDE_GUARD_HPP
/// \file
/// \brief Utility functions for point cloud manipulation

#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

namespace disinfection_robot
{

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

/*! \brief reduce the number of points using voxel filter (downsampling)
*
*  \param in_cloud - the input point cloud, this point cloud will be modified
*  \param resolution - resolution of the filter
*/
void voxel_filter(PointCloud::Ptr in_cloud, double resolution);

} // namespace disinfection_robot

#endif