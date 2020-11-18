/// \file
/// \brief Node for generating observation poses given a 2D occupancy grid
///
/// PARAMETERS:
///     Clustering parameters - cluster_tolerance, min_cluster_size, max_cluster_size
///     Maximum size (length/width) of a potential object - size_max_
///     Loop rate of publish markers - loop_rate_

#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include "pcl/common/angles.h"
#include "pcl/common/common.h"
#include <nav_msgs/GetMap.h>
#include <nav_msgs/GetMapRequest.h>
#include <nav_msgs/GetMapResponse.h>
#include <nav_msgs/GetMapResult.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/MapMetaData.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <geometry_msgs/Quaternion.h>

#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

namespace disinfection_robot
{

class ObservationPoint
{

public:
    ros::ServiceClient map_client_;
    nav_msgs::MapMetaData map_info_;
    nav_msgs::OccupancyGrid map_;

    ros::Publisher observation_pose_pub_;
    ros::Publisher potential_objects_pub_;
    visualization_msgs::MarkerArray observation_poses_;
    visualization_msgs::MarkerArray potential_objects_;
    int marker_id_ = 0;
    double loop_rate_ = 2.0;

    double size_max_ = 2.0;

    PointCloud::Ptr occupied_cloud_;

public:
    ObservationPoint(ros::NodeHandle &nh)
    {
        map_client_ = nh.serviceClient<nav_msgs::GetMap>("static_map");
        observation_pose_pub_ = nh.advertise<visualization_msgs::MarkerArray>("observation_pose", 10);
        potential_objects_pub_ = nh.advertise<visualization_msgs::MarkerArray>("potential_objects", 10);

        nav_msgs::GetMap srv_map;
        map_client_.waitForExistence();
        if (map_client_.call(srv_map))
        {
            map_ = srv_map.response.map;
            map_info_ = map_.info;
            ROS_INFO("Got map width: %d height: %d", map_info_.width, map_info_.height);

            // ensure there is no rotation
            assertm(map_info_.origin.orientation.x == 0 &&
                        map_info_.origin.orientation.y == 0 &&
                        map_info_.origin.orientation.z == 0 &&
                        map_info_.origin.orientation.w == 1,
                    "There should be no rotation.");
        }
        else
        {
            ROS_ERROR("Failed to call map service");
        }
    }

    /*! \brief transform a point from grid coordinate to map coordinate (in meters)
    */
    Eigen::Vector2d occupancy_grid2map(Eigen::Vector2i grid_xy)
    {
        // Added 0.5 for an inconsistency visualized in Rviz, not sure why
        return Eigen::Vector2d((grid_xy[0] + 0.5) * map_info_.resolution + map_info_.origin.position.x,
                               (grid_xy[1] + 0.5) * map_info_.resolution + map_info_.origin.position.y);
    }

    /*! \brief create marker for each potential objects
    */
    visualization_msgs::Marker create_potential_objects(Eigen::Vector2d position,
                                                        geometry_msgs::Quaternion orientation,
                                                        Eigen::Vector2d scale)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "/map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "potential_objects";
        marker.id = marker_id_;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = position[0];
        marker.pose.position.y = position[1];
        marker.pose.position.z = 0;
        marker.pose.orientation = orientation;
        marker.scale.x = scale[0];
        marker.scale.y = scale[1];
        marker.scale.z = 0.05;
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.5f;

        marker.lifetime = ros::Duration(1 / loop_rate_);

        marker_id_++;

        return marker;
    }

    /*! \brief cluster all the points into clusters, return a vector of clusters
    */
    void euclidean_clustering(PointCloud::Ptr in_cloud, std::vector<PointCloud::Ptr> &out_clusters)
    {
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(in_cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointT> ec;

        ec.setClusterTolerance(2.0 * map_info_.resolution);
        ec.setMinClusterSize(10);
        ec.setMaxClusterSize(map_info_.height * map_info_.width);
        ec.setSearchMethod(tree);
        ec.setInputCloud(in_cloud);
        ec.extract(cluster_indices);

        for (int i = 0; i < cluster_indices.size(); i++)
        {
            // Reify indices into a point cloud of the object.
            PointCloud::Ptr new_cloud(new PointCloud());

            pcl::PointIndices::Ptr indices(new pcl::PointIndices);
            *indices = cluster_indices[i];
            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(in_cloud);
            extract.setIndices(indices);
            extract.filter(*new_cloud);

            out_clusters.push_back(new_cloud);
        }
    }

    /*! \brief given a pointer to a pose and dimension
     *          modify them as an axis aligned bounding box of given cloud
    */
    void axis_aligned_bounding_box(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                   geometry_msgs::Pose *pose,
                                   geometry_msgs::Vector3 *dimensions)
    {
        PointT min_pcl;
        PointT max_pcl;
        pcl::getMinMax3D<PointT>(*cloud, min_pcl, max_pcl);

        pose->position.x = (min_pcl.x + max_pcl.x) / 2.0;
        pose->position.y = (min_pcl.y + max_pcl.y) / 2.0;
        pose->position.z = (min_pcl.z + max_pcl.z) / 2.0;
        pose->orientation.w = 1;
        pose->orientation.x = 0;
        pose->orientation.y = 0;
        pose->orientation.z = 0;

        dimensions->x = ((max_pcl.x - min_pcl.x) != 0) ? max_pcl.x - min_pcl.x : map_info_.resolution;
        dimensions->y = ((max_pcl.y - min_pcl.y) != 0) ? max_pcl.y - min_pcl.y : map_info_.resolution;
        dimensions->z = ((max_pcl.z - min_pcl.z) != 0) ? max_pcl.z - min_pcl.z : map_info_.resolution;
    }

    /*! \brief eliminate unrealistic potential bounding box by its size and lenght/width radio
    */
    bool potential_object_validate(visualization_msgs::Marker box)
    {
        if (box.scale.x >= size_max_ || box.scale.y >= size_max_)
        {
            return false;
        }

        return true;
    }

    void pipeline()
    {
        occupied_cloud_.reset(new PointCloud);
        // traverse the 2D occupancy grid
        for (unsigned int x = 0; x < map_info_.width; x++)
        {
            for (unsigned int y = 0; y < map_info_.height; y++)
            {
                // skip unexplored region
                if (map_.data[x + map_info_.width * y] == -1)
                {
                    continue;
                }

                // if it is occupied
                if (map_.data[x + map_info_.width * y] == 100)
                {
                    Eigen::Vector2d position = occupancy_grid2map(Eigen::Vector2i(x, y));

                    PointT T;
                    T.x = position[0];
                    T.y = position[1];
                    T.z = 0;
                    occupied_cloud_->points.push_back(T);
                }
            }
        }

        // cluster all the occupied points
        std::vector<PointCloud::Ptr> clusters_on_map;
        euclidean_clustering(occupied_cloud_, clusters_on_map);

        for (auto ea : clusters_on_map)
        {
            visualization_msgs::Marker potential_object =
                create_potential_objects(Eigen::Vector2d(), geometry_msgs::Quaternion(), Eigen::Vector2d());

            axis_aligned_bounding_box(ea, &potential_object.pose, &potential_object.scale);

            if (potential_object_validate(potential_object))
            {
                potential_objects_.markers.push_back(potential_object);
            }
        }

        publish_marker();
    }

    /*! \brief looping and publish markers at a given rate
    */
    void publish_marker()
    {
        ros::Rate loop_rate(loop_rate_);
        while (ros::ok())
        {
            observation_pose_pub_.publish(observation_poses_);
            potential_objects_pub_.publish(potential_objects_);
            ros::spinOnce();
            loop_rate.sleep();
        }
    }
};

} // namespace disinfection_robot

int main(int argc, char **argv)
{
    ros::init(argc, argv, "observation_point_generation");
    ros::NodeHandle nh;

    disinfection_robot::ObservationPoint observation_points(nh);
    observation_points.pipeline();

    ros::spin();

    return 0;
}
