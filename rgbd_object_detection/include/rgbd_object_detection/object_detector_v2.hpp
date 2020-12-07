#ifndef OBJECT_DETECTOR_V2_INCLUDE_GUARD_HPP
#define OBJECT_DETECTOR_V2_INCLUDE_GUARD_HPP
/// \file
/// \brief The class for detecting object poses from masks and pointcloud (using Velodyne lidar)
///
/// PARAMETERS:
///     Clustering parameters - cluster_tolerance, min_cluster_size, max_cluster_size
///     Dowsampling voxel filter - resolution
///     Loop rate of this detection system - loop_rate_
///     Max distance from object centroid to origin - max_distance_

#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <rgbd_object_detection/MaskrcnnResult.h>
#include <iostream>
#include <std_msgs/Header.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <pcl/segmentation/extract_clusters.h>
#include "pcl/filters/extract_indices.h"
#include <pcl/filters/voxel_grid.h>
#include <rgbd_object_detection/camera_utils.hpp>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <memory>
#include <geometry_msgs/TransformStamped.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf2_ros/transform_listener.h>

namespace disinfection_robot
{

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

struct PointCloudProjection
{
    std::vector<Eigen::Matrix<int, 2, 1>> points_2d;
    std::vector<double> depth; // depth of each corresponding point
};

class ObjectDetectorV2
{

public:
    ros::Publisher objects_pub_;
    ros::Publisher clustered_pub_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> raw_pc_sub_;
    message_filters::Subscriber<rgbd_object_detection::MaskrcnnResult> result_sub_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, rgbd_object_detection::MaskrcnnResult> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync_;

    Camera my_camera_;

    int marker_id_ = 0;
    double loop_rate_ = 6.0; // approximately running at 6Hz, delete marker at this rate

    ros::Publisher convex_hull_pub_;

    double ground_plane_height_ = -0.35; // ground plane z axis value

    // define colormap for overlap image and pointcloud
    cv::Mat colormap_;

    ros::Publisher overlap_image_pub_;

    // transform listener
    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;

    float max_distance_ = 3.0;

public:
    ObjectDetectorV2(ros::NodeHandle &nh) : raw_pc_sub_(nh, "velodyne_points", 100),
                                            result_sub_(nh, "maskrcnn/bbox", 100),
                                            sync_(MySyncPolicy(100), raw_pc_sub_, result_sub_),
                                            my_camera_(nh),
                                            tfListener_(tfBuffer_)
    {

        objects_pub_ = nh.advertise<sensor_msgs::PointCloud2>("objects_clouds", 10);
        clustered_pub_ = nh.advertise<sensor_msgs::PointCloud2>("clustered_clouds", 10);

        // message_filters::TimeSynchronizer<sensor_msgs::Image, rgbd_object_detection::MaskrcnnResult> sync(depth_sub_, result_sub_, 10);

        // since realsense camera has syncronized the time stamp, use ExactTime config here
        sync_.registerCallback(boost::bind(&ObjectDetectorV2::mask_callback, this, _1, _2));

        convex_hull_pub_ = nh.advertise<visualization_msgs::MarkerArray>("object_convex_hull", 100);

        cv::Mat grayscale(256, 1, CV_8UC1);
        for (int i = 0; i < 256; i++)
        {
            grayscale.at<uchar>(i) = i;
        }
        cv::applyColorMap(grayscale, colormap_, cv::COLORMAP_JET);

        overlap_image_pub_ = nh.advertise<sensor_msgs::Image>("overlap_image", 10);
    }

    /*! \brief extract the depth image by mask
    *
    *  \param input_pc - input pointcloud from lidar
    *  \param mask - a const reference of the mask (object region has value of 255)
    *  \param object_pc_list - an array of all the pointclouds of objects
    *  \param cloud_2d - projected cloud on 2d image plane
    */
    void extract_by_mask(PointCloud::Ptr input_pc,
                         const cv::Mat &mask,
                         std::vector<PointCloud::Ptr> &object_pc_list,
                         const PointCloudProjection &cloud_2d);

    /*! \brief cluster the segmented object point cloud
    *
    *  \param object_cloud - segmented object point cloud; this pointcloud will be modified
    */
    void find_largest_cluster(PointCloud::Ptr object_cloud,
                              double cluster_tolerance = 0.2,
                              int min_cluster_size = 50,
                              int max_cluster_size = 307200);

    /*! \brief find the projected 2D convex hull of  the object
    *
    *  \param in_cloud - pointcloud of an object (2D)
    *  \return convex hull vertices
    */
    PointCloud::Ptr find_2D_convex_hull(PointCloud::Ptr in_cloud);

    void mask_callback(const sensor_msgs::PointCloud2::ConstPtr &raw_pc, // Learned: it has to be const pointer
                       const rgbd_object_detection::MaskrcnnResult::ConstPtr &mask_result);

    /*! \brief Publish the polygon given by the convex hull
    *
    *  \param polygon - indices points of a convex hull
    *  \param marker_array - return line strip markers
    */
    void polygon_marker(PointCloud::Ptr polygon,
                        visualization_msgs::MarkerArray &marker_array);

    /*! \brief project the point cloud onto image plane using extrinsics
    *
    *  \param in_cloud - input pointcloud
    *  \param out_cloud_2d - projected point cloud
    */
    void project2image_plane(PointCloud::Ptr in_cloud, PointCloudProjection &out_cloud_2d);

    /*! \brief Overlap a projected pointcloud onto image plane
    *           the intensity of input projected point cloud denotes its depth in the original pointcloud
    * 
    *  \param projected_cloud - the projected point cloud
    *  \param image - the image to be overlaped (image will be modified)
    *  \param colormap - the defined colormap
    */
    void pcl_image_overlap(const PointCloudProjection &projected_cloud,
                           cv::Mat &image,
                           cv::Mat colormap);

    /*! \brief Transform the convex hull from the velodyne frame to the map frame
    * 
    *  \param convex_hull_cloud - the point cloud of the convex hull, it will be modified
    */
    void velodyne2map_frame(PointCloud::Ptr convex_hull_cloud);

    /*! \brief Project a point cloud onto 2D plane
    * 
    *  \param in_cloud - the point cloud will be modified
    */
    void cloud_2d(PointCloud::Ptr in_cloud);

    /*! \brief Check if this object point cloud is valid
    *           Eliminate point cloud that is too far away or has the shape of wall
    * 
    *  \param in_cloud - the point cloud will be modified
    *  \return if this object point cloud is valid
    */
    bool is_valid_object(PointCloud::Ptr in_cloud);
};

} // namespace disinfection_robot

#endif