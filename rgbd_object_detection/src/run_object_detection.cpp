/// \file
/// \brief Node for object detection and map building
///
/// PARAMETERS:
///     Clustering parameters - cluster_tolerance, min_cluster_size, max_cluster_size
///     Dowsampling voxel filter - resolution
///     Color camera instrinsics - color_cx, color_cy , color_fx, color_fy

#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
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

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

float color_cx = 313.90618896484375;
float color_cy = 246.54579162597656;
float color_fx = 616.7605590820312;
float color_fy = 617.09521484375;

ros::Publisher objects_pub;
ros::Publisher clustered_pub;

/*! \brief extract the depth image by mask
*
*  \param depth - a const reference of the depth image
*  \param mask - a const reference of the mask (object region has value of 255)
*  \param object_pc_list - an array of all the pointclouds of objects
*/
void extract_by_mask(const cv::Mat &depth, const cv::Mat &mask, std::vector<PointCloud::Ptr> &object_pc_list)
{
    PointCloud::Ptr current_cloud(new PointCloud);
    for (int v = 0; v < mask.rows; v++)
    {
        for (int u = 0; u < mask.cols; u++)
        {
            // if it is not in the mask, ignore this pixel
            if (mask.ptr<unsigned char>(v)[u] != 255)
            {
                continue;
            }

            unsigned int d = depth.ptr<unsigned short>(v)[u];
            if (d == 0)
            {
                continue;
            }

            PointT p;
            p.z = float(d);
            p.x = (u - color_cx) * p.z / color_fx;
            p.y = (v - color_cy) * p.z / color_fy;

            p.z /= 1000.0;
            p.x /= 1000.0;
            p.y /= 1000.0;

            current_cloud->points.push_back(p);
        }
    }

    object_pc_list.push_back(current_cloud);
}

/*! \brief cluster the segmented object point cloud
*
*  \param object_cloud - segmented object point cloud; this pointcloud will be modified
*/
void find_largest_cluster(PointCloud::Ptr object_cloud,
                          double cluster_tolerance = 0.05,
                          int min_cluster_size = 50,
                          int max_cluster_size = 307200)
{

    std::vector<pcl::PointIndices> object_indices;

    pcl::EuclideanClusterExtraction<PointT> euclid;
    euclid.setInputCloud(object_cloud);
    euclid.setClusterTolerance(cluster_tolerance);
    euclid.setMinClusterSize(min_cluster_size);
    euclid.setMaxClusterSize(max_cluster_size);
    euclid.extract(object_indices);

    // Find the size of the largest object,
    // where size = number of points in the cluster
    size_t max_size = std::numeric_limits<size_t>::min();
    int max_cluster_id = 0;
    for (size_t i = 0; i < object_indices.size(); ++i)
    {
        size_t current_size = object_indices[0].indices.size();
        if (current_size > max_size)
        {
            max_size = current_size;
            max_cluster_id = i;
        }
    }

    // ROS_INFO("Found %ld objects, max size: %ld",
    //          object_indices.size(), max_size);

    if (max_size == 0)
    {
        ROS_INFO("No cluster found.");
        return;
    }

    // Reify indices into a point cloud of the object.
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    *indices = object_indices[max_cluster_id];
    PointCloud::Ptr new_cloud(new PointCloud());
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(object_cloud);
    extract.setIndices(indices);
    extract.filter(*new_cloud);

    new_cloud->swap(*object_cloud);
}

/*! \brief reduce the number of points using voxel filter (downsampling)
*
*  \param in_cloud - the input point cloud, this point cloud will be modified
*  \param resolution - resolution of the filter
*/
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

void mask_callback(const sensor_msgs::ImageConstPtr &depth,
                   const rgbd_object_detection::MaskrcnnResult::ConstPtr &mask_result)
{
    // std::cout << "Inside callback" << std::endl;
    // std::cout << depth->header.stamp << std::endl;
    // std::cout << mask_result->header.stamp << std::endl;

    // read depth image
    cv_bridge::CvImageConstPtr cv_ptr_depth;
    try
    {
        // learned: use toCvShare without format conversion to avoid copy
        cv_ptr_depth = cv_bridge::toCvShare(depth);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat depth_image = cv_ptr_depth->image;

    // std::cout << "Depth Rows: " << depth_image.rows << " Cols: " << depth_image.cols << std::endl;
    // std::cout << "Row length: " << depth_image.step
    //           << " Channels: " << depth_image.channels()
    //           << " Depth: " << depth_image.depth()
    //           << std::endl;

    // create an array of all the objects pointcloud
    std::vector<PointCloud::Ptr> objects_clouds;

    // read masks
    cv_bridge::CvImageConstPtr cv_ptr_mask;

    for (int i = 0; i < mask_result->class_ids.size(); i++)
    {
        try
        {
            // learned: this overload is convenient when you have a pointer to some other message type
            // that contains a sensor_msgs/Image you want to convert.
            cv_ptr_mask = cv_bridge::toCvShare(mask_result->masks[i], mask_result);
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat mask_image = cv_ptr_mask->image;

        // std::cout << "Mask Rows: " << mask_image.rows << " Cols: " << mask_image.cols << std::endl;
        // std::cout << "Row length: " << mask_image.step
        //           << " Channels: " << mask_image.channels()
        //           << " Depth: " << mask_image.depth()
        //           << std::endl;

        extract_by_mask(depth_image, mask_image, objects_clouds);
    }

    // visualization
    PointCloud::Ptr objects_visual(new PointCloud);
    for (int i = 0; i < objects_clouds.size(); i++)
    {
        *objects_visual += *(objects_clouds[i]);
    }
    sensor_msgs::PointCloud2 objects_visual_msg;
    pcl::toROSMsg(*objects_visual, objects_visual_msg);

    objects_visual_msg.header.stamp = ros::Time::now();
    objects_visual_msg.header.frame_id = "camera_color_optical_frame";
    objects_pub.publish(objects_visual_msg);

    // for each object cloud, do downsampling and clustering
    for (auto each_object : objects_clouds)
    {
        voxel_filter(each_object, 0.02);
        find_largest_cluster(each_object);
    }

    // visualize again after clustering
    PointCloud::Ptr clustered_visual(new PointCloud);
    for (int i = 0; i < objects_clouds.size(); i++)
    {
        *clustered_visual += *(objects_clouds[i]);
    }
    sensor_msgs::PointCloud2 clustered_visual_msg;
    pcl::toROSMsg(*clustered_visual, clustered_visual_msg);

    clustered_visual_msg.header.stamp = ros::Time::now();
    clustered_visual_msg.header.frame_id = "camera_color_optical_frame";
    clustered_pub.publish(clustered_visual_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "run_object_detection");
    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "camera/aligned_depth_to_color/image_raw", 1);
    message_filters::Subscriber<rgbd_object_detection::MaskrcnnResult> result_sub(nh, "maskrcnn/bbox", 1);

    objects_pub = nh.advertise<sensor_msgs::PointCloud2>("objects_clouds", 10);
    clustered_pub = nh.advertise<sensor_msgs::PointCloud2>("clustered_clouds", 10);

    // message_filters::TimeSynchronizer<sensor_msgs::Image, rgbd_object_detection::MaskrcnnResult> sync(depth_sub, result_sub, 10);

    // since realsense camera has syncronized the time stamp, use ExactTime config here
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, rgbd_object_detection::MaskrcnnResult> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depth_sub, result_sub);
    sync.registerCallback(boost::bind(&mask_callback, _1, _2));

    ros::spin();

    return 0;
}
