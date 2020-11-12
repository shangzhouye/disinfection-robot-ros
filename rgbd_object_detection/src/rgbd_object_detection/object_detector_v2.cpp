/// \file
/// \brief The class for detecting object poses from masks and pointcloud (using Velodyne lidar)
///

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
#include <rgbd_object_detection/object_detector_v2.hpp>
#include <rgbd_object_detection/pc_utils.hpp>
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
#include <pcl/surface/concave_hull.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <memory>

namespace disinfection_robot
{

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void ObjectDetectorV2::extract_by_mask(PointCloud::Ptr input_pc,
                                       const cv::Mat &mask,
                                       std::vector<PointCloud::Ptr> &object_pc_list,
                                       const PointCloudProjection &cloud_2d)
{
    PointCloud::Ptr current_cloud(new PointCloud);

    for (size_t i = 0; i < input_pc->points.size(); i++)
    {
        // if it is not inside image, continue
        if (cloud_2d.points_2d[i][0] < 0 || cloud_2d.points_2d[i][0] > my_camera_.image_width_ ||
            cloud_2d.points_2d[i][1] < 0 || cloud_2d.points_2d[i][1] > my_camera_.image_height_)
        {
            continue;
        }

        // if this point is not inside mask, continue
        if (mask.ptr<unsigned char>(cloud_2d.points_2d[i][1])[cloud_2d.points_2d[i][0]] != 255)
        {
            continue;
        }

        current_cloud->points.push_back(input_pc->points[i]);
    }

    object_pc_list.push_back(current_cloud);
}

void ObjectDetectorV2::find_largest_cluster(PointCloud::Ptr object_cloud,
                                            double cluster_tolerance,
                                            int min_cluster_size,
                                            int max_cluster_size)
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

void ObjectDetectorV2::project2image_plane(PointCloud::Ptr in_cloud, PointCloudProjection &out_cloud_2d)
{

    for (size_t i = 0; i < in_cloud->points.size(); i++)
    {
        Eigen::Matrix<double, 3, 1> p_l(in_cloud->points[i].x,
                                        in_cloud->points[i].y,
                                        in_cloud->points[i].z);
        Eigen::Matrix<int, 2, 1> p_s = my_camera_.lidar2pixel(p_l); // point position in sensor coordinate

        out_cloud_2d.points_2d.push_back(p_s);
        out_cloud_2d.depth.push_back(in_cloud->points[i].x);
    }

    return;
}

void ObjectDetectorV2::mask_callback(const sensor_msgs::PointCloud2::ConstPtr &raw_pc,
                                     const rgbd_object_detection::MaskrcnnResult::ConstPtr &mask_result)
{
    // std::cout << "Inside callback" << std::endl;

    PointCloud::Ptr input_pc(new PointCloud);
    pcl::fromROSMsg(*raw_pc, *input_pc);

    // create an array of all the objects pointcloud
    std::vector<PointCloud::Ptr> objects_clouds;

    // read masks
    cv_bridge::CvImageConstPtr cv_ptr_mask;

    PointCloudProjection cloud_on_image;
    project2image_plane(input_pc, cloud_on_image);

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

        extract_by_mask(input_pc, mask_image, objects_clouds, cloud_on_image);
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
    objects_visual_msg.header.frame_id = "velodyne";
    objects_pub_.publish(objects_visual_msg);

    visualization_msgs::MarkerArray polygon_array;

    // for each object cloud, do downsampling and clustering
    for (auto each_object : objects_clouds)
    {
        voxel_filter(each_object, 0.01);
        find_largest_cluster(each_object);

        // if there is no clusters, ignore this object
        if (each_object->points.size() <= 2)
        {
            continue;
        }
        PointCloud::Ptr convex_hull_cloud = find_2D_convex_hull(each_object);
        polygon_marker(convex_hull_cloud, polygon_array);
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
    clustered_visual_msg.header.frame_id = "velodyne";
    clustered_pub_.publish(clustered_visual_msg);

    convex_hull_pub_.publish(polygon_array);
}

PointCloud::Ptr ObjectDetectorV2::find_2D_convex_hull(PointCloud::Ptr in_cloud)
{

    PointCloud::Ptr cloud_2d(new PointCloud);

    for (size_t i = 0; i < in_cloud->points.size(); ++i)
    {
        PointT p;
        p.x = in_cloud->points[i].x;
        p.y = in_cloud->points[i].y;
        p.z = 0;

        cloud_2d->points.push_back(p);
    }

    PointCloud::Ptr convex_hull(new PointCloud);
    pcl::ConvexHull<pcl::PointXYZ> chull;
    chull.setInputCloud(cloud_2d);
    chull.reconstruct(*convex_hull);

    // std::cout << "Convex hull has: " << convex_hull->size()
    //           << " data points." << std::endl;

    return convex_hull;
}

void ObjectDetectorV2::polygon_marker(PointCloud::Ptr polygon,
                                      visualization_msgs::MarkerArray &marker_array)
{
    visualization_msgs::Marker line_strip;
    line_strip.header.frame_id = "velodyne";
    line_strip.header.stamp = ros::Time::now();
    line_strip.ns = "lines";
    line_strip.action = visualization_msgs::Marker::ADD;
    line_strip.pose.orientation.w = 1.0;

    line_strip.id = marker_id_;

    line_strip.type = visualization_msgs::Marker::LINE_STRIP;

    // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
    line_strip.scale.x = 0.05;

    // Line strip is blue
    line_strip.color.b = 1.0;
    line_strip.color.g = 1.0;

    line_strip.color.a = 1.0;

    // Create the vertices for the points and lines
    for (uint32_t i = 0; i < polygon->points.size(); ++i)
    {
        geometry_msgs::Point p;
        p.x = polygon->points[i].x;
        p.y = polygon->points[i].y;
        p.z = ground_plane_height_;
        line_strip.points.push_back(p);
    }

    geometry_msgs::Point p;
    p.x = polygon->points[0].x;
    p.y = polygon->points[0].y;
    p.z = ground_plane_height_;
    line_strip.points.push_back(p);

    line_strip.lifetime = ros::Duration(1.0 / loop_rate_);

    marker_array.markers.push_back(line_strip);

    marker_id_++;
}

void ObjectDetectorV2::pcl_image_overlap(const PointCloudProjection &projected_cloud,
                                         cv::Mat &image,
                                         cv::Mat colormap)
{
    int w = image.size().width;
    int h = image.size().height;

    float min_d, max_d;
    min_d = max_d = projected_cloud.depth[0];
    for (int i = 1; i < projected_cloud.depth.size(); i++)
    {
        float di = projected_cloud.depth[i];
        max_d = di > max_d ? di : max_d;
        min_d = di < min_d ? di : min_d;
    }
    float wid_d = max_d - min_d;

    for (int i = 0; i < projected_cloud.points_2d.size(); i++)
    {
        if (projected_cloud.points_2d[i][0] <= 0 && projected_cloud.points_2d[i][0] >= w &&
            projected_cloud.points_2d[i][1] <= 0 && projected_cloud.points_2d[i][1] >= h)
        {
            continue;
        }
        float distance = projected_cloud.depth[i];

        int colorid = wid_d ? ((distance - min_d) * 255 / wid_d) : 128;
        cv::Vec3b color = colormap.at<cv::Vec3b>(colorid);
        int r = color[0];
        int g = color[1];
        int b = color[2];
        cv::circle(image, cvPoint(projected_cloud.points_2d[i][0], projected_cloud.points_2d[i][1]),
                   2, CV_RGB(r, g, b), -1);
    }
}

} // namespace disinfection_robot