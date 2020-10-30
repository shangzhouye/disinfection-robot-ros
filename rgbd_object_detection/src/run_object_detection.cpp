/// \file
/// \brief Node for object detection and map building

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

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

float color_cx = 313.90618896484375;
float color_cy = 246.54579162597656;
float color_fx = 616.7605590820312;
float color_fy = 617.09521484375;

ros::Publisher objects_pub;

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
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "run_object_detection");
    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "camera/aligned_depth_to_color/image_raw", 1);
    message_filters::Subscriber<rgbd_object_detection::MaskrcnnResult> result_sub(nh, "maskrcnn/bbox", 1);

    objects_pub = nh.advertise<sensor_msgs::PointCloud2>("objects_clouds", 10);

    // message_filters::TimeSynchronizer<sensor_msgs::Image, rgbd_object_detection::MaskrcnnResult> sync(depth_sub, result_sub, 10);

    // since realsense camera has syncronized the time stamp, use ExactTime config here
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, rgbd_object_detection::MaskrcnnResult> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depth_sub, result_sub);
    sync.registerCallback(boost::bind(&mask_callback, _1, _2));

    ros::spin();

    return 0;
}
