#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <eigen_conversions/eigen_msg.h>
#include <tf2_ros/transform_broadcaster.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tf_linker");

    ros::NodeHandle nh;

    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);
    tf2_ros::TransformBroadcaster br;

    Eigen::Affine3d T_base2camera;
    T_base2camera = Eigen::Translation3d(0.23, 0, 0);

    ros::Rate rate(10.0);
    while (nh.ok())
    {
        geometry_msgs::TransformStamped transformStamped_gmap2base;
        geometry_msgs::TransformStamped transformStamped_rtab2camera;
        Eigen::Affine3d T_gmap2base;
        Eigen::Affine3d T_rtab2camera;
        try
        {
            transformStamped_gmap2base = tfBuffer.lookupTransform("gmapping_map", "base_link",
                                                                  ros::Time(0));
            transformStamped_rtab2camera = tfBuffer.lookupTransform("rtabmap_map", "camera_link",
                                                                    ros::Time(0));
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("%s", ex.what());
            ros::Duration(0.02).sleep();
            continue;
        }

        tf::transformMsgToEigen(transformStamped_gmap2base.transform, T_gmap2base);
        tf::transformMsgToEigen(transformStamped_rtab2camera.transform, T_rtab2camera);

        Eigen::Affine3d T_gmap2rtab;
        T_gmap2rtab = T_gmap2base * T_base2camera * T_rtab2camera.inverse();

        geometry_msgs::TransformStamped transformStamped_gmap2rtab;
        tf::transformEigenToMsg(T_gmap2rtab, transformStamped_gmap2rtab.transform);

        transformStamped_gmap2rtab.header.stamp = ros::Time::now();
        transformStamped_gmap2rtab.header.frame_id = "gmapping_map";
        transformStamped_gmap2rtab.child_frame_id = "rtabmap_map";

        br.sendTransform(transformStamped_gmap2rtab);

        rate.sleep();
    }
    return 0;
};