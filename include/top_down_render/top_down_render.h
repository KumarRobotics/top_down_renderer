#ifndef TOP_DOWN_RENDER_H_
#define TOP_DOWN_RENDER_H_

#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <chrono>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include "top_down_render/point_xyz_class_normal.h"
#include "top_down_render/top_down_map_polar.h"
#include "top_down_render/particle_filter.h"
#include "top_down_render/scan_renderer_polar.h"

class TopDownRender {
  public:
    TopDownRender(ros::NodeHandle &nh);
    void initialize();
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport *it_;
    tf2_ros::StaticTransformBroadcaster *static_broadcaster_;
    ros::Subscriber pc_sub_;
    ros::Subscriber gt_pose_sub_;
    ros::Publisher pose_pub_;
    image_transport::Publisher img_pub_;
    image_transport::Publisher scan_pub_;
    image_transport::Publisher geo_scan_pub_;
    image_transport::Publisher map_pub_;

    Eigen::Affine2f gt_pose_;
    cv::Point map_center_;
    cv::Mat color_lut_;
    cv::Mat background_img_;
    TopDownMapPolar *map_;
    ParticleFilter *filter_;
    ScanRendererPolar *renderer_;

    float current_res_ = 2;

    void publishSemanticTopDown(std::vector<Eigen::ArrayXXf> &top_down, std_msgs::Header &header);
    void publishGeometricTopDown(std::vector<Eigen::ArrayXXf> &top_down, std_msgs::Header &header);
    void publishLocalMap(int h, int w, Eigen::Vector2f center, float res, std_msgs::Header &header);
    void visualize(std::vector<Eigen::ArrayXXf> &classes, cv::Mat &img);
    cv::Mat visualizeAnalog(Eigen::ArrayXXf &cls, float scale);
    void updateFilter(std::vector<Eigen::ArrayXXf> &top_down, 
                      std::vector<Eigen::ArrayXXf> &top_down_geo, float res, std_msgs::Header &header);
    void pcCallback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&);
    void gtPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose);
};

#endif //TOP_DOWN_RENDER_H_
