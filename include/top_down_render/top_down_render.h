#ifndef TOP_DOWN_RENDER_H_
#define TOP_DOWN_RENDER_H_

#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <chrono>
#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include "top_down_render/point_xyz_class_normal.h"
#include "top_down_render/top_down_map_polar.h"
#include "top_down_render/particle_filter.h"
#include "top_down_render/scan_renderer_polar.h"
#include "top_down_render/point_os1.h"

class TopDownRender {
  public:
    TopDownRender(ros::NodeHandle &nh);
    void initialize();
  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport *it_;
    tf2_ros::TransformBroadcaster *tf2_broadcaster_;
    ros::Subscriber pc_sub_;
    message_filters::Subscriber<pcl::PointCloud<PointType>> *pc_sync_sub_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> *motion_prior_sync_sub_;
    message_filters::TimeSynchronizer<pcl::PointCloud<PointType>, geometry_msgs::PoseStamped> *scan_sync_sub_;

    ros::Subscriber map_image_sub_;
    ros::Subscriber map_loc_sub_;

    ros::Subscriber gt_pose_sub_;
    ros::Publisher pose_pub_;
    ros::Publisher scale_pub_;
    image_transport::Publisher map_pub_;
    image_transport::Publisher scan_pub_;
    image_transport::Publisher geo_scan_pub_;
    image_transport::Publisher debug_pub_;

    Eigen::Affine2f gt_pose_;
    Eigen::Affine3f last_prior_pose_;
    cv::Point map_center_;
    cv::Mat color_lut_;
    Eigen::VectorXi flatten_lut_;
    cv::Mat background_img_;
    TopDownMapPolar *map_;
    ParticleFilter *filter_;
    ScanRendererPolar *renderer_;

    long last_map_stamp_;
    std::map<long, const sensor_msgs::Image::ConstPtr> map_image_buf_;
    std::map<long, const geometry_msgs::PointStamped::ConstPtr> map_loc_buf_;

    float map_pub_scale_ = 1;
    float conf_factor_ = 1;

    float current_res_ = 4; //m/px range
    bool is_converged_ = false;

    void publishSemanticTopDown(std::vector<Eigen::ArrayXXf> &top_down, std_msgs::Header &header);
    void publishGeometricTopDown(std::vector<Eigen::ArrayXXf> &top_down, std_msgs::Header &header);
    void publishLocalMap(int h, int w, Eigen::Vector2f center, float res, std_msgs::Header &header, TopDownMap *map);
    void visualize(std::vector<Eigen::ArrayXXf> &classes, cv::Mat &img);
    cv::Mat visualizeAnalog(Eigen::ArrayXXf &cls, float scale);
    void updateFilter(std::vector<Eigen::ArrayXXf> &top_down, 
                      std::vector<Eigen::ArrayXXf> &top_down_geo, float res, 
                      Eigen::Affine3f &motion_prior, std_msgs::Header &header);
    void pcCallback(const pcl::PointCloud<PointType>::ConstPtr&, 
                    const geometry_msgs::PoseStamped::ConstPtr &motion_prior);
    void mapImageCallback(const sensor_msgs::Image::ConstPtr &map);
    void mapLocCallback(const geometry_msgs::PointStamped::ConstPtr &map_loc);
    void processMapBuffers();
    void gtPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose);
};

#endif //TOP_DOWN_RENDER_H_
