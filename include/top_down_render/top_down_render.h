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
#include <grid_map_msgs/GridMap.h>
#include <grid_map_comp/grid_map_comp.hpp>
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

#include "semantics_manager/semantics_manager.h"
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
    ros::Subscriber motion_prior_sub_;

    ros::Subscriber aerial_map_sub_;

    ros::Subscriber gt_pose_sub_;
    ros::Publisher pose_pub_;
    ros::Publisher scale_pub_;
    image_transport::Publisher map_viz_pub_;
    image_transport::Publisher scan_pub_;
    image_transport::Publisher geo_scan_pub_;
    image_transport::Publisher debug_pub_;

    Eigen::Affine2f gt_pose_;
    Eigen::Affine3f last_prior_pose_;
    cv::Point map_center_;
    SemanticColorLut color_lut_;
    Eigen::VectorXi flatten_lut_;
    std::vector<int> unflatten_lut_;
    cv::Mat background_img_;
    TopDownMapPolar *map_;
    ParticleFilter *filter_;
    ScanRendererPolar *renderer_;

    long last_map_stamp_ = 0;
    std::list<geometry_msgs::PoseStamped::ConstPtr> motion_prior_buf_;

    std::string map_frame_ = "map";
    std::string map_viz_frame_ = "sem_map";
    bool use_motion_prior_ = false;

    float map_pub_scale_ = 1;
    float conf_factor_ = 1;
    float range_scale_max_ = 4;
    float range_scale_min_ = 0.5;
    float target_uncertainty_m_ = 2.5;

    float current_range_scale_ = 4; //m/px range
    bool is_converged_ = false;
    bool published_pose_ = false;

    TopDownMap::Params getTopDownMapParams(
        const semantics_manager::ClassConfig& class_params,
        const semantics_manager::MapConfig& map_params);
    FilterParams getFilterParams(
        const semantics_manager::ClassConfig& class_params,
        const semantics_manager::MapConfig& map_params);

    void publishSemanticTopDown(std::vector<Eigen::ArrayXXf> &top_down, const std_msgs::Header &header);
    void publishGeometricTopDown(std::vector<Eigen::ArrayXXf> &top_down, const std_msgs::Header &header);
    void publishLocalMap(int h, int w, Eigen::Vector2f center, float res, const std_msgs::Header &header, TopDownMap *map);
    void publishPoseEst(const std_msgs::Header &header);
    void visualize(std::vector<Eigen::ArrayXXf> &classes, cv::Mat &img);
    cv::Mat visualizeAnalog(Eigen::ArrayXXf &cls, float scale);
    void updateFilter(std::vector<Eigen::ArrayXXf> &top_down, 
                      std::vector<Eigen::ArrayXXf> &top_down_geo, float res, 
                      Eigen::Affine3f &motion_prior, const std_msgs::Header &header);
    void pcCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg);
    void motionPriorCallback(const geometry_msgs::PoseStamped::ConstPtr &motion_prior); 
    void takeStep(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg, 
                  const geometry_msgs::PoseStamped::ConstPtr &motion_prior);
    void aerialMapCallback(const grid_map_msgs::GridMap::ConstPtr &map);
    void gtPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose);
};

#endif //TOP_DOWN_RENDER_H_
