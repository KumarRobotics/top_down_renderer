#ifndef TOP_DOWN_RENDER_H_
#define TOP_DOWN_RENDER_H_

#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <chrono>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include "top_down_render/point_xyz_class_normal.h"
#include "top_down_render/top_down_map.h"
#include "top_down_render/particle_filter.h"

class TopDownRender {
  public:
    TopDownRender(ros::NodeHandle &nh);
    void initialize();
  private:
    ros::NodeHandle nh_;
		image_transport::ImageTransport *it_;
    ros::Subscriber pc_sub_;
		image_transport::Publisher img_pub_;
		image_transport::Publisher scan_pub_;
		image_transport::Publisher map_pub_;

    cv::Mat flatten_lut_;
    cv::Mat color_lut_;
    cv::Mat background_img_;
    TopDownMap *map_;
    ParticleFilter *filter_;

    bool normal_filter_ = true;

    void publishTopDown(std::vector<Eigen::ArrayXXf> &top_down, std_msgs::Header &header);
    void publishLocalMap(int h, int w, Eigen::Vector2f center, float res, std_msgs::Header &header);
		void renderTopDown(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud, 
											 pcl::PointCloud<pcl::Normal>::Ptr& normals,	
											 float side_length, std::vector<Eigen::ArrayXXf> &imgs);
    void visualize(std::vector<Eigen::ArrayXXf> &classes, cv::Mat &img);
    void updateFilter(std::vector<Eigen::ArrayXXf> &top_down, std_msgs::Header &header);
    void pcCallback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&);
};

#endif //TOP_DOWN_RENDER_H_
