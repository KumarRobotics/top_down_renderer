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

typedef std::priority_queue<PointXYZClassNormal, 
                            std::vector<PointXYZClassNormal>, 
                            std::less<PointXYZClassNormal>> pt_q_type;

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
    std::vector<std::vector<pt_q_type>> org_pc_;
    TopDownMap *map_;
    ParticleFilter *filter_;

    void publishTopDown(cv::Mat& top_down_img, std_msgs::Header &header);
    void publishHeatMap(Eigen::ArrayXXc &top_down, float local_res, float heatmap_res, cv::Rect roi, std_msgs::Header &header);
		void renderTopDown(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud, 
											 pcl::PointCloud<pcl::Normal>::Ptr& normals,	
											 float side_length, Eigen::ArrayXXc &img);
    void updateFilter(Eigen::ArrayXXc &top_down, std_msgs::Header &header);
    void pcCallback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&);
};

#endif //TOP_DOWN_RENDER_H_
