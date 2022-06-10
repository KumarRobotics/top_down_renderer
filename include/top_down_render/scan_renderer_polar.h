#ifndef SCAN_RENDERER_POLAR_H_
#define SCAN_RENDERER_POLAR_H_

#include <Eigen/Dense>
#include <ros/ros.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>

#include "top_down_render/scan_renderer.h"
#include "top_down_render/point_os1.h"

class ScanRendererPolar : public ScanRenderer {
  public:
    ScanRendererPolar(const Eigen::VectorXi &flatten_lut);
    void renderSemanticTopDown(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, 
                               float res, float ang_res, std::vector<Eigen::ArrayXXf> &imgs);
    void renderGeometricTopDown(const pcl::PointCloud<PointType>::ConstPtr& cloud, 
                                float res, float ang_res, std::vector<Eigen::ArrayXXf> &imgs);
};

#endif //SCAN_RENDERER_POLAR_H_
