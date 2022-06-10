#include "top_down_render/scan_renderer.h"

ScanRenderer::ScanRenderer(const Eigen::VectorXi &flatten_lut) {
  flatten_lut_ = flatten_lut;
}

void ScanRenderer::renderGeometricTopDown(const pcl::PointCloud<PointType>::ConstPtr& cloud, 
                                          float res, std::vector<Eigen::ArrayXXf> &imgs) {
  if (imgs.size() < 2) return;
  Eigen::Vector2i img_size(imgs[0].cols(), imgs[0].rows());

  for (int i=0; i<imgs.size(); i++) {
    imgs[i].setZero();
  }

  for (size_t idx=0; idx<cloud->width; idx++) {
    Eigen::Vector3f last_pt(0,0,0); 
    Eigen::Vector3f pt(0,0,0); 
    Eigen::Vector2i last_ind = img_size/2;
    bool last_high_grad = false;

    //Scan up a vertical scan line
    for (size_t idy=0; idy<cloud->height; idy++) {
      PointType pcl_pt = cloud->at(idx, idy);
      pt << pcl_pt.x, pcl_pt.y, pcl_pt.z;
      if (pt[0] == 0 && pt[1] == 0) continue;
      int x_ind = std::round(pt[0]/res)+img_size[0]/2;
      int y_ind = std::round(pt[1]/res)+img_size[1]/2;

      float dist = (pt-last_pt).head<2>().norm(); //dist in xy plane
      float slope = abs(pt(2)-last_pt(2))/dist;
      if (slope > 1) {
        if (x_ind >= 0 && x_ind < img_size[0] && y_ind >= 0 && y_ind < img_size[1]) {
          imgs[1](y_ind, x_ind) += 1;
        }
        last_high_grad = true;
      } else if (slope < 0.3 && last_high_grad == false) {
        Eigen::Vector2i diff = Eigen::Vector2i(x_ind, y_ind)-last_ind;
        for (float i=0; i<1; i+=1./diff.norm()) {
          Eigen::Vector2i interp_ind(round(last_ind[0]+i*diff[0]), round(last_ind[1]+i*diff[1]));
          if (interp_ind[0] >= 0 && interp_ind[0] < img_size[0] && interp_ind[1] >= 0 && 
              interp_ind[1] < img_size[1]) {
            imgs[0](interp_ind[1], interp_ind[0]) += 1;
          }
        }
      } else {
        last_high_grad = false;
      }
      last_pt = pt;
      last_ind << x_ind, y_ind;
    }
  }
}

void ScanRenderer::renderSemanticTopDown(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, 
                                         float res, std::vector<Eigen::ArrayXXf> &imgs) {
  if (imgs.size() < 1) return;
  Eigen::Vector2i img_size(imgs[0].cols(), imgs[0].rows());

  for (int i=0; i<imgs.size(); i++) {
    imgs[i].setZero();
  }

  //Generate bins of points
  for (size_t idx=0; idx<cloud->height*cloud->width; idx++) {
    auto pt = cloud->points[idx];
    if (pt.x == 0 && pt.y == 0) continue;

    int x_ind = std::round(pt.x/res)+img_size[0]/2;
    int y_ind = std::round(pt.y/res)+img_size[1]/2;
    if (x_ind >= 0 && x_ind < img_size[0] && y_ind >= 0 && y_ind < img_size[1]) {
      int pt_class = cloud->points[idx].intensity;
      imgs[flatten_lut_[pt_class]-1](y_ind, x_ind)++;
    }
  }
}

