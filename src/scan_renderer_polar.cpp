#include "top_down_render/scan_renderer_polar.h"

ScanRendererPolar::ScanRendererPolar(bool nf) : ScanRenderer(nf) {
}

void ScanRendererPolar::renderGeometricTopDown(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud, 
                                               float res, float ang_res, std::vector<Eigen::ArrayXXf> &imgs) {
  if (imgs.size() < 2) return;
  Eigen::Vector2i img_size(imgs[0].rows(), imgs[0].cols());

  for (int i=0; i<imgs.size(); i++) {
    imgs[i].setZero();
  }

  for (size_t idx=0; idx<cloud->width; idx++) {
    Eigen::Vector3f last_pt(0,0,0); 
    Eigen::Vector3f pt(0,0,0); 
    Eigen::Vector2i last_ind(img_size[0]/2, 0);
    bool last_high_grad = false;

    //Scan up a vertical scan line
    for (size_t idy=0; idy<cloud->height; idy++) {
      pcl::PointXYZRGB pcl_pt = cloud->at(idx, idy);
      pt << pcl_pt.x, pcl_pt.y, pcl_pt.z;
      if (pt[0] == 0 && pt[1] == 0) continue;
      //Convert to polar
      float theta = atan2(pt[0], pt[1]);
      float r = sqrt(pt[0]*pt[0] + pt[1]*pt[1]);

      int theta_ind = std::round(theta/ang_res)+img_size[0]/2;
      int r_ind = std::round(r/res);

      float dist = (pt-last_pt).head<2>().norm(); //dist in xy plane
      float slope = abs(pt(2)-last_pt(2))/dist;
      if (slope > 1) {
        if (theta_ind >= 0 && theta_ind < img_size[0] && r_ind >= 0 && r_ind < img_size[1]) {
          imgs[1](theta_ind, r_ind) += 1;
        }
        last_high_grad = true;
      } else if (slope < 0.3 && last_high_grad == false) {
        Eigen::Vector2i diff = Eigen::Vector2i(theta_ind, r_ind)-last_ind;
        for (float i=0; i<1; i+=1./diff.norm()) {
          Eigen::Vector2i interp_ind(round(last_ind[0]+i*diff[0]), round(last_ind[1]+i*diff[1]));
          if (interp_ind[0] >= 0 && interp_ind[0] < img_size[0] && interp_ind[1] >= 0 && 
              interp_ind[1] < img_size[1]) {
            imgs[0](interp_ind[0], interp_ind[1]) += 1;
          }
        }
      } else {
        last_high_grad = false;
      }
      last_pt = pt;
      last_ind << theta_ind, r_ind;
    }
  }
}

void ScanRendererPolar::renderSemanticTopDown(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud, 
                                              pcl::PointCloud<pcl::Normal>::Ptr& normals, 
                                              float res, float ang_res, std::vector<Eigen::ArrayXXf> &imgs) {
  if (imgs.size() < 1) return;
  Eigen::Vector2i img_size(imgs[0].rows(), imgs[0].cols());

  for (int i=0; i<imgs.size(); i++) {
    imgs[i].setZero();
  }

  //Generate bins of points
  for (size_t idx=0; idx<cloud->height*cloud->width; idx++) {
    auto pt = cloud->points[idx];
    if (pt.x == 0 && pt.y == 0) continue;
    //Convert to polar
    float theta = atan2(pt.x, pt.y);
    float r = sqrt(pt.x*pt.x + pt.y*pt.y);

    int theta_ind = std::round(theta/ang_res)+img_size[0]/2;
    int r_ind = std::round(r/res);
    if (theta_ind >= 0 && theta_ind < img_size[0] && r_ind >= 0 && r_ind < img_size[1]) {
      if (!normal_filter_ || normals->points[idx].normal_z < 0.9) {
        int pt_class = *reinterpret_cast<const int*>(&cloud->points[idx].rgb) & 0xff;
        imgs[flatten_lut_[pt_class]-1](theta_ind, r_ind)++;
      }
    }
  }
}

