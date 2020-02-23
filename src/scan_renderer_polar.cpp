#include "top_down_render/scan_renderer_polar.h"

ScanRendererPolar::ScanRendererPolar(const Eigen::VectorXi &flatten_lut) : ScanRenderer(flatten_lut) {
}

void ScanRendererPolar::renderGeometricTopDown(const pcl::PointCloud<PointType>::ConstPtr& cloud, 
                                               float res, float ang_res, std::vector<Eigen::ArrayXXf> &imgs) {
  if (imgs.size() < 2) return;
  Eigen::Vector2i img_size(imgs[0].rows(), imgs[0].cols());

  for (int i=0; i<imgs.size(); i++) {
    imgs[i].setZero();
  }

  // Bin for each row in polar image.  Vector is xyzr 
  // Keep r to be able to rapidly sort later
  std::vector<std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>> ang_bins;
  ang_bins.reserve(img_size[0]);
  for (size_t i=0; i<img_size[0]; i++) {
    ang_bins.push_back(std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>());
  }

  // Populate the bins
  PointType pcl_pt;
  float theta, r;
  int theta_ind, r_ind;
  for (size_t idx=0; idx<cloud->width; idx++) {
    for (size_t idy=0; idy<cloud->height; idy++) {
      pcl_pt = cloud->at(idx, idy);
      if (pcl_pt.x == 0 && pcl_pt.y == 0) continue;

      theta = atan2(pcl_pt.x, pcl_pt.y);
      r = sqrt(pcl_pt.x*pcl_pt.x + pcl_pt.y*pcl_pt.y);

      // clamp to provide safely guarantees
      theta_ind = std::clamp<float>(
        std::round(theta/ang_res)+img_size[0]/2, 0, img_size[0]-1);
      
      ang_bins[theta_ind].push_back(Eigen::Vector4f(pcl_pt.x, pcl_pt.y, pcl_pt.z, r));
    }
  }

  float dist, slope;
  bool last_high_grad;
  int last_r_ind;
  theta_ind = 0;
  for (auto& bin : ang_bins) {
    // Sort each bin based on r
    std::sort(bin.begin(), bin.end(), [](Eigen::Vector4f& a, Eigen::Vector4f& b) {
      return a[3] > b[3];
    });

    // Loop through bin
    Eigen::Vector3f last_pt(0,0,0); 
    last_high_grad = false;
    last_r_ind = 0;
    for (const auto& pt : bin) {
      dist = (pt.head<3>()-last_pt).head<2>().norm(); //dist in xy plane
      slope = abs(pt[2]-last_pt[2])/dist;
      r_ind = std::round(pt[3]/res);

      if (slope > 1) {
        if (r_ind >= 0 && r_ind < img_size[1]) {
          imgs[1](theta_ind, r_ind) += 1;
        }
        last_high_grad = true;
      } else if (slope < 0.3 && last_high_grad == false) {
        for (int i=last_r_ind; i<=r_ind; i+=1) {
          if (i < img_size[1]) {
            imgs[0](theta_ind, i) += 1;
          }
        }
      } else {
        last_high_grad = false;
      }
      last_pt = pt.head<3>();
      last_r_ind = r_ind;
    }
    theta_ind++;
  }
}

void ScanRendererPolar::renderSemanticTopDown(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, 
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
      int pt_class = cloud->points[idx].intensity;
      if (flatten_lut_[pt_class] >= 0) {  
        imgs[flatten_lut_[pt_class]](theta_ind, r_ind) += 1;
      }
    }
  }
}

