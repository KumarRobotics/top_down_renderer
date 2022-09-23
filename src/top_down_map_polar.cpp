#include "top_down_render/top_down_map_polar.h"

TopDownMapPolar::TopDownMapPolar(const Params& params) : TopDownMap(params) {
  samplePtsPolar(Eigen::Vector2i(100,50), 2*M_PI/100);
}

void TopDownMapPolar::samplePtsPolar(Eigen::Vector2i shape, float ang_res) {
  Eigen::Array2Xf ang_pts(2, shape[0]*shape[1]);
  ang_sample_pts_ = Eigen::Array2Xf(2, shape[0]*shape[1]);
  samplePts(Eigen::Vector2f::Zero(), 0, ang_pts, shape[1], shape[0], 1);
  ang_pts.row(1) += -ang_pts.row(1)[0]; //shift to start at 0

  ang_pts.row(0) *= ang_res;
  ang_pts.row(1) *= 1./params_.resolution;

  //convert to cartesian
  ang_sample_pts_.row(0) = Eigen::cos(ang_pts.row(0))*ang_pts.row(1); //x
  ang_sample_pts_.row(1) = Eigen::sin(ang_pts.row(0))*ang_pts.row(1); //y
}

void TopDownMapPolar::getLocalMap(Eigen::Vector2f center, 
                                  float scale, float res, 
                                  std::vector<Eigen::ArrayXXf> &dists) {
  if (dists.size() < 1) return;

  //Generate list of indices
  Eigen::Array2Xf pts = ang_sample_pts_*scale*res;
  pts.row(0) += center[1]/params_.resolution;
  pts.row(1) += center[0]/params_.resolution;
  Eigen::Array2Xi pts_int = pts.round().cast<int>();

  for (int cls=0; cls<dists.size(); cls++) {
    for (int idx=0; idx<dists[0].rows()*dists[0].cols(); idx++) {
      if (pts_int(0, idx) >= 0 && pts_int(0, idx) < class_maps_[cls].rows() &&
          pts_int(1, idx) >= 0 && pts_int(1, idx) < class_maps_[cls].cols()) {
        dists[cls](idx) = class_maps_[cls](pts_int(0, idx), pts_int(1, idx));
      } else {
        dists[cls](idx) = params_.out_of_bounds_const;
      }
    }
  }
}

void TopDownMapPolar::getLocalGeoMap(Eigen::Vector2f center, 
                                     float scale, float res,
                                     std::vector<Eigen::ArrayXXf> &dists) {
  if (dists.size() < 1) return;

  //Generate list of indices
  Eigen::Array2Xf pts = ang_sample_pts_*scale*res;
  pts.row(0) += center[1]/params_.resolution;
  pts.row(1) += center[0]/params_.resolution;
  Eigen::Array2Xi pts_int = pts.round().cast<int>();

  for (int cls=0; cls<dists.size(); cls++) {
    for (int idx=0; idx<dists[0].rows()*dists[0].cols(); idx++) {
      if (pts_int(0, idx) >= 0 && pts_int(0, idx) < geo_maps_[cls].rows() &&
          pts_int(1, idx) >= 0 && pts_int(1, idx) < geo_maps_[cls].cols()) {
        dists[cls](idx) = geo_maps_[cls](pts_int(0, idx), pts_int(1, idx));
      } else {
        dists[cls](idx) = params_.out_of_bounds_const;
      }
    }
  }
}

void TopDownMapPolar::getLocalMap(Eigen::Vector2f center, float res,
                                  std::vector<Eigen::ArrayXXf> &dists) {
  getLocalMap(center, 1, res, dists);
}

void TopDownMapPolar::getLocalGeoMap(Eigen::Vector2f center, float res,
                                     std::vector<Eigen::ArrayXXf> &dists) {
  getLocalGeoMap(center, 1, res, dists);
}

