#include "top_down_render/state_particle.h"

StateParticle::StateParticle(State s, float width, float height, TopDownMapPolar *map) {
  state_ = s;
  map_ = map;
  width_ = width;
  height_ = height;
}

StateParticle::StateParticle(std::mt19937 *gen, float width, float height, TopDownMapPolar *map) {
  std::uniform_real_distribution<float> dist(0.,1.);

  state_.x = dist(*gen)*width;
  state_.y = dist(*gen)*height;
  state_.theta = dist(*gen)*2*M_PI;
  map_ = map;
  width_ = width;
  height_ = height;
}

void StateParticle::propagate(std::mt19937 *gen) {
  std::normal_distribution<float> disp_dist{0, 0.5};
  std::normal_distribution<float> theta_dist{0, M_PI/30};
  
  state_.x = std::max(std::min(width_, state_.x+disp_dist(*gen)), static_cast<float>(0));
  state_.y = std::max(std::min(height_, state_.y+disp_dist(*gen)), static_cast<float>(0));
  state_.theta += theta_dist(*gen);
}

void StateParticle::setState(State s) {
  state_ = s;
}

State StateParticle::state() {
  return state_;
}

float StateParticle::weight() {
  return weight_;
}

float StateParticle::getCostForRot(std::vector<Eigen::ArrayXXf> &top_down_scan,
                                   std::vector<Eigen::ArrayXXf> &top_down_geo,
                                   std::vector<Eigen::ArrayXXf> &classes,
                                   std::vector<Eigen::ArrayXXf> &geo_cls, float rot) {
  //number of bins to shift by
  int num_bins = top_down_scan[0].rows();
  int rot_shift = static_cast<int>(std::round(rot*num_bins/2/M_PI));

  //normalize rotation
  while (rot_shift >= num_bins) rot_shift -= num_bins;
  while (rot_shift < 0) rot_shift += num_bins;

  float cost = 0;
  float normalization = 0;
  for (int i=0; i<map_->numClasses(); i++) {
    //semantic cost
    cost += (top_down_scan[i].bottomRows(rot_shift) * classes[i].topRows(rot_shift)).sum()*0.01;
    cost += (top_down_scan[i].topRows(num_bins-rot_shift) * classes[i].bottomRows(num_bins-rot_shift)).sum()*0.01;
    normalization += top_down_scan[i].sum();
  }
  for (int i=0; i<2; i++) {
    //geometric cost
    cost += (top_down_geo[i].bottomRows(rot_shift) * geo_cls[i].topRows(rot_shift)).sum()*0.001;
    cost += (top_down_geo[i].topRows(num_bins-rot_shift) * geo_cls[i].bottomRows(num_bins-rot_shift)).sum()*0.001;
    normalization += top_down_scan[i].sum();
  }

  return cost/normalization;
}

void StateParticle::computeWeight(std::vector<Eigen::ArrayXXf> &top_down_scan, 
                                  std::vector<Eigen::ArrayXXf> &top_down_geo, float res) {
  Eigen::Vector2f center(state_.x, state_.y);
  std::vector<Eigen::ArrayXXf> classes;
  for (int i=0; i<map_->numClasses(); i++) {
    classes.push_back(Eigen::ArrayXXf(top_down_scan[0].rows(), top_down_scan[0].cols()));
  }
  std::vector<Eigen::ArrayXXf> geo_cls;
  for (int i=0; i<2; i++) {
    geo_cls.push_back(Eigen::ArrayXXf(top_down_scan[0].rows(), top_down_scan[0].cols()));
  }

  map_->getLocalMap(center, classes);
  map_->getLocalGeoMap(center, geo_cls);

  float cost = getCostForRot(top_down_scan, top_down_geo, classes, geo_cls, state_.theta);

  weight_ = 1/(cost+0.01); //Add epsilon to avoid divide-by-zero problems
}
