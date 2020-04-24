#include "top_down_render/state_particle.h"

StateParticle::StateParticle(State s, float width, float height, TopDownMap *map) {
  state_ = s;
  map_ = map;
  width_ = width;
  height_ = height;
}

StateParticle::StateParticle(std::mt19937 *gen, float width, float height, TopDownMap *map) {
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

  map_->getLocalMap(center, state_.theta, res, classes);
  map_->getLocalGeoMap(center, state_.theta, res, geo_cls);

  float cost = 0;
  float normalization = 0;
  Eigen::ArrayXXf tmp;
  for (int i=0; i<map_->numClasses(); i++) {
    //semantic cost
    tmp = top_down_scan[i] * classes[i]*0.01;
    cost += tmp.sum();
    normalization += top_down_scan[i].sum();
  }
  for (int i=0; i<2; i++) {
    //geometric cost
    tmp = top_down_geo[i] * geo_cls[i]*0.001;
    cost += tmp.sum();
    normalization += top_down_scan[i].sum();
  }
  cost /= normalization;

  weight_ = 1/(cost+0.01); //Add epsilon to avoid divide-by-zero problems
}
