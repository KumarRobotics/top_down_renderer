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
  std::normal_distribution<float> disp_dist{0, 1};
  std::normal_distribution<float> theta_dist{0, M_PI/16};
  
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

float StateParticle::computeWeight(Eigen::ArrayXXc &top_down_scan) {
  Eigen::Vector2f center(state_.x, state_.y);
  Eigen::ArrayXXc cls(top_down_scan.rows(), top_down_scan.cols());

  map_->getLocalMap(center, state_.theta, 1, cls);
  Eigen::ArrayXXc diff = cls.cwiseNotEqual(top_down_scan).cast<uint8_t>() * top_down_scan;
  float cost = static_cast<float>(diff.count())/top_down_scan.count();

  return 1/(cost+0.01); //Add epsilon to avoid divide-by-zero problems
}
