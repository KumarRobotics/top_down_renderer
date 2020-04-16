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

float StateParticle::weight() {
  return weight_;
}

void StateParticle::computeWeight(Eigen::ArrayXXc &top_down_scan, Eigen::ArrayXXf &top_down_weights) {
  Eigen::Vector2f center(state_.x, state_.y);
  std::vector<Eigen::ArrayXXf> classes;
  for (int i=0; i<map_->numClasses(); i++) {
    classes.push_back(Eigen::ArrayXXf(top_down_scan.rows(), top_down_scan.cols()));
  }

  map_->getLocalMap(center, state_.theta, 1, classes);

  float cost = 0;
  Eigen::ArrayXXf tmp;
  for (int i=1; i<=map_->numClasses(); i++) {
    tmp = (top_down_scan == i).cast<float>() * classes[i-1] * top_down_weights;
    cost += tmp.sum();
  }
  cost /= top_down_weights.sum();

  weight_ = 1/(cost+0.01); //Add epsilon to avoid divide-by-zero problems
}
