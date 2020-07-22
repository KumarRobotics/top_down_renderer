#include "top_down_render/state_particle.h"

StateParticle::StateParticle(std::mt19937 *gen, float width, float height, TopDownMapPolar *map, FilterParams &params) {
  params_ = params;
  gen_ = gen;
  std::uniform_real_distribution<float> dist(0.,1.);

  std::vector<int> cls_vec;
  while (true) {
    state_.x = dist(*gen)*width;
    state_.y = dist(*gen)*height;
    map->getClassesAtPoint(Eigen::Vector2f(state_.x, state_.y), cls_vec);
    if (std::find(cls_vec.begin(), cls_vec.end(), 1) != cls_vec.end()) {
      break; //Particle is on the road
    }
  }

  state_.theta_particles = std::make_shared<std::vector<ThetaParticle>>();
  for (int i=0; i<30; i++) {
    ThetaParticle p;
    p.theta = i*2*M_PI/30;
    p.weight = 1;
    state_.theta_particles->push_back(p);
  }
  ml_theta_ = 0;

  map_ = map;
  width_ = width;
  height_ = height;

  class_weights_.push_back(0.75); //terrain
  class_weights_.push_back(2); //road
  class_weights_.push_back(1.5); //dirt road
  class_weights_.push_back(1.5); //building
  class_weights_.push_back(0.5); //trees
}

void StateParticle::propagate(Eigen::Vector2f &trans, float omega) {
  Eigen::Vector2f trans_global = Eigen::Rotation2D<float>(ml_theta_) * trans;
  state_.x += trans_global[0];
  state_.y += trans_global[1];

  //std::normal_distribution<float> disp_dist{0, 0.5};
  //std::normal_distribution<float> theta_dist{0, M_PI/30};
  std::normal_distribution<float> disp_dist{0, params_.pos_cov};
  std::normal_distribution<float> theta_dist{0, params_.theta_cov};
  
  state_.x = std::max(std::min(width_, state_.x+disp_dist(*gen_)), static_cast<float>(0));
  state_.y = std::max(std::min(height_, state_.y+disp_dist(*gen_)), static_cast<float>(0));

  for (int i=0; i<state_.theta_particles->size(); i++) {
    (*state_.theta_particles)[i].theta += theta_dist(*gen_) + omega;
  }
  ml_theta_ += omega;
}

void StateParticle::setState(State s) {
  state_.x = s.x;
  state_.y = s.y;
  state_.theta_particles->clear();
  float max_weight = 0;
  for (auto p : *s.theta_particles) {
    state_.theta_particles->push_back(p);
    if (max_weight < p.weight) {
      max_weight = p.weight;
      ml_theta_ = p.theta;
    }
  }
}

State StateParticle::state() {
  return state_;
}

Eigen::Vector3f StateParticle::mlState() {
  return Eigen::Vector3f(state_.x, state_.y, ml_theta_);
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
    cost += (top_down_scan[i].topRows(rot_shift) * classes[i].bottomRows(rot_shift)).sum()*0.01*class_weights_[i];
    cost += (top_down_scan[i].bottomRows(num_bins-rot_shift) * classes[i].topRows(num_bins-rot_shift)).sum()*0.01*class_weights_[i];
    normalization += top_down_scan[i].sum();
  }
  /*
  for (int i=0; i<2; i++) {
    //geometric cost
    cost += (top_down_geo[i].topRows(rot_shift) * geo_cls[i].bottomRows(rot_shift)).sum()*0.001;
    cost += (top_down_geo[i].bottomRows(num_bins-rot_shift) * geo_cls[i].topRows(num_bins-rot_shift)).sum()*0.001;
    normalization += top_down_geo[i].sum();
  }
  */

  return cost/normalization;
}

float StateParticle::thetaCov() {
  float sum = 0;
  for (auto p : *state_.theta_particles) {
    float diff = abs(p.theta - ml_theta_);
    while (diff > M_PI) diff -= 2*M_PI;
    sum += diff*diff;
  }
  return sum/state_.theta_particles->size();
}

void StateParticle::resampleParticles(int num_part) {
  //Minimum 4 particles
  if (num_part < 4) num_part = 4;

  float weight_sum = 0;
  for (int i=0; i<state_.theta_particles->size(); i++) {
    weight_sum += (*state_.theta_particles)[i].weight;
  }
  std::vector<ThetaParticle> new_part;

  std::uniform_real_distribution<float> shift_dist(0.,1.);
  float shift = shift_dist(*gen_); //Add a random shift
  for (int i=0; i<num_part; i++) {
    float running_sum = 0;
    float sample = weight_sum*(static_cast<float>(i)+shift)/num_part;
    int j=0;
    for (; j<state_.theta_particles->size(); j++) {
      running_sum += (*state_.theta_particles)[j].weight;
      if (running_sum > sample || j == state_.theta_particles->size()-1) {
        break; 
      }
    }
    new_part.push_back((*state_.theta_particles)[j]);
  }

  (*state_.theta_particles) = new_part;
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

  map_->getLocalMap(center, res, classes);
  map_->getLocalGeoMap(center, res, geo_cls);

  float cost = 0;
  weight_ = 0;
  for (int i=0; i<state_.theta_particles->size();  i++) {
    cost = getCostForRot(top_down_scan, top_down_geo, classes, geo_cls, (*state_.theta_particles)[i].theta);
    (*state_.theta_particles)[i].weight = 1/(cost + params_.regularization);

    //Particle weight is based on best angle
    if ((*state_.theta_particles)[i].weight > weight_) {
      weight_ = (*state_.theta_particles)[i].weight;
      ml_theta_ = (*state_.theta_particles)[i].theta;
    }
  }
  
  float cov = thetaCov();
  resampleParticles(cov*10);
}
