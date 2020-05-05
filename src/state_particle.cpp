#include "top_down_render/state_particle.h"

StateParticle::StateParticle(std::mt19937 *gen, float width, float height, TopDownMapPolar *map) {
  gen_ = gen;
  std::uniform_real_distribution<float> dist(0.,1.);

  state_.x = dist(*gen)*width;
  state_.y = dist(*gen)*height;
  state_.theta_particles = std::make_shared<std::vector<ThetaParticle>>();
  for (int i=0; i<30; i++) {
    ThetaParticle p;
    p.theta = dist(*gen)*2*M_PI;
    p.weight = 1;
    state_.theta_particles->push_back(p);
  }
  ml_theta_ = 0;

  map_ = map;
  width_ = width;
  height_ = height;
}

void StateParticle::propagate() {
  std::normal_distribution<float> disp_dist{0, 0.5};
  std::normal_distribution<float> theta_dist{0, M_PI/30};
  
  state_.x = std::max(std::min(width_, state_.x+disp_dist(*gen_)), static_cast<float>(0));
  state_.y = std::max(std::min(height_, state_.y+disp_dist(*gen_)), static_cast<float>(0));

  for (int i=0; i<state_.theta_particles->size(); i++) {
    (*state_.theta_particles)[i].theta += theta_dist(*gen_);
  }
}

void StateParticle::setState(State s) {
  state_.x = s.x;
  state_.y = s.y;
  *state_.theta_particles = *s.theta_particles; //copy contents
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
    cost += (top_down_scan[i].bottomRows(rot_shift) * classes[i].topRows(rot_shift)).sum()*0.01;
    cost += (top_down_scan[i].topRows(num_bins-rot_shift) * classes[i].bottomRows(num_bins-rot_shift)).sum()*0.01;
    normalization += top_down_scan[i].sum();
  }
  for (int i=0; i<2; i++) {
    //geometric cost
    cost += (top_down_geo[i].bottomRows(rot_shift) * geo_cls[i].topRows(rot_shift)).sum()*0.001;
    cost += (top_down_geo[i].topRows(num_bins-rot_shift) * geo_cls[i].bottomRows(num_bins-rot_shift)).sum()*0.001;
    normalization += top_down_geo[i].sum();
  }

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
    (*state_.theta_particles)[i].weight = 1/(cost+0.01);

    //Particle weight is based on best angle
    if ((*state_.theta_particles)[i].weight > weight_) {
      weight_ = (*state_.theta_particles)[i].weight;
      ml_theta_ = (*state_.theta_particles)[i].theta;
    }
  }
  
  float cov = thetaCov();
  resampleParticles(cov*10);
}
