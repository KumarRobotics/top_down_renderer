#include "top_down_render/particle_filter.h"

ParticleFilter::ParticleFilter(int N, float width, float height, TopDownMap *map) {
  std::random_device rd;
  gen_ = new std::mt19937(rd());

  num_particles_ = N;
  map_ = map;

  //Weights should be even
  weights_ = Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(num_particles_)/num_particles_;
  for (int i=0; i<num_particles_; i++) {
    StateParticle *particle = new StateParticle(gen_, width, height, map);
    particles_.push_back(particle);

    //Allocate memory for new array too, then we can swap back and forth without allocating
    StateParticle *new_particle = new StateParticle(gen_, width, height, map);
    new_particles_.push_back(new_particle);
  }
}

//In the future this function should take in a motion prior
void ParticleFilter::propagate() {
  for (auto p : particles_) {
    p->propagate(gen_);
  }
}

void ParticleFilter::update(std::vector<Eigen::ArrayXXf> &top_down_scan) {
  //Recompute weights
  std::for_each(std::execution::par, particles_.begin(), particles_.end(), 
                std::bind(&StateParticle::computeWeight, std::placeholders::_1, top_down_scan));
  for (int i; i<particles_.size(); i++) {
    weights_[i] = particles_[i]->weight();
  }
  weights_ = weights_/weights_.sum(); //Renormalize

  ROS_INFO_STREAM("particle reweighting complete");
  
  //Resample
  for (int i=0; i<particles_.size(); i++) {
    float running_sum = 0;
    float sample = static_cast<float>(i)/particles_.size();
    int j=0;
    for (;; j++) {
      running_sum += weights_[j];
      if (running_sum > sample || j>=particles_.size()) {
        if (j == particles_.size()) j--;
        break; 
      }
    }
    new_particles_[i]->setState(particles_[j]->state());
  }
  particles_.swap(new_particles_);
}

void ParticleFilter::visualize(cv::Mat &img) {
  for (auto p : particles_) {
    cv::Point pt(p->state().x*map_->scale(), img.size().height-p->state().y*map_->scale());
    cv::circle(img, pt, 3, cv::Scalar(0,0,255), -1);
  }
}
