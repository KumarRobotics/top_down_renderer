#include "top_down_render/particle_filter.h"

ParticleFilter::ParticleFilter(int N, float width, float height, TopDownMapPolar *map) {
  std::random_device rd;
  gen_ = new std::mt19937(rd());

  num_particles_ = N;
  map_ = map;
  width_ = width;
  height_ = height;

  //Weights should be even
  weights_ = Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(num_particles_)/num_particles_;
  for (int i=0; i<num_particles_; i++) {
    std::shared_ptr<StateParticle> particle = std::make_shared<StateParticle>(gen_, width, height, map);
    particles_.push_back(particle);

    //Allocate memory for new array too, then we can swap back and forth without allocating
    std::shared_ptr<StateParticle> new_particle = std::make_shared<StateParticle>(gen_, width, height, map);
    new_particles_.push_back(new_particle);
  }
  max_likelihood_particle_ = particles_[0];
}

//In the future this function should take in a motion prior
void ParticleFilter::propagate() {
  for (auto p : particles_) {
    p->propagate();
  }
}

void ParticleFilter::update(std::vector<Eigen::ArrayXXf> &top_down_scan, 
                            std::vector<Eigen::ArrayXXf> &top_down_geo, float res) {
  //Recompute weights
  std::for_each(std::execution::par, particles_.begin(), particles_.end(), 
                std::bind(&StateParticle::computeWeight, std::placeholders::_1, top_down_scan, top_down_geo, res));

  weights_ = Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(num_particles_)/num_particles_;
  for (int i; i<particles_.size(); i++) {
    weights_[i] = particles_[i]->weight();
  }
  weights_ = weights_/weights_.sum(); //Renormalize

  //Find the maximum likelihood particle
  Eigen::VectorXf::Index max_weight;
  weights_.maxCoeff(&max_weight);
  max_likelihood_particle_ = particles_[max_weight];

  ROS_INFO_STREAM("particle reweighting complete");

  Eigen::Matrix2f cov;
  computeCov(cov);
  num_particles_ = std::min(static_cast<int>(sqrt(cov(0,0))*sqrt(cov(1,1)))*5, 10000); //Approximation of area of cov ellipse
  ROS_INFO_STREAM(num_particles_ << " particles");

  //resize num_particles_
  if (num_particles_ < new_particles_.size()) {
    new_particles_.resize(num_particles_);
  } else {
    while (new_particles_.size() < num_particles_) {
      new_particles_.push_back(std::make_shared<StateParticle>(gen_, width_, height_, map_));
    }
  }
  
  //Resample
  std::uniform_real_distribution<float> shift_dist(0.,1.);
  float shift = shift_dist(*gen_); //Add a random shift
  for (int i=0; i<num_particles_; i++) {
    float running_sum = 0;
    float sample = (static_cast<float>(i)+shift)/num_particles_;
    int j=0;
    for (; j<particles_.size(); j++) {
      running_sum += weights_[j];
      if (running_sum > sample || j == particles_.size()-1) {
        break; 
      }
    }
    new_particles_[i]->setState(particles_[j]->state());
  }
  particles_.swap(new_particles_);
}

void ParticleFilter::computeCov(Eigen::Matrix2f &cov) {
  cov.setZero();
  Eigen::Vector2f max_likelihood_state = max_likelihood_particle_->mlState().head<2>();
  for (auto particle : particles_) {
    Eigen::Vector2f state = particle->mlState().head<2>() - max_likelihood_state;
    cov += state*state.transpose();
  }
  cov /= particles_.size()-1;
}

void ParticleFilter::visualize(cv::Mat &img) {
  //Particle dist
  for (auto p : particles_) {
    Eigen::Vector2f state = p->mlState().head<2>();
    cv::Point pt(state[0]*map_->scale(), img.size().height-state[1]*map_->scale());
    cv::circle(img, pt, 3, cv::Scalar(0,0,255), -1);
  }
  //Max likelihood
  Eigen::Vector3f ml_state = max_likelihood_particle_->mlState();
  cv::Point pt(ml_state[0]*map_->scale(), 
               img.size().height-ml_state[1]*map_->scale());
  cv::Point dir(cos(ml_state[2])*5, sin(ml_state[2])*5);
  cv::arrowedLine(img, pt-dir, pt+dir, cv::Scalar(255,0,0), 2, CV_AA, 0, 0.3);
}
