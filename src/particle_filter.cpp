#include "top_down_render/particle_filter.h"

ParticleFilter::ParticleFilter(int N, float width, float height, TopDownMapPolar *map) {
  std::random_device rd;
  gen_ = new std::mt19937(rd());

  num_particles_ = N;
  num_gaussians_ = 1;
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
  
  //GMM
  std::vector<Eigen::Vector3f> means;
  std::vector<Eigen::Matrix3f> covs;
  computeGMM(means, covs);
  for (auto m : means) {
    ROS_INFO_STREAM(m);
  }

  //should really use eigenvalues instead of diagonal
  num_particles_ = 0;
  for (auto cov : covs) {
    Eigen::Vector2cf eig = cov.block<2,2>(0,0).eigenvalues(); //complex vector
    num_particles_ += static_cast<int>(sqrt(eig[0].real())*sqrt(eig[1].real()))*5; //Approximation of area of cov ellipse
  }
  num_particles_ = std::min(std::max(num_particles_, 10), 10000); //bounds
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

void ParticleFilter::computeGMM(std::vector<Eigen::Vector3f> &means, std::vector<Eigen::Matrix3f> &covs) {
  //We do this recursively: GMM in euclidean space, and then GMM on each cluster in theta space
  cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
  em->setClustersNumber(num_gaussians_);
  em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_GENERIC);

  //Build sample array
  int num_samples = std::min(500, static_cast<int>(particles_.size()));
  cv::Mat samples = cv::Mat(num_samples, 2, CV_64F);
  for (int i=0; i<num_samples; i+=1) {
    samples.at<double>(i, 0) = particles_[i*particles_.size()/num_samples]->mlState()[0];
    samples.at<double>(i, 1) = particles_[i*particles_.size()/num_samples]->mlState()[1];
  }

  cv::Mat likelihoods, labels, cluster_means;
  em->trainEM(samples, likelihoods, labels);
  float likelihood_mean = cv::mean(likelihoods)[0];

  int dir = 0;
  //try increasing
  if (num_gaussians_*50 < num_particles_) {
    em->setClustersNumber(num_gaussians_ + 1);
    em->trainEM(samples, likelihoods, labels);
    if (likelihood_mean + 0.2 < cv::mean(likelihoods)[0]) {
      dir = 1;
    }
  }
  //try decr
  if (num_gaussians_ > 1) {
    em->setClustersNumber(num_gaussians_ - 1);
    em->trainEM(samples, likelihoods, labels);
    if (likelihood_mean - 0.2 < cv::mean(likelihoods)[0]) {
      dir = -1;
    }
  }

  num_gaussians_ += dir;
  em->setClustersNumber(num_gaussians_);
  em->trainEM(samples, likelihoods, labels);
  cluster_means = em->getMeans();
  std::vector<cv::Mat> cluster_covs;
  em->getCovs(cluster_covs);

  //Convert to Eigen by manually copying elements
  for (int i=0; i<cluster_covs.size(); i++) {
    means.push_back(Eigen::Vector3f(cluster_means.at<double>(i,0), cluster_means.at<double>(i,1), 0));
    Eigen::Matrix3f cov;
    cov << cluster_covs[i].at<double>(0,0), cluster_covs[i].at<double>(0,1), 0,
           cluster_covs[i].at<double>(1,0), cluster_covs[i].at<double>(1,1), 0,
           0, 0, 1;
    covs.push_back(cov);
  }
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
