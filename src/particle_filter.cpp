#include "top_down_render/particle_filter.h"

ParticleFilter::ParticleFilter(int N, TopDownMapPolar *map, FilterParams &params) {
  std::random_device rd;
  gen_ = new std::mt19937(rd());

  num_gaussians_ = 1;
  map_ = map;
  params_ = params;
  max_num_particles_ = N;
  num_particles_ = 0;
  last_map_center_ = Eigen::Vector2i::Zero();

  if (map_->haveMap()) {
    initializeParticles();
  }
}

void ParticleFilter::initializeParticles() {
  size_t num_at_scale = 1;
  if (params_.fixed_scale < 0) {
    num_at_scale = 10;
  } else {
    scale_frozen_ = true;
  }

  if (scale_frozen_ && params_.init_pos_m_x != std::numeric_limits<float>::infinity()) {
    Eigen::Vector2i map_center = map_->mapCenter();
    params_.init_pos_px_x = (params_.init_pos_m_x*params_.fixed_scale) + map_center.x();
    params_.init_pos_px_y = (params_.init_pos_m_y*params_.fixed_scale) + map_center.y();

    if (params_.init_pos_px_x < 0 || params_.init_pos_px_x >= map_->size()[0] ||
        params_.init_pos_px_y < 0 || params_.init_pos_px_y >= map_->size()[1]) {
      ROS_WARN("No map received for input loc");
      return;
    } 

    bool good_init = false;
    for (int dx=-4; dx<=4; dx++) {
      for (int dy=-4; dy<=4; dy++) {
        std::vector<int> cls_vec;
        map_->getClassesAtPoint(Eigen::Vector2i(params_.init_pos_px_x+dx, params_.init_pos_px_y+dy), cls_vec);
        if (std::find(cls_vec.begin(), cls_vec.end(), 1) != cls_vec.end()) {
          good_init = true;
          break;
        }
      }
    }
    if (!good_init) {
      ROS_WARN("No road in map at init location");
      return;
    }
  }

  //Weights should be even
  ROS_INFO_STREAM("Initializing particles...");
  for (int i=0; i<max_num_particles_/num_at_scale; i++) {
    StateParticle proto_part(gen_, map_, &params_);
    for (float scale=0; scale<1; scale+=1./num_at_scale) {
      std::shared_ptr<StateParticle> particle = std::make_shared<StateParticle>(gen_, map_, &params_);
      if (params_.fixed_scale < 0) {
        particle->setState(proto_part.state());
        particle->setScale(std::pow(10., scale));
      }
      particles_.push_back(particle);

      //Allocate memory for new array too, then we can swap back and forth without allocating
      std::shared_ptr<StateParticle> new_particle = std::make_shared<StateParticle>(gen_, map_, &params_);
      new_particles_.push_back(new_particle);
    }
  }
  max_likelihood_particle_ = particles_[0];

  num_particles_ = particles_.size();
  weights_ = Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(num_particles_)/num_particles_;

  //best_rel_pos_ = Eigen::Vector2f(0,0);
  //active_loc_ = new ActiveLocalizer(map_);

  //Initialize Gaussians
  computeGMM();
  gmm_thread_ = new std::thread(std::bind(&ParticleFilter::gmmThread, this));
  ROS_INFO_STREAM("Particles initialized");
}

void ParticleFilter::propagate(Eigen::Vector2f &trans, float omega) {
  particle_lock_.lock();
  for (auto& p : particles_) {
    p->propagate(trans, omega, scale_frozen_);
  }
  particle_lock_.unlock();
}

void ParticleFilter::update(std::vector<Eigen::ArrayXXf> &top_down_scan, 
                            std::vector<Eigen::ArrayXXf> &top_down_geo, float res) {
  if (num_particles_ == 0 && map_->haveMap()) {
    //threads haven't started yet so don't need lock
    initializeParticles();
    if (num_particles_ == 0) {
      //If still no particles, then nevermind, try again later
      return;
    }
  }

  particle_lock_.lock();
  ROS_INFO_STREAM("computing weights...");
  //Recompute weights
  std::for_each(std::execution::par, particles_.begin(), particles_.end(), 
                std::bind(&StateParticle::computeWeight, std::placeholders::_1, top_down_scan, top_down_geo, res));

  weights_ = Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(particles_.size())/particles_.size();
  for (int i; i<particles_.size(); i++) {
    weights_[i] = particles_[i]->weight();
  }
  ROS_INFO_STREAM("particle weights: " << weights_.sum());
  weights_ = weights_/weights_.sum(); //Renormalize

  //Regularize weights based on distance travelled
  for (int i; i<particles_.size(); i++) {
    float d = std::min<float>(particles_[i]->lastDist()*5, 1);
    weights_[i] = d*weights_[i] + (1-d)/particles_.size();
  }
  weights_ = weights_/weights_.sum(); //Renormalize

  //Find the maximum likelihood particle
  Eigen::VectorXf::Index max_weight;
  weights_.maxCoeff(&max_weight);
  max_likelihood_particle_ = particles_[max_weight];

  ROS_INFO_STREAM("particle reweighting complete");

  int last_num_particles = num_particles_;
  num_particles_ = 0;
  for (const auto& cov : covs_) {
    Eigen::Vector2cf eig = cov.block<2,2>(0,0).eigenvalues(); //complex vector
    num_particles_ += static_cast<int>(sqrt(eig[0].real())*sqrt(eig[1].real())); //Approximation of area of cov ellipse
  }
  num_particles_ = std::min(std::max(num_particles_, 3*last_num_particles/4+10), max_num_particles_); //bounds
  ROS_INFO_STREAM(num_particles_ << " particles");

  //resize new_particles_
  if (num_particles_ < new_particles_.size()) {
    new_particles_.resize(num_particles_);
  } else {
    ROS_DEBUG_STREAM("current size: " << new_particles_.size());
    while (new_particles_.size() < num_particles_) {
      new_particles_.push_back(std::make_shared<StateParticle>(gen_, map_, &params_, false));
    }
    ROS_DEBUG("Created new particles");
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
  ROS_DEBUG("Resampled");
  particles_.swap(new_particles_);
  particle_lock_.unlock();
}

void ParticleFilter::meanLikelihood(Eigen::Vector4f &mean_state) {
  mean_state = Eigen::Vector4f::Zero();
  float cos_sum = 0;
  float sin_sum = 0;
  for (const auto& particle : particles_) {
    Eigen::Vector4f state = particle->mlState();
    mean_state += state;
    cos_sum += cos(state[2]);
    sin_sum += sin(state[2]);
  }
  mean_state /= particles_.size();
  mean_state[2] = atan2(sin_sum/particles_.size(), cos_sum/particles_.size());
}

void ParticleFilter::computeMeanCov(Eigen::Matrix4f &cov) {
  cov.setZero();
  if (num_particles_ < 1) {
    return;
  }
  Eigen::Vector4f mean_likelihood_state;
  meanLikelihood(mean_likelihood_state);

  for (const auto& particle : particles_) {
    Eigen::Vector4f state = particle->mlState() - mean_likelihood_state;
    while (state[2] > M_PI) state[2] -= 2*M_PI;
    while (state[2] < -M_PI) state[2] += 2*M_PI;
    cov += state*state.transpose();
  }
  cov /= particles_.size()-1;
}

void ParticleFilter::maxLikelihood(Eigen::Vector4f &state) {
  state = max_likelihood_particle_->mlState();
}

void ParticleFilter::computeCov(Eigen::Matrix4f &cov) {
  cov.setZero();
  Eigen::Vector4f max_likelihood_state = max_likelihood_particle_->mlState();
  for (const auto& particle : particles_) {
    Eigen::Vector4f state = particle->mlState() - max_likelihood_state;
    while (state[2] > M_PI) state[2] -= 2*M_PI;
    while (state[2] < -M_PI) state[2] += 2*M_PI;
    cov += state*state.transpose();
  }
  cov /= particles_.size()-1;
}

void ParticleFilter::getGMM(std::vector<Eigen::Vector3f> &means, std::vector<Eigen::Matrix3f> &covs) {
  gmm_lock_.lock();
  means = means_;
  covs = covs_;
  gmm_lock_.unlock();
}

void ParticleFilter::gmmThread() {
  while (true) {
    computeGMM();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

void ParticleFilter::computeGMM() {
  //We do this recursively: GMM in euclidean space, and then GMM on each cluster in theta space
  cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
  em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_GENERIC);

  //Build sample array
  particle_lock_.lock();
  num_gaussians_ = std::min(static_cast<int>(particles_.size()/20)+1, num_gaussians_);
  em->setClustersNumber(num_gaussians_);

  int num_samples = std::min(1000, static_cast<int>(particles_.size()));
  cv::Mat samples = cv::Mat(num_samples, 4, CV_64F);
  for (int i=0; i<num_samples; i+=1) {
    Eigen::Vector3f state = particles_[std::min<int>(particles_.size() - 1, 
        i*particles_.size()/num_samples)]->mlState().head<3>();
    samples.at<double>(i, 0) = state[0];
    samples.at<double>(i, 1) = state[1];
    samples.at<double>(i, 2) = 50*cos(state[2]);
    samples.at<double>(i, 3) = 50*sin(state[2]);
  }
  particle_lock_.unlock();

  cv::Mat likelihoods, labels, cluster_means;
  em->trainEM(samples, likelihoods, labels);
  float likelihood_mean = cv::mean(likelihoods)[0];

  int dir = 0;
  //try increasing
  if (num_gaussians_*50 < num_particles_) {
    em->setClustersNumber(num_gaussians_ + 1);
    em->trainEM(samples, likelihoods, labels);
    if (likelihood_mean + 0.3 < cv::mean(likelihoods)[0]) {
      dir = 1;
    }
  }
  //try decr
  if (num_gaussians_ > 1) {
    em->setClustersNumber(num_gaussians_ - 1);
    em->trainEM(samples, likelihoods, labels);
    if (likelihood_mean - 0.3 < cv::mean(likelihoods)[0]) {
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
  gmm_lock_.lock();
  means_.clear();
  covs_.clear();
  for (int i=0; i<cluster_covs.size(); i++) {
    means_.push_back(Eigen::Vector3f(cluster_means.at<double>(i,0), cluster_means.at<double>(i,1),
                     atan2(cluster_means.at<double>(i,3), cluster_means.at<double>(i,2))));
    Eigen::Matrix3f cov;
    cov << cluster_covs[i].at<double>(0,0), cluster_covs[i].at<double>(0,1), 0,
           cluster_covs[i].at<double>(1,0), cluster_covs[i].at<double>(1,1), 0,
           0, 0, 1;
    covs_.push_back(cov);
  }
  //best_rel_pos_ = active_loc_->getBestRelPos(means_);
  gmm_lock_.unlock();
}

void ParticleFilter::updateMap(const cv::Mat &map, const Eigen::Vector2i& map_center) {
  ROS_INFO_STREAM("updating map");
  map_->updateMap(map, map_center);

  ROS_INFO_STREAM("updating particles");
  Eigen::Vector2i map_center_delta = map_center - last_map_center_;
  particle_lock_.lock();
  for (auto& particle : particles_) {
    State s = particle->state();
    s.init_x_px += map_center_delta[0];
    s.init_y_px += map_center_delta[1];
    particle->setState(s);
    particle->updateSize();
  }
  particle_lock_.unlock();
  last_map_center_ = map_center;
}

void ParticleFilter::freezeScale() {
  if (!scale_frozen_) {
    float geo_mean = 1;
    for (const auto& p : particles_) {
      geo_mean *= std::pow(p->state().scale, 1./particles_.size());
    }

    for (auto& p : particles_) {
      p->setScale(geo_mean);
    }
    scale_frozen_ = true;

    ROS_INFO_STREAM("scale converged and locked to " << geo_mean);
  }
}

float ParticleFilter::scale() const {
  if (params_.fixed_scale > 0) {
    return params_.fixed_scale;
  }
  if (scale_frozen_) {
    return particles_[0]->state().scale;
  }
  return -1;
}

int ParticleFilter::numParticles() const {
  return num_particles_;
}

void ParticleFilter::visualize(cv::Mat &img) {
  //Particle dist
  for (const auto& p : particles_) {
    Eigen::Vector3f state = p->mlState().head<3>();
    cv::Point pt(state[0], 
                 img.size().height-state[1]);
    if (pt.x < 0 || pt.x > img.size().width || pt.y < 0 || pt.y > img.size().height) {
      //pt out of bounds, show green circle on border
      pt.x = std::clamp<float>(pt.x, 5, img.size().width-5);
      pt.y = std::clamp<float>(pt.y, 5, img.size().height-5);
      cv::circle(img, pt, 2, cv::Scalar(0,255,0), -1);
    } else {
      cv::Point dir(cos(state[2])*5, -sin(state[2])*5);
      cv::arrowedLine(img, pt-dir, pt+dir, cv::Scalar(0,0,255), 2, CV_AA, 0, 0.3);
    }
  }
  //GMM
  gmm_lock_.lock();
  for (int i=0; i<means_.size(); i++) {
    Eigen::Matrix2f pos_cov = covs_[i].block<2,2>(0,0);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigensolver(pos_cov);
    if (eigensolver.info() != Eigen::Success) break;
    Eigen::Vector2f evals = eigensolver.eigenvalues();
    Eigen::Vector2f maj_axis = eigensolver.eigenvectors().col(0);
    if (evals[0] < 0 || evals[1] < 0) break; //We better be PSD

    cv::Size size(sqrt(evals[0]), sqrt(evals[1]));
    float angle = atan2(-maj_axis[1], maj_axis[0]);
    cv::Point center(means_[i][0], 
                     img.size().height-means_[i][1]);
    cv::ellipse(img, center, size*2, angle*180/M_PI, 0, 360, cv::Scalar(255,0,0), 2);

    cv::Point dir(cos(means_[i][2])*5, -sin(means_[i][2])*5);
    cv::arrowedLine(img, center-dir, center+dir, cv::Scalar(255,0,0), 2, CV_AA, 0, 0.3);

    //relative pose stuff
    //ROS_INFO_STREAM(best_rel_pos_[1]);
    //cv::Point rel_pt(best_rel_pos_[0]*cos(best_rel_pos_[1]+means_[i][2]), 
    //                 best_rel_pos_[0]*sin(best_rel_pos_[1]+means_[i][2]));
    //cv::circle(img, rel_pt+center, 3, cv::Scalar(0,255,0), -1);
  }
  gmm_lock_.unlock();
  //Max likelihood
  if (max_likelihood_particle_) {
    Eigen::Vector3f ml_state = max_likelihood_particle_->mlState().head<3>();
    cv::Point pt(ml_state[0], 
                 img.size().height-ml_state[1]);
    cv::Point dir(cos(ml_state[2])*5, -sin(ml_state[2])*5);
    cv::arrowedLine(img, pt-dir, pt+dir, cv::Scalar(255,0,0), 2, CV_AA, 0, 0.3);
  }
}
