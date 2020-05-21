#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "top_down_render/state_particle.h"
#include "top_down_render/active_localizer.h"

#include <random>
#include <execution>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp> //cv::circle

class ParticleFilter {
  public:
    ParticleFilter(int N, float width, float height, TopDownMapPolar *map);
    void propagate();
    void update(std::vector<Eigen::ArrayXXf> &top_down_scan, 
                std::vector<Eigen::ArrayXXf> &top_down_geo, float res);
    void computeCov(Eigen::Matrix2f &cov);
    void getGMM(std::vector<Eigen::Vector3f> &means, std::vector<Eigen::Matrix3f> &covs);
    void visualize(cv::Mat &img);
  private:
    int num_particles_;
    std::mutex particle_lock_;
    std::vector<std::shared_ptr<StateParticle>> particles_;
    std::vector<std::shared_ptr<StateParticle>> new_particles_;

    std::shared_ptr<StateParticle> max_likelihood_particle_;
    std::mt19937 *gen_;
    Eigen::VectorXf weights_;
    TopDownMapPolar* map_;
    ActiveLocalizer* active_loc_;
    float width_;
    float height_;

    //GMM stuff
    int num_gaussians_; //Only used in GMM thread
    std::mutex gmm_lock_;
    Eigen::Vector2f best_rel_pos_;
    std::vector<Eigen::Vector3f> means_;
    std::vector<Eigen::Matrix3f> covs_;

    std::thread *gmm_thread_;

    void computeGMM();
    void gmmThread();
};

#endif //PARTICLE_FILTER_H_
