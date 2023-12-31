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

#ifndef CV_AA
// OpenCV4 compat
#define CV_AA cv::LINE_AA
#endif

class ParticleFilter {
  public:
    ParticleFilter(int N, TopDownMapPolar *map, FilterParams &params);
    void propagate(Eigen::Vector2f &trans, float omega);
    void update(std::vector<Eigen::ArrayXXf> &top_down_scan, 
                std::vector<Eigen::ArrayXXf> &top_down_geo, float res);
    void computeCov(Eigen::Matrix4f &cov);
    void maxLikelihood(Eigen::Vector4f &state);

    void computeMeanCov(Eigen::Matrix4f &cov);
    void meanLikelihood(Eigen::Vector4f &state);

    void getGMM(std::vector<Eigen::Vector3f> &means, std::vector<Eigen::Matrix3f> &covs);
    void visualize(cv::Mat &img);
    void freezeScale();
    bool isScaleFrozen() { return scale_frozen_; }
    float scale() const;
    int numParticles() const;

    void updateMap(const cv::Mat &map, const Eigen::Vector2i& map_center);
  private:
    int num_particles_;
    int max_num_particles_;
    std::mutex particle_lock_;
    std::vector<std::shared_ptr<StateParticle>> particles_;
    std::vector<std::shared_ptr<StateParticle>> new_particles_;

    bool scale_frozen_ = false;

    std::shared_ptr<StateParticle> max_likelihood_particle_;
    std::mt19937 *gen_;
    Eigen::VectorXf weights_;
    TopDownMapPolar* map_;
    ActiveLocalizer* active_loc_;

    FilterParams params_;

    Eigen::Vector2i last_map_center_;
    void initializeParticles();

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
