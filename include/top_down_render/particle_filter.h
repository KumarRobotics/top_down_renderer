#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "top_down_render/state_particle.h"

#include <random>
#include <execution>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> //cv::circle

class ParticleFilter {
  public:
    ParticleFilter(int N, float width, float height, TopDownMapPolar *map);
    void propagate();
    void update(std::vector<Eigen::ArrayXXf> &top_down_scan, 
                std::vector<Eigen::ArrayXXf> &top_down_geo, float res);
    void computeCov(Eigen::Matrix2f &cov);
    void visualize(cv::Mat &img);
  private:
    int num_particles_;
    std::vector<StateParticle*> particles_;
    std::vector<StateParticle*> new_particles_;

    StateParticle* max_likelihood_particle_;
    std::mt19937 *gen_;
    Eigen::VectorXf weights_;
    TopDownMapPolar* map_;
};

#endif //PARTICLE_FILTER_H_
