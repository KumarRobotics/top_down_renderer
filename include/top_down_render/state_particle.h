#ifndef STATE_PARTICLE_H_
#define STATE_PARTICLE_H_

#include "top_down_render/top_down_map_polar.h"

#include <random>
#include <chrono>

typedef struct ThetaParticle {
  float theta;
  float weight;
} ThetaParticle;

typedef struct State {
  float init_x_px;
  float init_y_px;
  float dx_m = 0;
  float dy_m = 0;
  float scale; //px/m
  std::shared_ptr<std::vector<ThetaParticle>> theta_particles;
} State;

typedef struct FilterParams {
  float pos_cov;
  float theta_cov;
  float regularization;
  float fixed_scale = -1;
} FilterParams;

class StateParticle {
  public:
    StateParticle(std::mt19937 *gen, TopDownMapPolar *map, FilterParams &params);
    
    void propagate(Eigen::Vector2f &trans, float omega, bool scale_freeze=false);
    State state();
    Eigen::Vector4f mlState();
    void setState(State s);
    void computeWeight(std::vector<Eigen::ArrayXXf> &top_down_scan, 
                       std::vector<Eigen::ArrayXXf> &top_down_geo, float res);
    float weight();
    float thetaCov();
    void setScale(float scale);
  private:
    //State
    State state_;
    float width_;
    float height_;
    float weight_;
    float ml_theta_;
    TopDownMapPolar *map_;
    std::mt19937 *gen_;

    FilterParams params_;
    std::vector<float> class_weights_;

    float getCostForRot(std::vector<Eigen::ArrayXXf> &top_down_scan,
                        std::vector<Eigen::ArrayXXf> &top_down_geo,
                        std::vector<Eigen::ArrayXXf> &classes,
                        std::vector<Eigen::ArrayXXf> &geo_cls, float rot);
    void resampleParticles(int num_part);
};

#endif //STATE_PARTICLE_H_
