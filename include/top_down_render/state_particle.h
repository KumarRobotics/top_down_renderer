#ifndef STATE_PARTICLE_H_
#define STATE_PARTICLE_H_

#include "top_down_render/top_down_map_polar.h"

#include <random>
#include <chrono>

typedef struct State {
  float init_x_px = 0;
  float init_y_px = 0;
  float dx_m = 0;
  float dy_m = 0;
  float theta = 0;
  float scale = 1; //px/m
  bool have_init = false;
} State;

typedef struct FilterParams {
  float pos_cov;
  float theta_cov;
  float regularization;
  float init_pos_px_x = -1; 
  float init_pos_px_y = -1;
  float init_pos_px_cov = -1;

  float init_pos_m_x = -1; 
  float init_pos_m_y = -1;
  float init_pos_deg_theta = -1;
  float init_pos_deg_cov = -1;

  bool force_on_map = false;
  float fixed_scale = -1;
  float scale_log_min = -0.1;
  float scale_log_max = 1;

  std::vector<float> class_weights;
} FilterParams;

class StateParticle {
  public:
    StateParticle(std::mt19937 *gen, TopDownMapPolar *map, FilterParams *params, bool init=true);
    
    void propagate(Eigen::Vector2f &trans, float omega, bool scale_freeze=false);
    State state() const;
    Eigen::Vector4f mlState();
    void setState(const State& s);
    void computeWeight(std::vector<Eigen::ArrayXXf> &top_down_scan, 
                       std::vector<Eigen::ArrayXXf> &top_down_geo, float res);
    float weight() const;
    float lastDist() const;
    void setScale(float scale);
    void updateSize();
  private:
    //State
    State state_;
    float width_;
    float height_;
    float weight_;
    float last_dist_;
    TopDownMapPolar *map_;
    std::mt19937 *gen_;

    FilterParams *params_;

    float getCostForRot(const std::vector<Eigen::ArrayXXf> &top_down_scan,
                        const std::vector<Eigen::ArrayXXf> &top_down_geo,
                        const std::vector<Eigen::ArrayXXf> &classes,
                        const std::vector<Eigen::ArrayXXf> &geo_cls, 
                        const Eigen::ArrayXXf &mask, float rot);
};

#endif //STATE_PARTICLE_H_
