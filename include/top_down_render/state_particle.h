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
  float x;
  float y;
  std::vector<ThetaParticle> *theta_particles;
} State;

class StateParticle {
  public:
    StateParticle(std::mt19937 *gen, float width, float height, TopDownMapPolar *map);
    
    void propagate();
    State state();
    Eigen::Vector3f mlState();
    void setState(State s);
    void computeWeight(std::vector<Eigen::ArrayXXf> &top_down_scan, 
                       std::vector<Eigen::ArrayXXf> &top_down_geo, float res);
    float weight();
    float thetaCov();
  private:
    //State
    State state_;
    float width_;
    float height_;
    float weight_;
    float ml_theta_;
    TopDownMapPolar *map_;
    std::mt19937 *gen_;

    float getCostForRot(std::vector<Eigen::ArrayXXf> &top_down_scan,
                        std::vector<Eigen::ArrayXXf> &top_down_geo,
                        std::vector<Eigen::ArrayXXf> &classes,
                        std::vector<Eigen::ArrayXXf> &geo_cls, float rot);
    void resampleParticles(int num_part);
};

#endif //STATE_PARTICLE_H_
