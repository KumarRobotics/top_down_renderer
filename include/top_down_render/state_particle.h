#ifndef STATE_PARTICLE_H_
#define STATE_PARTICLE_H_

#include "top_down_render/top_down_map.h"

#include <random>
#include <chrono>

typedef struct State {
  float x;
  float y;
  float theta;
} State;

class StateParticle {
  public:
    StateParticle(State s, float width, float height, TopDownMap *map);
    StateParticle(std::mt19937 *gen, float width, float height, TopDownMap *map);
    
    void propagate(std::mt19937 *gen);
    State state();
    void setState(State s);
    void computeWeight(std::vector<Eigen::ArrayXXf> &top_down_scan);
    float weight();
  private:
    //State
    State state_;
    float width_;
    float height_;
    float weight_;
    TopDownMap *map_;
};

#endif //STATE_PARTICLE_H_
