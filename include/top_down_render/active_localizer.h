#ifndef ACTIVE_LOCALIZER_H_
#define ACTIVE_LOCALIZER_H_

#include "top_down_render/top_down_map_polar.h"

class ActiveLocalizer {
  public:
    ActiveLocalizer(TopDownMapPolar *map);
    Eigen::Vector2f getBestRelPos(std::vector<Eigen::Vector3f> &preds);
  private:
    TopDownMapPolar *map_;

    float computeTotalDifference(std::vector<std::vector<Eigen::ArrayXXf>> &local_maps);
    void getLocalMap(Eigen::Vector3f &state, std::vector<Eigen::ArrayXXf> &local_map);
};

#endif //ACTIVE_LOCALIZER_H_
