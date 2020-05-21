#include "top_down_render/active_localizer.h"

ActiveLocalizer::ActiveLocalizer(TopDownMapPolar *map) {
  map_ = map;
}

float ActiveLocalizer::computeTotalDifference(std::vector<std::vector<Eigen::ArrayXXf>> &local_maps) {
  float total_difference = 0;
  int cnt = 0;
  for (int i=0; i<local_maps.size(); i++) {
    for (int j=0; j<i; j++) {
      for (int cls=0; cls<local_maps[0].size(); cls++) {
        total_difference += (local_maps[i][cls] - local_maps[j][cls]).cwiseAbs().sum();
        cnt += 1;
      }
    }
  }

  return total_difference/cnt;
}

void ActiveLocalizer::getLocalMap(Eigen::Vector3f &state, std::vector<Eigen::ArrayXXf> &local_map) {
  std::vector<Eigen::ArrayXXf> local_map_orig;
  for (int n=0; n<map_->numClasses(); n++) {
    local_map_orig.push_back(Eigen::ArrayXXf(100, 25));
  }

  map_->getLocalMap(state.head<2>(), 2, local_map_orig);

  int num_bins = local_map[0].rows();
  int rot_shift = static_cast<int>(std::round(state[2]*num_bins/2/M_PI));
  //normalize rotation
  while (rot_shift >= num_bins) rot_shift -= num_bins;
  while (rot_shift < 0) rot_shift += num_bins;

  for (int n=0; n<map_->numClasses(); n++) {
    local_map[n].bottomRows(rot_shift) = local_map_orig[n].topRows(rot_shift);
    local_map[n].topRows(num_bins-rot_shift) = local_map_orig[n].bottomRows(num_bins-rot_shift);
  }
}

//Get the best relative position to minimize pose uncertainty
Eigen::Vector2f ActiveLocalizer::getBestRelPos(std::vector<Eigen::Vector3f> &preds) {
  //setup data structures
  std::vector<std::vector<Eigen::ArrayXXf>> local_maps;
  for (int i=0; i<preds.size(); i++) {
    std::vector<Eigen::ArrayXXf> local_map;
    for (int n=0; n<map_->numClasses(); n++) {
      local_map.push_back(Eigen::ArrayXXf(100, 25));
    }
    local_maps.push_back(local_map);
  }

  float dist=50;
  float best_diff = 0;
  Eigen::Vector2f best_rel_pos(0, 0);
  while (best_diff < 6000 && dist < 150) {
    for (float theta=0; theta<2*M_PI; theta+=M_PI/8) {
      int idx=0;
      for (auto pred : preds) {
        Eigen::Vector3f possible_pos = pred;
        possible_pos.head<2>() += dist*Eigen::Vector2f(cos(theta + pred[2]), sin(theta + pred[2]));

        getLocalMap(possible_pos, local_maps[idx]);
        idx++;
      }

      float diff = computeTotalDifference(local_maps);
      if (diff > best_diff) {
        best_diff = diff;
        best_rel_pos = Eigen::Vector2f(dist, theta);
      }
    }
    
    dist += 25;
  }

  ROS_INFO_STREAM("Max diff: " << best_diff);

  return best_rel_pos;
}
