#ifndef TOP_DOWN_MAP_H_
#define TOP_DOWN_MAP_H_

#include <ros/ros.h> //Just for prints
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

#define NANOSVG_CPLUSPLUS
#include "top_down_render/nanosvg.h"

namespace Eigen {
  typedef Array<bool, 1, Dynamic> Array1Xb;
  typedef Array<uint8_t, 1, Dynamic> Array1Xc;
  typedef Array<uint8_t, Dynamic, Dynamic> ArrayXXc;
}

class TopDownMap {
  public:
    TopDownMap(std::string path, cv::Mat& color_lut, int num_classes, float scale);

    void getLocalMap(Eigen::Vector2f &center, float rot, float res, Eigen::ArrayXXc &classes);
    float scale();
  private:
    std::vector<std::vector<std::vector<Eigen::Vector2f>>> poly_;
    float scale_; //pixels per meter

    void getClasses(Eigen::Ref<Eigen::Array2Xf> pts, Eigen::Ref<Eigen::Array1Xc> classes);
};

#endif
