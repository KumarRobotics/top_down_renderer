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
  typedef Array<float, 1, Dynamic> Array1Xf;
  typedef Array<uint8_t, Dynamic, Dynamic> ArrayXXc;
  typedef Array<float, Dynamic, Dynamic> ArrayXXf;
}

class TopDownMap {
  public:
    TopDownMap(std::string path, cv::Mat& color_lut, int num_classes, int num_ex, float scale, float res);

    void getLocalMap(Eigen::Vector2f center, float rot, float res, std::vector<Eigen::ArrayXXf> &dists);
    void getLocalGeoMap(Eigen::Vector2f center, float rot, float res, std::vector<Eigen::ArrayXXf> &dists);
    float scale();
    int numClasses();
  protected:
    std::vector<std::vector<std::vector<Eigen::Vector2f>>> poly_;
    std::vector<Eigen::ArrayXXf> class_maps_;
    std::vector<Eigen::ArrayXXf> geo_maps_;
    float scale_; //pixels per meter for svg
    float resolution_; //meters per pixel for rasterized map
    int num_classes_;
    int num_exclusive_classes_;

    void getRasterMap(Eigen::Vector2f center, float rot, float res, std::vector<Eigen::ArrayXXf> &classes);
    void getGeoRasterMap(Eigen::Vector2f center, float rot, float res, std::vector<Eigen::ArrayXXf> &geo_cls);
    void computeDists(std::vector<Eigen::ArrayXXf> &classes);
    void getClasses(Eigen::Ref<Eigen::Array2Xf> pts, std::vector<Eigen::ArrayXXf> &classes);
    void samplePts(Eigen::Vector2f center, float rot, Eigen::Array2Xf &pts, int cols, int rows, float res);
};

#endif
