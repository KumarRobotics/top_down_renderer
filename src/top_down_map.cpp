#include "top_down_render/top_down_map.h"

//For some reason the implementation is in a separate #ifdef
//and not within the include guards
#define NANOSVG_IMPLEMENTATION
#include "top_down_render/nanosvg.h"

TopDownMap::TopDownMap(std::string path, cv::Mat& color_lut, int num_classes, float scale, float res) {
  scale_ = scale;
  resolution_ = res;
  num_classes_ = num_classes;
  //parse svg
  NSVGimage* map;

  ROS_INFO_STREAM("Loading map " << path);
  map = nsvgParseFromFile(path.c_str(), "px", 96);

  if (map == NULL) {
    ROS_ERROR("Map loading failed");
    return;
  }

  for (int cls=1; cls<=num_classes; cls++) {
    std::vector<std::vector<Eigen::Vector2f>> class_poly;
    cv::Vec3b color = color_lut.at<cv::Vec3b>(cls);
    int color_compressed = color[0] << 16 | color[1] << 8 | color[2];

    //iterate through shapes
    for (NSVGshape *shape = map->shapes; shape != NULL; shape = shape->next) {
      if ((shape->fill.color & 0xFFFFFF) == color_compressed) {

        //iterate through paths (assume 1 path per shape, really)
        for (NSVGpath *path = shape->paths; path != NULL; path = path->next) {
          std::vector<Eigen::Vector2f> eig_path;
          ROS_DEBUG_STREAM("path");

          for (int i=0; i<path->npts-1; i+=3) {
            float* p = &path->pts[i*2];
            eig_path.push_back(Eigen::Vector2f(p[0], map->height-p[1]));
            ROS_DEBUG_STREAM(p[0] << ", " << map->height-p[1]);
          }
          class_poly.push_back(eig_path);
        }
      }
    }
    poly_.push_back(class_poly);
  }

  ROS_INFO("Map loaded.");
  ROS_INFO_STREAM("Size: " << map->width << " x " << map->height);

  //Generate full rasterized map
  ROS_INFO_STREAM("Rasterizing map...");
  for (int i=0; i<num_classes; i++) {
    Eigen::ArrayXXf class_map(static_cast<int>(map->height/resolution_/scale_), 
                              static_cast<int>(map->width/resolution_/scale_));; //0 inside obstacles, 1 elsewhere
    class_maps_.push_back(class_map);
  }
  ROS_INFO_STREAM("Rasterized map size: " << class_maps_[0].cols() << " x " << class_maps_[0].rows());
  getRasterMap(Eigen::Vector2f(map->width/2/scale_, map->height/2/scale_), 0, resolution_, class_maps_);
  ROS_INFO_STREAM("Rasterization complete");
}

float TopDownMap::scale() {
  return scale_;
}

int TopDownMap::numClasses() {
  return num_classes_;
}

void TopDownMap::getClasses(Eigen::Ref<Eigen::Array2Xf> pts, std::vector<Eigen::ArrayXXf> &classes) {
  if (classes.size() < poly_.size()) return;
  //Algorithm modified from https://en.wikipedia.org/wiki/Even-odd_rule
  int cls_id = 0;
  for (auto cls : poly_) {
    Eigen::Map<Eigen::Array1Xf> class_fills(classes[cls_id].data(), 1, classes[cls_id].size());
    class_fills = -1;
    for (auto poly : cls) {
      int j = poly.size()-1;
      for (int i=0; i<poly.size(); i++) {
        class_fills *= -2*((pts.row(0) < poly[i][1]).cwiseNotEqual(pts.row(0) < poly[j][1]) * 
                       (pts.row(1) < (poly[i][0] + ((poly[j][0]-poly[i][0]) * (pts.row(0)-poly[i][1]) / 
                       (poly[j][1]-poly[i][1]))))).cast<float>() + 1;
        j = i;
      }
    }
    class_fills *= -1; //invert
    class_fills += 1;
    class_fills /= 2;
    cls_id++;
  }
}

void TopDownMap::samplePts(Eigen::Vector2f center, float rot, Eigen::Array2Xf &pts, int cols, int rows, float res) {
  //Generate the sampling coordinates
  Eigen::Map<Eigen::ArrayXXf, 0, Eigen::Stride<2, Eigen::Dynamic>> x_vals(
      pts.data(), cols, rows, 
      Eigen::Stride<2, Eigen::Dynamic>(2, rows*2));
  Eigen::Map<Eigen::ArrayXXf, 0, Eigen::Stride<Eigen::Dynamic, 2>> y_vals(
      pts.row(1).data(), rows, cols, 
      Eigen::Stride<Eigen::Dynamic, 2>(rows*2, 2));

  x_vals = Eigen::RowVectorXf::LinSpaced(rows, 
      -res*(rows-1)/2., res*(rows-1)/2.).replicate(1, cols);
  y_vals = Eigen::RowVectorXf::LinSpaced(cols, 
      -res*(cols-1)/2., res*(cols-1)/2.).replicate(1, rows);

  //Transform coordinates
  Eigen::Matrix2f rotm;
  rotm << cos(rot), -sin(rot),
          sin(rot), cos(rot);
  pts = rotm*pts.matrix();

  x_vals += center[1];
  y_vals += center[0];
}

void TopDownMap::getRasterMap(Eigen::Vector2f center, float rot, float res, std::vector<Eigen::ArrayXXf> &classes) {
  if (classes.size() < 1) return;

  Eigen::Array2Xf pts(2, classes[0].rows()*classes[0].cols());
  samplePts(center*scale_, rot, pts, classes[0].cols(), classes[0].rows(), scale_*res);

  getClasses(pts, classes);
}

void TopDownMap::getLocalMap(Eigen::Vector2f center, float rot, float res, std::vector<Eigen::ArrayXXf> &dists) {
  if (dists.size() < 1) return;
  Eigen::Array2Xf pts(2, dists[0].rows()*dists[0].cols());
  samplePts(center/resolution_, rot, pts, dists[0].cols(), dists[0].rows(), res/resolution_);

  //Generate list of indices
  Eigen::Array2Xi pts_int = pts.round().cast<int>();

  for (int cls=0; cls<dists.size(); cls++) {
    for (int idx=0; idx<dists[0].rows()*dists[0].cols(); idx++) {
      if (pts_int(0, idx) >= 0 && pts_int(0, idx) < class_maps_[cls].rows() &&
          pts_int(1, idx) >= 0 && pts_int(1, idx) < class_maps_[cls].cols()) {
        dists[cls](idx) = class_maps_[cls](pts_int(0, idx), pts_int(1, idx));
      } else {
        dists[cls](idx) = 100;
      }
    }
  }
}
