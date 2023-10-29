#include <filesystem>
#include "top_down_render/top_down_map.h"

// For some reason the implementation is in a separate #ifdef
// and not within the include guards
#define NANOSVG_IMPLEMENTATION
#include "top_down_render/nanosvg.h"

TopDownMap::TopDownMap(const TopDownMap::Params& params) {
  params_ = params;
  map_center_ = Eigen::Vector2i::Zero();
  if (params_.map_path == "") {
    // No static map case
    have_map_ = false;
    return;
  }

  if (loadCacheMetaData(params_.map_path)) {
    loadCachedMaps();
    ROS_INFO_STREAM("\033[32m" << "[XView] Loaded cached maps" << "\033[0m");
  } else {
    if (params_.map_path.substr(params_.map_path.size()-4) == ".svg") { 
      std::vector<std::vector<std::vector<Eigen::Vector2f>>> poly;
      auto map_size = loadSvg(params_.map_path, poly);

      getRasterMap(map_size, 0, params_.resolution, poly);
      ROS_INFO_STREAM("\033[32m" << "[XView] Map rasterized, size: " << 
          class_maps_[0].cols() << " x " << class_maps_[0].rows() << "\033[0m");

      saveRasterizedMaps(params_.map_path.substr(0, params_.map_path.size()-4) + "_raster_cache");
    } else if (params_.map_path.substr(params_.map_path.size()-4) == ".png" ||
               params_.map_path.substr(params_.map_path.size()-4) == ".jpg") 
    {
      ROS_INFO_STREAM("\033[32m" << "[XView] Loading compressed raster map " << params_.map_path << "\033[0m");
      cv::Mat color_map = cv::imread(params_.map_path);
      if (!color_map.empty()) {
        cv::Mat class_map;
        params_.color_lut.color2Ind(color_map, class_map);
        loadCompressedRasterMap(class_map);
      } else {
        ROS_ERROR("[XView] Compressed raster map loading failed");
      }
    } else {
      ROS_INFO_STREAM("\033[32m" << "[XView] Loading raster cache " << params_.map_path << "\033[0m");
      loadRasterizedMaps(params_.map_path);
    }

    for (size_t i=0; i<2; i++) {
      Eigen::ArrayXXf geo_map(class_maps_[0].rows(), 
                              class_maps_[0].cols()); // 0 inside obstacles, 1 elsewhere
      geo_maps_.push_back(geo_map);
    }
    getGeoRasterMap(geo_maps_);

    // Do this after so we can reuse maps
    computeDists(class_maps_, class_mask_);
    Eigen::ArrayXXc tmp;
    computeDists(geo_maps_, tmp);
    ROS_INFO_STREAM("\033[32m" << "[XView] Map Loading Complete" << "\033[0m");

    saveCachedMaps(params_.map_path);
  }
  have_map_ = true;
}

Eigen::Vector2i TopDownMap::loadSvg(
    const std::string& svg_path, std::vector<std::vector<std::vector<Eigen::Vector2f>>>& poly)
{
  ROS_INFO_STREAM("\033[32m" << "[XView] Loading vector map " << params_.map_path << "\033[0m");
  NSVGimage* map = nsvgParseFromFile(svg_path.c_str(), "px", 96);

  if (map == NULL) {
    ROS_ERROR("[XView] Svg map loading failed");
    return {0, 0};
  }

  poly.resize(params_.num_classes);
  for (size_t cls=0; cls<params_.flatten_lut.size(); cls++) {
    std::vector<std::vector<Eigen::Vector2f>> class_poly;
    auto color = SemanticColorLut::unpackColor(params_.color_lut.ind2Color(cls));
    // The svg is 0xBBGGRR instead of 0xRRGGBB
    uint32_t color_compressed = color[0] << 16 | color[1] << 8 | color[2];

    // iterate through shapes
    for (NSVGshape *shape = map->shapes; shape != NULL; shape = shape->next) {
      if ((shape->fill.color & 0xFFFFFF) == color_compressed) {

        // iterate through paths (assume 1 path per shape, really)
        for (NSVGpath *path = shape->paths; path != NULL; path = path->next) {
          std::vector<Eigen::Vector2f> eig_path;
          ROS_DEBUG_STREAM("path");

          for (size_t i=0; i<path->npts-1; i+=3) {
            float* p = &path->pts[i*2];
            eig_path.push_back(Eigen::Vector2f(p[0], map->height-p[1]));
            //ROS_DEBUG_STREAM(p[0] << ", " << map->height-p[1]);
          }
          class_poly.push_back(eig_path);
        }
      }    
    }
    // Append class_poly contents into the appropriate class
    poly[params_.flatten_lut[cls]].insert(poly[params_.flatten_lut[cls]].end(), 
        class_poly.begin(), class_poly.end());
  }

  ROS_INFO_STREAM("\033[32m" << "[XView] Vector map loaded, size: " << 
      map->width << " x " << map->height << "\033[0m");

  Eigen::Vector2i map_size{map->width, map->height};
  nsvgDelete(map);

  return map_size;
}

void TopDownMap::loadCompressedRasterMap(const cv::Mat& map) {
  class_maps_.clear();
  for (size_t i=0; i<params_.num_classes; i++) {
    // 0 inside obstacles, 1 elsewhere
    Eigen::ArrayXXf class_map =
      Eigen::ArrayXXf::Constant(static_cast<int>(map.size().height/params_.resolution), 
                                static_cast<int>(map.size().width/params_.resolution), 1.0); 
    class_maps_.push_back(class_map);
  }

  // Not actually used at the moment
  geo_maps_.clear();
  for (size_t i=0; i<2; i++) {
    Eigen::ArrayXXf geo_map =
      Eigen::ArrayXXf::Constant(static_cast<int>(map.size().height/params_.resolution), 
                                static_cast<int>(map.size().width/params_.resolution), 1.0); 
    geo_maps_.push_back(geo_map);
  }

  for (size_t xi=0; xi<class_maps_[0].cols(); xi++) {
    for (size_t yi=0; yi<class_maps_[0].rows(); yi++) {
      int cls = params_.flatten_lut[map.at<uint8_t>(std::max<int>(map.size().height-yi*params_.resolution-1, 0), 
                                               std::min<int>(xi*params_.resolution, map.size().width-1))];
      if (cls >= 0 && cls < class_maps_.size()) {
        class_maps_[cls](yi, xi) = 0;
      }
    }
  }
}

void TopDownMap::updateMap(const cv::Mat &map, const Eigen::Vector2i &map_center) {
  map_center_ = map_center;
  loadCompressedRasterMap(map);

  if (!class_maps_[1].isZero(0)) {
    have_map_ = true;
  } else {
    ROS_WARN("[XView] Received map with no road");
  }

  computeDists(class_maps_, class_mask_);
}

void TopDownMap::getClassesAtPoint(const Eigen::Vector2i &center_ind, std::vector<int> &classes) {
  Eigen::Vector2i center = (center_ind.cast<float>() / params_.resolution).cast<int>();
  classes.clear();
  for (int cls=0; cls<params_.num_classes; cls++) {
    if (center[0] < class_maps_[cls].cols() && center[1] < class_maps_[cls].rows() &&
        center[0] >= 0 && center[1] >= 0) {
      if (class_maps_[cls](center[1], center[0]) < 1) {
        classes.push_back(cls);
      }
    }
  }
}

void TopDownMap::getClassesAtPoint(const Eigen::Vector2f &center, std::vector<int> &classes) {
  Eigen::Vector2i center_ind = (center/params_.resolution).cast<int>();
  getClassesAtPoint(center_ind, classes);
}

int TopDownMap::numClasses() const {
  return params_.num_classes;
}

Eigen::Vector2i TopDownMap::size() const {
  return Eigen::Vector2i(class_maps_[0].cols(), class_maps_[0].rows());
}

Eigen::Vector2i TopDownMap::mapCenter() const {
  return map_center_;
}

float TopDownMap::resolution() const {
  return params_.resolution;
}

bool TopDownMap::haveMap() const {
  return have_map_;
}

void TopDownMap::saveRasterizedMaps(const std::string &path) {
  mkdir(path.c_str(), S_IRWXU);

  size_t ind = 0;
  cv::Mat cv_map, cv_map_scaled;
  for (const auto &map : class_maps_) {
    //eigen2cv doesn't like arrays
    Eigen::MatrixXf mat_map = map.array();
    cv::eigen2cv(mat_map, cv_map);
    cv_map.convertTo(cv_map_scaled, CV_8UC1, 255);
    // Flip to look like the input map
    cv::flip(cv_map_scaled, cv_map_scaled, 0);
    cv::imwrite(path+"/class"+std::to_string(ind++)+".png", cv_map_scaled);
  }
}

void TopDownMap::loadRasterizedMaps(const std::string &map_path) {
  cv::Mat cv_map, cv_map_float;
  for (size_t i=0; i<params_.num_classes; i++) {
    cv_map = cv::imread(map_path+"/class"+std::to_string(i)+".png", cv::IMREAD_GRAYSCALE);
    // We flipped when we saved, have to flip back
    cv::flip(cv_map, cv_map, 0);
    cv_map.convertTo(cv_map_float, CV_32FC1, 1./255);
    Eigen::MatrixXf mat_map;
    cv::cv2eigen(cv_map_float, mat_map);
    class_maps_.push_back(mat_map.array());
  }
}

bool TopDownMap::loadCacheMetaData(const std::string &map_path) {
  std::ifstream data_file(std::string(getenv("HOME")) + "/.ros/xview_cache/cached_data.txt");
  if (!data_file) return false;

  ROS_INFO_STREAM("\033[32m" << "[XView] Found cache, checking if parameters have changed" << "\033[0m");
  
  //Check that metadata agrees
  std::string line;
  std::getline(data_file, line);
  if (line != map_path) return false;
  std::getline(data_file, line);
  if (std::stoi(line) != params_.num_classes) return false;
  std::getline(data_file, line);
  if (std::abs(std::stof(line) - params_.resolution) > 0.01) return false;

  return true;
}

void TopDownMap::loadCachedMaps() {
  for (int cls=0; cls<params_.num_classes; cls++) {
    Eigen::ArrayXXf class_map;
    std::string name = std::string(getenv("HOME")) + "/.ros/xview_cache/class_map" + std::to_string(cls) + ".eig";
    read_binary(name, class_map);
    class_maps_.push_back(class_map);
  }

  for (int cls=0; cls<2; cls++) {
    Eigen::ArrayXXf geo_map;
    std::string name = std::string(getenv("HOME")) + "/.ros/xview_cache/geo_map" + std::to_string(cls) + ".eig";
    read_binary(name, geo_map);
    geo_maps_.push_back(geo_map);
  }

  std::string name = std::string(getenv("HOME")) + "/.ros/xview_cache/class_mask.eig";
  read_binary(name, class_mask_);
}

void TopDownMap::saveCachedMaps(const std::string &map_path) {
  using namespace std::filesystem;
  path cache = path(getenv("HOME")) / path(".ros/xview_cache/cached_data.txt");
  create_directory(cache.parent_path());

  std::ofstream data_file(cache, 
                          std::ofstream::out | std::ofstream::trunc);
  data_file << map_path << std::endl;
  data_file << params_.num_classes << std::endl;
  data_file << params_.resolution << std::endl;

  for (int cls=0; cls<params_.num_classes; cls++) {
    std::string name = std::string(getenv("HOME")) + "/.ros/xview_cache/class_map" + std::to_string(cls) + ".eig";
    write_binary(name, class_maps_[cls]);
  }

  for (int cls=0; cls<2; cls++) {
    std::string name = std::string(getenv("HOME")) + "/.ros/xview_cache/geo_map" + std::to_string(cls) + ".eig";
    write_binary(name, geo_maps_[cls]);
  }

  std::string name = std::string(getenv("HOME")) + "/.ros/xview_cache/class_mask.eig";
  write_binary(name, class_mask_);
}

//For each cell, compute the distance to other cells
void TopDownMap::computeDists(std::vector<Eigen::ArrayXXf> &classes, Eigen::ArrayXXc &mask) {
  ROS_INFO_STREAM("\033[32m" << "[XView] Computing distance maps..." << "\033[0m");

  auto start = std::chrono::high_resolution_clock::now();
  //Compute all locations with no known class
  mask = Eigen::ArrayXXc::Zero(classes[0].rows(), classes[0].cols()); 
  for (int cls_id=0; cls_id<classes.size(); cls_id++) {
    mask += classes[cls_id].cast<uint8_t>();
  }
  cv::Mat mask_mat(mask.cols(), mask.rows(), CV_8UC1, (void*)mask.data());
  cv::threshold(mask_mat, mask_mat, classes.size()-1, 255, cv::THRESH_BINARY);

  for (int cls_id=0; cls_id<classes.size(); cls_id++) {
    //Copy class buffer
    Eigen::ArrayXXf class_buf = classes[cls_id];
    cv::Mat class_mat(class_buf.cols(), class_buf.rows(), CV_32FC1, (void*)class_buf.data());
    cv::Mat binary_class_mat;
    class_mat.convertTo(binary_class_mat, CV_8UC1);

    //When we write to this, we modify the original data
    cv::Mat dist_mat(classes[cls_id].cols(), classes[cls_id].rows(), CV_32FC1, 
        (void*)classes[cls_id].data());

    cv::distanceTransform(binary_class_mat, dist_mat, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    //Normalize by map res and thresh
    dist_mat *= params_.resolution;
    cv::threshold(dist_mat, dist_mat, 50, 0, cv::THRESH_TRUNC);

    dist_mat.setTo(/*params_.out_of_bounds_const*/ 0, mask_mat); 

    ROS_INFO_STREAM("\033[32m" << "[XView] Distance map for class " << cls_id << " complete" << "\033[0m");
  }
  mask /= 255;

  auto end = std::chrono::high_resolution_clock::now();
  ROS_INFO_STREAM("\033[32m" << "[XView] Distance maps computed" << "\033[0m");
  //ROS_INFO_STREAM("dist gen took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000 << "ms");
}

void TopDownMap::getClasses(Eigen::Ref<Eigen::Array2Xf> pts, 
    const std::vector<std::vector<std::vector<Eigen::Vector2f>>> &poly,
    std::vector<Eigen::ArrayXXf>& classes)
{
  if (classes.size() < poly.size()) return;
  //Algorithm modified from https://en.wikipedia.org/wiki/Even-odd_rule
  int cls_id = 0;
  for (auto cls : poly) {
    Eigen::Map<Eigen::Array1Xf> class_fills(classes[cls_id].data(), 1, classes[cls_id].size());
    Eigen::Array1Xf class_fills_buf(classes[cls_id].size());
    class_fills = -1;
    for (auto poly : cls) {
      int j = poly.size()-1;
      class_fills_buf = -1;
      for (int i=0; i<poly.size(); i++) {
        class_fills_buf *= -2*((pts.row(0) < poly[i][1]).cwiseNotEqual(pts.row(0) < poly[j][1]) * 
                              (pts.row(1) < (poly[i][0] + ((poly[j][0]-poly[i][0]) * (pts.row(0)-poly[i][1]) / 
                              (poly[j][1]-poly[i][1]))))).cast<float>() + 1;
        j = i;
      }
      class_fills = class_fills.max(class_fills_buf);
    }
    class_fills *= -1; //invert
    class_fills += 1;
    class_fills /= 2;
    cls_id++;
  }

  //Only one ground type per cell
  for (int under_cls_id : params_.exclusive_classes) {
    for (int cls_id : params_.exclusive_classes) {
      if (under_cls_id < cls_id) {
        classes[under_cls_id] += 1-classes[cls_id];
      }
    }
    classes[under_cls_id] = classes[under_cls_id].cwiseMin(1);
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
      -res*(rows-1)/2., res*(rows-1)/2.).replicate(cols, 1); 
  y_vals = Eigen::RowVectorXf::LinSpaced(cols, 
      -res*(cols-1)/2., res*(cols-1)/2.).replicate(rows, 1);

  //Transform coordinates
  Eigen::Matrix2f rotm;
  rotm << cos(rot), -sin(rot),
          sin(rot), cos(rot);
  pts = rotm*pts.matrix();

  x_vals += center[1];
  y_vals += center[0];
}

void TopDownMap::getRasterMap(const Eigen::Vector2i& map_size, float rot, float res, 
    std::vector<std::vector<std::vector<Eigen::Vector2f>>> &poly) 
{
  if (poly.size() < 1) return;

  // Generate full rasterized map
  ROS_INFO_STREAM("\033[32m" << "[XView] Rasterizing map..." << "\033[0m");
  for (size_t i=0; i<params_.num_classes; i++) {
    Eigen::ArrayXXf class_map(static_cast<int>(map_size[1]/params_.resolution), 
                              static_cast<int>(map_size[0]/params_.resolution)); //0 inside obstacles, 1 elsewhere
    class_maps_.push_back(class_map);
  }

  Eigen::Array2Xf pts(2, class_maps_[0].rows()*class_maps_[0].cols());
  samplePts(map_size.cast<float>()/2, rot, pts, class_maps_[0].cols(), class_maps_[0].rows(), res);

  getClasses(pts, poly, class_maps_);
}

void TopDownMap::getGeoRasterMap(std::vector<Eigen::ArrayXXf> &geo_cls) {
  if (geo_cls.size() < 2) return;

  for (int i=0; i<geo_cls.size(); i++) {
    geo_cls[i].setZero();
  }

  for (int i=3; i<params_.num_classes; i++) {
    geo_cls[1] += 1-class_maps_[i]; //geometric classes
  }

  //Re-binarize
  for (int i=0; i<geo_cls.size(); i++) {
    geo_cls[i] = geo_cls[i].cwiseMin(1);
    geo_cls[i] = 1-geo_cls[i];
  }
  geo_cls[0] = 1-geo_cls[1];
}

void TopDownMap::getLocalMap(Eigen::Vector2f center, float rot, float res, 
    std::vector<Eigen::ArrayXXf> &dists, Eigen::ArrayXXc &mask) 
{
  if (dists.size() < 1) return;
  Eigen::Array2Xf pts(2, dists[0].rows()*dists[0].cols());
  samplePts(center/params_.resolution, rot, pts, dists[0].cols(), dists[0].rows(), res/params_.resolution);

  //Generate list of indices
  Eigen::Array2Xi pts_int = pts.round().cast<int>();

  for (int cls=0; cls<dists.size(); cls++) {
    for (int idx=0; idx<dists[0].rows()*dists[0].cols(); idx++) {
      if (pts_int(0, idx) >= 0 && pts_int(0, idx) < class_maps_[cls].rows() &&
          pts_int(1, idx) >= 0 && pts_int(1, idx) < class_maps_[cls].cols()) {
        dists[cls](idx) = class_maps_[cls](pts_int(0, idx), pts_int(1, idx));
      } else {
        dists[cls](idx) = 0; //params_.out_of_bounds_const;
      }
    }
  }

  for (int idx=0; idx<dists[0].rows()*dists[0].cols(); idx++) {
    if (pts_int(0, idx) >= 0 && pts_int(0, idx) < class_mask_.rows() &&
        pts_int(1, idx) >= 0 && pts_int(1, idx) < class_mask_.cols()) {
      mask(idx) = class_mask_(pts_int(0, idx), pts_int(1, idx));
    } else {
      // out of bounds
      mask(idx) = 1;
    }
  }
}

void TopDownMap::getLocalGeoMap(Eigen::Vector2f center, float rot, float res, 
    std::vector<Eigen::ArrayXXf> &dists) 
{
  if (dists.size() < 1) return;
  Eigen::Array2Xf pts(2, dists[0].rows()*dists[0].cols());
  samplePts(center/params_.resolution, rot, pts, dists[0].cols(), dists[0].rows(), res/params_.resolution);

  //Generate list of indices
  Eigen::Array2Xi pts_int = pts.round().cast<int>();

  for (int cls=0; cls<dists.size(); cls++) {
    for (int idx=0; idx<dists[0].rows()*dists[0].cols(); idx++) {
      if (pts_int(0, idx) >= 0 && pts_int(0, idx) < geo_maps_[cls].rows() &&
          pts_int(1, idx) >= 0 && pts_int(1, idx) < geo_maps_[cls].cols()) {
        dists[cls](idx) = geo_maps_[cls](pts_int(0, idx), pts_int(1, idx));
      } else {
        dists[cls](idx) = 0; //params_.out_of_bounds_const;
      }
    }
  }
}
