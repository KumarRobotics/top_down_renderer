#include <iostream>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkGenericDataObjectReader.h>

class MapRefiner {
public:
  MapRefiner(ros::NodeHandle &nh);
  void refineMap();

private:
  cv::Mat color_lut_;
  std::map<uint32_t, uint8_t> class_lut_;
  ros::NodeHandle nh_;
  size_t num_classes_, num_exclusive_classes_;
  float res_;

  void loadSemOccGrid(const std::string &path, std::vector<cv::Mat> &sem_maps, const cv::Size &size);
  void loadOriginalMap(const std::string &path, std::vector<cv::Mat> &sem_maps);
  void saveUpdatedMaps(const std::string &path, const std::vector<cv::Mat> &sem_maps,
                       const std::vector<cv::Mat> &original_maps);
};

MapRefiner::MapRefiner(ros::NodeHandle &nh) {
  nh_ = nh;

  num_classes_ = 6;
  num_exclusive_classes_ = 4;

  color_lut_ = cv::Mat::ones(256, 1, CV_8UC3)*255;
  color_lut_.at<cv::Vec3b>(0) = cv::Vec3b(255,255,255); //unlabeled
  color_lut_.at<cv::Vec3b>(1) = cv::Vec3b(0,100,0);     //terrain
  color_lut_.at<cv::Vec3b>(2) = cv::Vec3b(255,0,0);     //road
  color_lut_.at<cv::Vec3b>(3) = cv::Vec3b(255,0,255);   //dirt
  color_lut_.at<cv::Vec3b>(4) = cv::Vec3b(0,0,255);     //building
  color_lut_.at<cv::Vec3b>(5) = cv::Vec3b(0,255,0);     //veg
  color_lut_.at<cv::Vec3b>(6) = cv::Vec3b(255,255,0);   //car

  for (int i=1; i<=num_classes_; i++) {
    cv::Vec3b color = color_lut_.at<cv::Vec3b>(i);
    //bgr
    uint32_t color_proc = (static_cast<uint32_t>(color[0]) << 16) + 
                          (static_cast<uint32_t>(color[1]) << 8) +
                           static_cast<uint32_t>(color[2]);
    class_lut_[color_proc] = i-1;
  }
}

void MapRefiner::loadSemOccGrid(const std::string &path, std::vector<cv::Mat> &sem_maps, const cv::Size &size) {
  vtkSmartPointer<vtkGenericDataObjectReader> reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
  reader->SetFileName(path.c_str());
  reader->Update();

  if (!reader->IsFilePolyData()) {
    ROS_ERROR("occ grid file formatted incorrectly");
    return;
  }

  for (size_t i=0; i<num_classes_; i++) {
    sem_maps.push_back(cv::Mat(size, CV_8UC1, cv::Scalar(0)));
  }
  
  vtkPolyData *data = reader->GetPolyDataOutput();
  vtkPoints *pts = data->GetPoints();
  vtkDataArray *classes = data->GetPointData()->GetScalars();

  double pt[3];
  int ind[3];
  for (size_t i=0; i<data->GetNumberOfPoints(); i++) {
    data->GetPoint(i, pt);
    for (size_t dim=0; dim<3; dim++) {
      ind[dim] = static_cast<int>(std::floor(pt[dim]/res_));
    }
    if (ind[0] < 0 || ind[0] >= size.width || ind[1] < 0 || ind[1] >= size.height)
      continue;

    classes->GetTuple(i, pt);
    //data is rgb
    uint32_t color_proc = (static_cast<uint32_t>(std::floor(pt[2])) << 16) + 
                          (static_cast<uint32_t>(std::floor(pt[1])) << 8) +
                           static_cast<uint32_t>(std::floor(pt[0]));
    if (class_lut_.count(color_proc) < 1) continue;

    sem_maps[class_lut_[color_proc]].at<uint8_t>(ind[1], ind[0]) += 1;
  }
}

void MapRefiner::loadOriginalMap(const std::string &path, std::vector<cv::Mat> &sem_maps) {
  for (size_t i=0; i<num_classes_; i++) {
    cv::Mat class_img = cv::imread(path+"/class"+std::to_string(i)+".png", cv::IMREAD_GRAYSCALE);
    sem_maps.push_back(class_img);
  }
}

void MapRefiner::saveUpdatedMaps(const std::string &path, const std::vector<cv::Mat> &sem_maps,
                                 const std::vector<cv::Mat> &original_maps) {
  cv::Mat map_viz(sem_maps[0].size(), CV_8UC3);
  std::vector<cv::Mat> refined_maps;
  for (const auto& map : original_maps) {
    refined_maps.push_back(map);
  }

  for (size_t x=0; x<sem_maps[0].size().height; x++) {
    for (size_t y=0; y<sem_maps[0].size().width; y++) {
      //First check if we have any info here at all
      bool have_new_data = false;
      for (size_t cls=0; cls<num_classes_; cls++) {
        if (sem_maps[cls].at<uint8_t>(x,y) > 0) {
          have_new_data = true;
          break;
        }
      }

      if (have_new_data) {
        //Can only have one of the exclusive classes set
        size_t best_ex_class_count = 0;
        size_t best_ex_class = num_classes_+1;
        for (size_t cls=0; cls<num_exclusive_classes_; cls++) {
          if (sem_maps[cls].at<uint8_t>(x,y) > best_ex_class_count) {
            best_ex_class = cls;
            best_ex_class_count = sem_maps[cls].at<uint8_t>(x,y);
          }
        }
        
        //Set the best exclusive class to occupied
        if (best_ex_class < num_classes_) {
          for (size_t cls=0; cls<num_exclusive_classes_; cls++) {
            if (cls == best_ex_class) {
              refined_maps[cls].at<uint8_t>(x,y) = 0;
            } else {
              refined_maps[cls].at<uint8_t>(x,y) = 255;
            }
          }
        }

        //Handle non-exclusive classes
        for (size_t cls=num_exclusive_classes_; cls<num_classes_; cls++) {
          if (sem_maps[cls].at<uint8_t>(x,y) > 0) {
            refined_maps[cls].at<uint8_t>(x,y) = 0;
          } else {
            refined_maps[cls].at<uint8_t>(x,y) = 255;
          }
        }
      }

      //Handle viz
      for (size_t cls=0; cls<num_classes_; cls++) {
        if (refined_maps[cls].at<uint8_t>(x,y) < 255) {
          map_viz.at<cv::Vec3b>(x,y) = color_lut_.at<cv::Vec3b>(cls+1);
        }
      }
    }
  }

  cv::imwrite(path+"/refined_map_viz.png", map_viz);

  for (size_t cls=0; cls<num_classes_; cls++) {
    cv::imwrite(path+"/refined_class"+std::to_string(cls)+".png", refined_maps[cls]);
  }
}

void MapRefiner::refineMap() {
  std::string sem_occ_grid_path, original_map_path;
  if (!nh_.getParam("sem_occ_grid_path", sem_occ_grid_path)) {
    ROS_ERROR("Need to specify path to occ grid");
    return;
  }
  if (!nh_.getParam("original_map_path", original_map_path)) {
    ROS_ERROR("Need to specify path to original map");
    return;
  }
  nh_.param<float>("res", res_, 1);

  std::vector<cv::Mat> sem_occ_map, original_map;

  ROS_INFO("Loading data...");
  loadOriginalMap(original_map_path, original_map);
  loadSemOccGrid(sem_occ_grid_path, sem_occ_map, original_map[0].size());

  ROS_INFO("Saving data...");
  saveUpdatedMaps(original_map_path, sem_occ_map, original_map);
  ROS_INFO("Complete");
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "refine_map");
  ros::NodeHandle nh("~");

  MapRefiner refiner(nh);
  refiner.refineMap();
  return 0;
}
