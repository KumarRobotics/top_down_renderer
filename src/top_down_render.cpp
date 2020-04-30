#include "top_down_render/top_down_render.h"

TopDownRender::TopDownRender(ros::NodeHandle &nh) {
  nh_ = nh;
}

void TopDownRender::initialize() {
  pc_sub_ = nh_.subscribe<pcl::PointCloud<pcl::PointXYZRGB>>("pc", 10, &TopDownRender::pcCallback, this);
  gt_pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("gt_pose", 10, &TopDownRender::gtPoseCallback, this);
  it_ = new image_transport::ImageTransport(nh_);
  img_pub_ = it_->advertise("img", 1);
  scan_pub_ = it_->advertise("scan", 1);
  geo_scan_pub_ = it_->advertise("geo_scan", 1);
  map_pub_ = it_->advertise("map_max", 1);

  //This order determines priority as well
  color_lut_ = cv::Mat::ones(256, 1, CV_8UC3)*255;
  color_lut_.at<cv::Vec3b>(0) = cv::Vec3b(255,255,255); //unlabeled
  color_lut_.at<cv::Vec3b>(1) = cv::Vec3b(0,100,0);     //terrain
  color_lut_.at<cv::Vec3b>(2) = cv::Vec3b(255,0,0);     //road
  color_lut_.at<cv::Vec3b>(3) = cv::Vec3b(255,0,255);   //dirt
  color_lut_.at<cv::Vec3b>(4) = cv::Vec3b(0,0,255);     //building
  color_lut_.at<cv::Vec3b>(5) = cv::Vec3b(0,255,0);     //veg
  color_lut_.at<cv::Vec3b>(6) = cv::Vec3b(255,255,0);   //car

  std::string map_path;
  nh_.getParam("map", map_path);
  background_img_ = cv::imread(map_path+".png", cv::IMREAD_COLOR);

  float svg_res, raster_res;
  nh_.getParam("svg_res", svg_res);
  nh_.getParam("raster_res", raster_res);

  int svg_origin_x, svg_origin_y;
  nh_.getParam("svg_origin_x", svg_origin_x);
  nh_.getParam("svg_origin_y", svg_origin_y);
  map_center_ = cv::Point(svg_origin_x, background_img_.size().height-svg_origin_y);

  map_ = new TopDownMapPolar(map_path+".svg", color_lut_, 6, 4, svg_res, raster_res);
  filter_ = new ParticleFilter(3000, background_img_.size().width/svg_res, background_img_.size().height/svg_res, map_);
  renderer_ = new ScanRendererPolar(false);

  ROS_INFO_STREAM("Setup complete");

  //DEBUG FOR VISUALIZATION
  //ros::Rate rate(1);
  //while (ros::ok()) {
  //  ROS_INFO("Debug loop");
  //  std_msgs::Header img_header;
  //  //image 1006 x 633
  //  map_->samplePtsPolar(Eigen::Vector2i(100, 50), 1, 2*M_PI/100);
  //  publishLocalMap(100, 50, Eigen::Vector2f(547/2.64, 270/2.64), 1., img_header);
  //  rate.sleep();
  //}
}

void TopDownRender::publishSemanticTopDown(std::vector<Eigen::ArrayXXf> &top_down, std_msgs::Header &header) {
  cv::Mat map_color;
  visualize(top_down, map_color);

  //Convert to ROS and publish
  sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", map_color).toImageMsg();
  img_msg->header = header;
  scan_pub_.publish(img_msg);
}

void TopDownRender::publishGeometricTopDown(std::vector<Eigen::ArrayXXf> &top_down, std_msgs::Header &header) {
  cv::Mat map_color;
  visualize(top_down, map_color);

  //Convert to ROS and publish
  sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", map_color).toImageMsg();
  img_msg->header = header;
  geo_scan_pub_.publish(img_msg);
}

cv::Mat TopDownRender::visualizeAnalog(Eigen::ArrayXXf &cls, float scale) {
  cv::Mat map_mat(cls.cols(), cls.rows(), CV_32FC1, (void*)cls.data());
  cv::Mat map_char, map_color;
  map_mat.convertTo(map_char, CV_8UC1, 255./scale);
  cv::cvtColor(map_char, map_color, cv::COLOR_GRAY2BGR);

  return map_color;
}

void TopDownRender::visualize(std::vector<Eigen::ArrayXXf> &classes, cv::Mat &img) {
  cv::Mat map_mat(classes[0].cols(), classes[0].rows(), CV_8UC1, cv::Scalar(0));

  for (int idx=0; idx<map_mat.size().width; idx++) {
    for (int idy=0; idy<map_mat.size().height; idy++) {
      char best_cls = 0;
      float best_cls_score = -std::numeric_limits<float>::infinity();
      float worst_cls_score = std::numeric_limits<float>::infinity();
      char cls_id = 1;
      for (auto cls : classes) {
        if (cls(idx, idy) >= best_cls_score) {
          best_cls_score = cls(idx, idy);
          best_cls = cls_id;
        }
        if (cls(idx, idy) < worst_cls_score) {
          worst_cls_score = cls(idx, idy);
        }
        cls_id++;
      }

      if (best_cls_score == worst_cls_score) {
        //All scores are the same, show white
        map_mat.at<uint8_t>(idy, idx) = 0;
      } else {
        map_mat.at<uint8_t>(idy, idx) = best_cls;
      }
    }
  }

  cv::Mat map_multichannel;
  cv::cvtColor(map_mat, map_multichannel, cv::COLOR_GRAY2BGR);
  cv::LUT(map_multichannel, color_lut_, img);
}

//Debug function
void TopDownRender::publishLocalMap(int h, int w, Eigen::Vector2f center, float res, std_msgs::Header &header) {
  std::vector<Eigen::ArrayXXf> classes;
  for (int i=0; i<map_->numClasses(); i++) {
    Eigen::ArrayXXf cls(h, w);
    classes.push_back(cls);
  }
  map_->getLocalMap(center, classes);

  //Invert for viz
  for (int i=0; i<classes.size(); i++) {
    classes[i] = -1*classes[i];
  }
  ROS_INFO("viz");

  cv::Mat map_color;
  visualize(classes, map_color);

  //Convert to ROS and publish
  sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", map_color).toImageMsg();
  img_msg->header = header;
  map_pub_.publish(img_msg);
}

void TopDownRender::updateFilter(std::vector<Eigen::ArrayXXf> &top_down, 
                                 std::vector<Eigen::ArrayXXf> &top_down_geo, float res,
                                 std_msgs::Header &header) {
  auto start = std::chrono::high_resolution_clock::now();
  filter_->update(top_down, top_down_geo, res);
  auto stop = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  ROS_INFO_STREAM("Filter update " << dur.count() << " ms");

  filter_->propagate();

  cv::Mat background_copy = background_img_.clone();
  filter_->visualize(background_copy);

  //Draw gt pose
  Eigen::Vector2f front(2,0);
  front = gt_pose_.linear()*front;
  cv::Point img_rot(-front[0]*map_->scale(), front[1]*map_->scale());
  cv::Point img_pos(-gt_pose_.translation()[0]*map_->scale(), gt_pose_.translation()[1]*map_->scale());
  cv::arrowedLine(background_copy, map_center_+img_pos-img_rot, map_center_+img_pos+img_rot, 
                  cv::Scalar(0,255,0), 2, CV_AA, 0, 0.3);

  //Publish visualization
  sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", background_copy).toImageMsg();
  img_msg->header = header;
  img_pub_.publish(img_msg);
}

void TopDownRender::pcCallback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud) {
  auto start = std::chrono::high_resolution_clock::now();

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
  ne.setMaxDepthChangeFactor(1.0);
  ne.setNormalSmoothingSize(10.0);
  ne.setBorderPolicy(ne.BORDER_POLICY_MIRROR);
  ne.setInputCloud(cloud);
  ne.compute(*normals);

  //Generate top down render and remap
  std::vector<Eigen::ArrayXXf> top_down, top_down_geo;
  for (int i=0; i<map_->numClasses(); i++) {
    Eigen::ArrayXXf img(100, 50);
    top_down.push_back(img);
  }
  for (int i=0; i<2; i++) {
    Eigen::ArrayXXf img(100, 50);
    top_down_geo.push_back(img);
  }

  ROS_INFO_STREAM("Starting render");
  renderer_->renderSemanticTopDown(cloud, normals, 1, 2*M_PI/100, top_down);
  renderer_->renderGeometricTopDown(cloud, 1, 2*M_PI/100, top_down_geo);

  //convert pointcloud header to ROS header
  std_msgs::Header img_header;
  publishSemanticTopDown(top_down, img_header);
  publishGeometricTopDown(top_down_geo, img_header);

  pcl_conversions::fromPCL(cloud->header, img_header);

  auto stop = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  ROS_INFO_STREAM("Render took " << dur.count() << " ms");

  //publishLocalMap(50, 50, Eigen::Vector2f(575/2.64, 262/2.64), 1, img_header);
  updateFilter(top_down, top_down_geo, current_res_, img_header);
  Eigen::Matrix2f cov;
  filter_->computeCov(cov);

  if (std::max(cov(0,0), cov(1,1)) > 15 && current_res_ < 2) {
    //cov big, expand local region
    current_res_ += 0.05;
  } else if (current_res_ > 0.5) {
    //we gucci, shrink to refine
    current_res_ -= 0.02;
  }

  //Normal visualization
  //pcl::visualization::PCLVisualizer viewer("PCL Viewer");
  //viewer.setBackgroundColor (0.0, 0.0, 0.0);
  //viewer.addPointCloud<pcl::PointXYZRGB>(cloud);
  //viewer.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 1, 1, "normals");

  //while (!viewer.wasStopped ())
  //{
  //  viewer.spinOnce ();
  //}
}

void TopDownRender::gtPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose) {
  //Convert to Eigen
  Eigen::Affine3f gt_pose3 = Eigen::Affine3f::Identity();
  gt_pose3 *= Eigen::Quaternionf(pose->pose.orientation.w, 
                                 pose->pose.orientation.x, 
                                 pose->pose.orientation.y, 
                                 pose->pose.orientation.z);
  gt_pose3.translation() << pose->pose.position.x, pose->pose.position.y, pose->pose.position.z;

  //Now we have to project into 2D
  Eigen::Vector3f x_axis = gt_pose3.linear()*Eigen::Vector3f::UnitX();
  float theta = atan2(x_axis[1], x_axis[0]);

  gt_pose_.translation() = gt_pose3.translation().head(2);
  gt_pose_.linear() << cos(theta), -sin(theta),
                       sin(theta), cos(theta);
}

