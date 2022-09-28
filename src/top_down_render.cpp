#include "top_down_render/top_down_render.h"
#include <yaml-cpp/yaml.h>

TopDownRender::TopDownRender(ros::NodeHandle &nh) {
  nh_ = nh;
}

void TopDownRender::initialize() {
  last_prior_pose_ = Eigen::Affine3f::Identity();
  bool use_motion_prior;
  nh_.param<bool>("use_motion_prior", use_motion_prior, false);
  if (use_motion_prior) {
    pc_sync_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, "pc", 50);
    motion_prior_sync_sub_ = new message_filters::Subscriber<geometry_msgs::PoseStamped>(nh_, "motion_prior", 50);
    scan_sync_sub_ = new message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, geometry_msgs::PoseStamped>(
                      *pc_sync_sub_, *motion_prior_sync_sub_, 50);
    scan_sync_sub_->registerCallback(&TopDownRender::pcCallback, this);
  } else {
    pc_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>("pc", 10, 
                std::bind(&TopDownRender::pcCallback, this, std::placeholders::_1, 
                          geometry_msgs::PoseStamped::ConstPtr()));
  }

  gt_pose_ = Eigen::Affine2f::Identity();
  gt_pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("gt_pose", 10, 
      &TopDownRender::gtPoseCallback, this);

  pose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose_est", 1);
  scale_pub_ = nh_.advertise<std_msgs::Float32>("scale", 1);
  it_ = new image_transport::ImageTransport(nh_);
  map_pub_ = it_->advertise("map", 1);
  scan_pub_ = it_->advertise("scan", 1);
  geo_scan_pub_ = it_->advertise("geo_scan", 1);
  debug_pub_ = it_->advertise("debug", 1);

  std::string world_config_path;
  nh_.getParam("world_config_path", world_config_path);

  semantics_manager::MapConfig map_params(semantics_manager::getMapPath(world_config_path));
  semantics_manager::ClassConfig class_params(semantics_manager::getClassesPath(world_config_path));
  auto top_down_map_params = getTopDownMapParams(class_params, map_params);

  // Deprecated, but leaving here for now
  int svg_origin_x, svg_origin_y;
  nh_.param<int>("svg_origin_x", svg_origin_x, 0);
  nh_.param<int>("svg_origin_y", svg_origin_y, 0);

  //Get filter parameters
  auto filter_params = getFilterParams(class_params, map_params);

  int particle_count;
  nh_.param<int>("particle_count", particle_count, 20000);

  //DEBUG FOR VISUALIZATION
  //ros::Rate rate(1);
  //TopDownMap *cart_map = new TopDownMap(map_path, color_lut_, 6, 6, raster_res);
  //while (ros::ok()) {
  //  ROS_INFO("Debug loop");
  //  //image 1006 x 633
  //  //map_->samplePtsPolar(Eigen::Vector2i(100, 50), 2*M_PI/100);
  //  publishLocalMap(500, 500, Eigen::Vector2f(500, 500), 1., img_msg->header, cart_map);
  //  //std::vector<int> classes;
  //  //map_->getClassesAtPoint(Eigen::Vector2f(1447/1.31, 523/1.31), classes);
  //  //for (auto cls : classes) {
  //  //  ROS_INFO_STREAM("class " << cls);
  //  //}
  //  rate.sleep();
  //}
  //END DEBUG

  map_ = new TopDownMapPolar(top_down_map_params);

  if (map_params.dynamic) {
    map_image_sub_ = nh_.subscribe<sensor_msgs::Image>("map_image", 1,
        &TopDownRender::mapImageCallback, this);
    map_loc_sub_ = nh_.subscribe<geometry_msgs::PointStamped>("map_loc", 50,
        &TopDownRender::mapLocCallback, this);
  } else {
    // Load background map
    nh_.param<float>("map_pub_scale", map_pub_scale_, 0.2);
    background_img_ = cv::imread(map_params.viz_path, cv::IMREAD_COLOR);

    cv::Mat background_copy_small;
    cv::resize(background_img_, background_copy_small,
        cv::Size(static_cast<int>(background_img_.cols*map_pub_scale_), 
                 static_cast<int>(background_img_.rows*map_pub_scale_)));
    //Publish the map image
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), 
        "bgr8", background_copy_small).toImageMsg();
    map_pub_.publish(img_msg);
  }
  map_center_ = cv::Point(svg_origin_x, background_img_.size().height-svg_origin_y);
  map_->samplePtsPolar(Eigen::Vector2i(100, 25), 2*M_PI/100);
  filter_ = new ParticleFilter(particle_count, map_, filter_params);
  renderer_ = new ScanRendererPolar(flatten_lut_);

  //static transform broadcaster for map viz
  tf2_broadcaster_ = new tf2_ros::TransformBroadcaster(); 

  constexpr int width = 30;
  using namespace std;
  ROS_INFO_STREAM("\033[32m" << "[XView]" << endl << "[ROS] ======== Configuration ========" << 
    endl << left << 
    setw(width) << "[ROS] world_config_path: " << world_config_path << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] particle_count: " << particle_count << endl <<
    setw(width) << "[ROS] filter_pos_cov: " << filter_params.pos_cov << endl <<
    "[ROS] ====== End Configuration ======" << "\033[0m");
}

TopDownMap::Params TopDownRender::getTopDownMapParams(
    const semantics_manager::ClassConfig& class_params,
    const semantics_manager::MapConfig& map_params) {
  TopDownMap::Params params;
  if (!map_params.dynamic) {
    if (map_params.svg_path != "") {
      params.map_path = map_params.svg_path;
    } else {
      params.map_path = map_params.raster_path;
    }
  }

  params.color_lut = class_params.color_lut;
  params.flatten_lut = class_params.class_to_flattened;
  params.num_classes = class_params.flattened_to_class.size();

  params.exclusive_classes.resize(params.num_classes);
  int flattened_id = 0;
  for (int class_id : class_params.flattened_to_class) {
    if (class_params.exclusivity[class_id]) {
      params.exclusive_classes.push_back(flattened_id);
    }
  }

  // This is really the scale factor for the map.  Fixed to 1 pretty much always
  // Leaving this in in case we ever want this feature for some reason.
  params.resolution = 1;
  nh_.param<float>("out_of_bounds_const", params.out_of_bounds_const);

  return params;
}

FilterParams TopDownRender::getFilterParams(
    const semantics_manager::ClassConfig& class_params,
    const semantics_manager::MapConfig& map_params) {
  FilterParams filter_params;

  nh_.param<float>("filter_pos_cov", filter_params.pos_cov, 0.3);
  nh_.param<float>("filter_theta_cov", filter_params.theta_cov, M_PI/100);
  nh_.param<float>("filter_regularization", filter_params.regularization, 0.15);
  filter_params.fixed_scale = map_params.resolution;

  nh_.param<float>("conf_factor", conf_factor_, 1);

  nh_.param<float>("init_pos_px_x", filter_params.init_pos_px_x, -1);
  nh_.param<float>("init_pos_px_y", filter_params.init_pos_px_y, -1);
  nh_.param<float>("init_pos_px_cov", filter_params.init_pos_px_cov, -1);

  constexpr float inf = std::numeric_limits<float>::infinity();
  nh_.param<float>("init_pos_m_x", filter_params.init_pos_m_x, inf);
  nh_.param<float>("init_pos_m_y", filter_params.init_pos_m_y, inf);
  nh_.param<float>("init_pos_deg_theta", filter_params.init_pos_deg_theta, inf);
  nh_.param<float>("init_pos_deg_cov", filter_params.init_pos_deg_cov, 10);

  int flattened_id = 0;
  for (int class_id : class_params.flattened_to_class) {
    filter_params.class_weights.push_back(class_params.loc_weight[class_id]);
  }

  return filter_params;
}

void TopDownRender::publishSemanticTopDown(std::vector<Eigen::ArrayXXf> &top_down, const std_msgs::Header &header) {
  cv::Mat map_color;
  visualize(top_down, map_color);

  //Convert to ROS and publish
  sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", map_color).toImageMsg();
  img_msg->header = header;
  scan_pub_.publish(img_msg);
}

void TopDownRender::publishGeometricTopDown(std::vector<Eigen::ArrayXXf> &top_down, const std_msgs::Header &header) {
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
void TopDownRender::publishLocalMap(int h, int w, Eigen::Vector2f center, float res, const std_msgs::Header &header, TopDownMap *map) {
  std::vector<Eigen::ArrayXXf> classes;
  for (int i=0; i<map->numClasses(); i++) {
    Eigen::ArrayXXf cls(h, w);
    classes.push_back(cls);
  }
  map->getLocalMap(center, 0, res, classes);

  //Invert for viz
  //for (int i=0; i<classes.size(); i++) {
  //  classes[i] = -1*classes[i];
  //}
  ROS_INFO("viz");

  cv::Mat map_color = visualizeAnalog(classes[1], 50);

  //Convert to ROS and publish
  sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", map_color).toImageMsg();
  img_msg->header = header;
  debug_pub_.publish(img_msg);
}

void TopDownRender::updateFilter(std::vector<Eigen::ArrayXXf> &top_down, 
                                 std::vector<Eigen::ArrayXXf> &top_down_geo, float res,
                                 Eigen::Affine3f &motion_prior, const std_msgs::Header &header) {
  auto start = std::chrono::high_resolution_clock::now();
  //Project 3d prior to 2d
  Eigen::Vector2f motion_priort = motion_prior.translation().head<2>();
  Eigen::Vector3f proj_rot = motion_prior.rotation() * Eigen::Vector3f::UnitX();
  float motion_priora = std::atan2(proj_rot[1], proj_rot[0]);
  ROS_INFO_STREAM(motion_priort << ", " << motion_priora);
  filter_->propagate(motion_priort, motion_priora);
  ROS_DEBUG("Filter propagate");

  filter_->update(top_down, top_down_geo, res);
  auto stop = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  ROS_INFO_STREAM("Filter update " << dur.count() << " ms");
  ROS_DEBUG("test");

  cv::Mat background_copy = background_img_.clone();
  filter_->visualize(background_copy);

  //Draw gt pose
  Eigen::Vector2f front(2,0);
  front = gt_pose_.linear()*front;
  cv::Point img_rot(front[0], -front[1]);
  cv::Point img_pos(gt_pose_.translation()[0], -gt_pose_.translation()[1]);
  cv::arrowedLine(background_copy, map_center_+img_pos-img_rot, map_center_+img_pos+img_rot, 
                  cv::Scalar(0,255,0), 2, CV_AA, 0, 0.3);

  cv::Mat background_copy_small;
  cv::resize(background_copy, background_copy_small,
      cv::Size(static_cast<int>(background_copy.cols*map_pub_scale_), 
               static_cast<int>(background_copy.rows*map_pub_scale_)));

  //Publish visualization
  sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", background_copy_small).toImageMsg();
  img_msg->header = header;
  map_pub_.publish(img_msg);
}

void TopDownRender::pcCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
                               const geometry_msgs::PoseStamped::ConstPtr& motion_prior) {
  ROS_INFO_STREAM("pc cb");
  if (!map_->haveMap()) {
    ROS_WARN_STREAM("No map received yet");
    return;  
  }

  auto start = std::chrono::high_resolution_clock::now();

  pcl::PointCloud<PointType>::Ptr cloud_ptr(new pcl::PointCloud<PointType>());
  pcl::fromROSMsg(*cloud_msg, *cloud_ptr);

  //pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  //pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  //ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
  //ne.setMaxDepthChangeFactor(1.0);
  //ne.setNormalSmoothingSize(10.0);
  //ne.setBorderPolicy(ne.BORDER_POLICY_MIRROR);
  //ne.setInputCloud(cloud);
  //ne.compute(*normals);

  //Generate top down render and remap
  std::vector<Eigen::ArrayXXf> top_down, top_down_geo;
  for (int i=0; i<map_->numClasses(); i++) {
    Eigen::ArrayXXf img(100, 25);
    top_down.push_back(img);
  }
  for (int i=0; i<2; i++) {
    Eigen::ArrayXXf img(100, 25);
    top_down_geo.push_back(img);
  }

  ROS_INFO_STREAM("Starting render");
  renderer_->renderSemanticTopDown(cloud_ptr, current_res_, 2*M_PI/100, top_down);
  //renderer_->renderGeometricTopDown(cloud, current_res_, 2*M_PI/100, top_down_geo);

  //convert pointcloud header to ROS header
  publishSemanticTopDown(top_down, cloud_msg->header);
  publishGeometricTopDown(top_down_geo, cloud_msg->header);

  auto stop = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  ROS_INFO_STREAM("Render took " << dur.count() << " ms");
  
  //Compute delta motion
  Eigen::Affine3d motion_prior_eig = Eigen::Affine3d::Identity();
  if (motion_prior) {
    tf::poseMsgToEigen(motion_prior->pose, motion_prior_eig);
  }
  Eigen::Affine3f delta_pose = last_prior_pose_.inverse() * motion_prior_eig.cast<float>();
  last_prior_pose_ = motion_prior_eig.cast<float>();

  //publishLocalMap(50, 50, Eigen::Vector2f(575/2.64, 262/2.64), 1, img_header);
  updateFilter(top_down, top_down_geo, current_res_, delta_pose, cloud_msg->header);
  Eigen::Matrix4f cov;
  filter_->computeMeanCov(cov);

  if (std::max(cov(0,0), cov(1,1)) > 15 && current_res_ < 4) {
    //cov big, expand local region
    current_res_ += 0.05;
  } else if (current_res_ > 0.5) {
    //we gucci, shrink to refine
    current_res_ -= 0.02;
  }

  if (filter_->numParticles() < 1) {
    // Haven't converged yet
    return;
  }

  //get mean likelihood state
  Eigen::Vector4f ml_state;
  filter_->meanLikelihood(ml_state);

  ROS_INFO_STREAM("scale uncertainty: " << cov(3,3));
  if (cov(3,3) < 0.003*ml_state[3]) {
    //freeze scale
    ROS_INFO_STREAM("scale: " << ml_state[3]);
    filter_->freezeScale();
  }

  //Only publish if converged to unimodal dist
  float scale = filter_->scale();
  float scale_2 = scale*scale;
  if (cov(0,0)/scale_2 < 40 && cov(1,1)/scale_2 < 40 && cov(2,2) < 0.5 && filter_->scale() > 0) {
    is_converged_ = true;
  }

  if (is_converged_) {
    geometry_msgs::PoseWithCovarianceStamped pose;
    pose.header = cloud_msg->header;
    pose.header.frame_id = "world";

    std_msgs::Float32 scale_msg;
    scale_msg.data = scale;
    scale_pub_.publish(scale_msg);

    //pose
    pose.pose.pose.position.x = (ml_state[0] - map_center_.x)/scale;
    pose.pose.pose.position.y = (ml_state[1] - (background_img_.size().height-map_center_.y))/scale;
    pose.pose.pose.position.z = 2;
    pose.pose.pose.orientation.x = 0;
    pose.pose.pose.orientation.y = 0;
    pose.pose.pose.orientation.z = sin(ml_state[2]/2);
    pose.pose.pose.orientation.w = cos(ml_state[2]/2);

    float conf_factor_2 = conf_factor_*conf_factor_;

    //cov
    pose.pose.covariance[0] = cov(0,0)/scale_2/conf_factor_2;
    pose.pose.covariance[1] = cov(0,1)/scale_2/conf_factor_2;
    pose.pose.covariance[5] = cov(0,2)/scale/conf_factor_;
    pose.pose.covariance[6] = cov(1,0)/scale_2/conf_factor_2;
    pose.pose.covariance[7] = cov(1,1)/scale_2/conf_factor_2;
    pose.pose.covariance[11] = cov(1,2)/scale/conf_factor_;
    pose.pose.covariance[30] = cov(2,0)/scale/conf_factor_;
    pose.pose.covariance[31] = cov(2,1)/scale/conf_factor_;
    pose.pose.covariance[35] = cov(2,2)/conf_factor_2;

    pose_pub_.publish(pose);
  }

  geometry_msgs::TransformStamped map_svg_transform;
  map_svg_transform.header.stamp = cloud_msg->header.stamp;
  map_svg_transform.header.frame_id = "world";
  map_svg_transform.child_frame_id = "sem_map";
  map_svg_transform.transform.translation.x = (background_img_.size().width/2-map_center_.x)/scale;
  map_svg_transform.transform.translation.y = -(background_img_.size().height/2-map_center_.y)/scale;
  map_svg_transform.transform.translation.z = -2;
  map_svg_transform.transform.rotation.x = 1; //identity rot
  tf2_broadcaster_->sendTransform(map_svg_transform);

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

void TopDownRender::mapImageCallback(const sensor_msgs::Image::ConstPtr &map) {
  if (map->header.stamp.toNSec() > last_map_stamp_) {
    // Initial sanity check
    if (map->height > 0 && map->width > 0) {
      ROS_DEBUG("Got map img");
      map_image_buf_.insert({map->header.stamp.toNSec(), map});
      processMapBuffers();
    }
  }
}

void TopDownRender::mapLocCallback(const geometry_msgs::PointStamped::ConstPtr &map_loc) {
  if (map_loc->header.stamp.toNSec() > last_map_stamp_) {
    ROS_DEBUG("Got map loc");
    map_loc_buf_.insert({map_loc->header.stamp.toNSec(), map_loc});
    processMapBuffers();
  }
}

void TopDownRender::processMapBuffers() {
  // Loop starting with most recent
  for (auto loc_it = map_loc_buf_.rbegin(); loc_it != map_loc_buf_.rend(); ++loc_it) {
    auto img = map_image_buf_.find(loc_it->first);
    if (img != map_image_buf_.end()) {
      ROS_INFO_STREAM("Got new map");
      //Convert to cv and rotate
      cv::Mat map_img;
      cv::rotate(cv_bridge::toCvShare(img->second, sensor_msgs::image_encodings::BGR8)->image, 
          map_img, cv::ROTATE_90_CLOCKWISE);

      cv::Mat viz_lut = cv::Mat::ones(256, 1, CV_8UC3)*255;
      for (int original_cls=0; original_cls<flatten_lut_.size(); original_cls++) {
        viz_lut.at<cv::Vec3b>(original_cls) = color_lut_.at<cv::Vec3b>(flatten_lut_[original_cls]);
      }
      cv::LUT(map_img, viz_lut, background_img_);

      Eigen::Vector2i map_loc_eig(-loc_it->second->point.x, -loc_it->second->point.y);
      map_loc_eig *= filter_->scale();
      map_loc_eig += Eigen::Vector2i(map_img.size().width/2, map_img.size().height/2);

      map_center_ = cv::Point(map_loc_eig[0], map_img.size().height - map_loc_eig[1]);
      filter_->updateMap(map_img, map_loc_eig);
      last_map_stamp_ = loc_it->first;
      
      //Clean up buffers
      map_image_buf_.erase(map_image_buf_.begin(), ++img);
      map_loc_buf_.erase(map_loc_buf_.begin(), loc_it.base());
      break;
    }
  }
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
