#include "top_down_render/top_down_render.h"
#include <yaml-cpp/yaml.h>

TopDownRender::TopDownRender(ros::NodeHandle &nh) {
  nh_ = nh;
}

void TopDownRender::initialize() {
  last_prior_pose_ = Eigen::Affine3f::Identity();
  nh_.param<bool>("use_motion_prior", use_motion_prior_, false);
  pc_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(
      "pc", 10, &TopDownRender::pcCallback, this);
  motion_prior_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(
      "motion_prior", 10, &TopDownRender::motionPriorCallback, this);

  gt_pose_ = Eigen::Affine2f::Identity();
  gt_pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("gt_pose", 10, 
      &TopDownRender::gtPoseCallback, this);

  // Want to latch output
  pose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose_est", 1, true);
  scale_pub_ = nh_.advertise<std_msgs::Float32>("scale", 1);
  it_ = new image_transport::ImageTransport(nh_);
  map_viz_pub_ = it_->advertise("map_viz", 1);
  scan_pub_ = it_->advertise("scan", 1);
  geo_scan_pub_ = it_->advertise("geo_scan", 1);
  debug_pub_ = it_->advertise("debug", 1);

  std::string world_config_path;
  nh_.getParam("world_config_path", world_config_path);

  semantics_manager::MapConfig map_params(semantics_manager::getMapPath(world_config_path));
  semantics_manager::ClassConfig class_params(semantics_manager::getClassesPath(world_config_path));
  color_lut_ = class_params.color_lut;
  auto top_down_map_params = getTopDownMapParams(class_params, map_params);

  // Deprecated, but leaving here for now
  int svg_origin_x, svg_origin_y;
  nh_.param<int>("svg_origin_x", svg_origin_x, 0);
  nh_.param<int>("svg_origin_y", svg_origin_y, 0);

  nh_.param<std::string>("map_frame", map_frame_, "map");
  nh_.param<std::string>("map_viz_frame", map_viz_frame_, "sem_map");

  nh_.param<float>("range_scale_min", range_scale_min_, 0.5);
  nh_.param<float>("range_scale_max", range_scale_max_, 4);
  current_range_scale_ = range_scale_max_;

  //Get filter parameters
  auto filter_params = getFilterParams(class_params, map_params);

  int particle_count;
  nh_.param<int>("particle_count", particle_count, 20000);

  // Convert to flatten_lut format
  unflatten_lut_ = class_params.flattened_to_class;
  flatten_lut_ = -Eigen::VectorXi::Ones(256);
  int map_class = 0;
  for (auto flattened_class : class_params.class_to_flattened) {
    flatten_lut_[map_class] = flattened_class;
    ++map_class;
  }

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

  // Scaling factor for visualization
  float map_pub_resolution = 1;
  nh_.param<float>("map_pub_resolution", map_pub_resolution, 1);
  if (filter_params.fixed_scale > 0) {
    map_pub_scale_ = map_pub_resolution / filter_params.fixed_scale;
  } else {
    // If we don't know the scale now, then just publish at native
    map_pub_scale_ = 1;
  }

  if (map_params.dynamic) {
    aerial_map_sub_ = nh_.subscribe<grid_map_msgs::GridMap>("aerial_map", 5,
        &TopDownRender::aerialMapCallback, this);
  } else {
    // Load background map
    background_img_ = cv::imread(map_params.viz_path, cv::IMREAD_COLOR);

    cv::Mat background_copy_small;
    cv::resize(background_img_, background_copy_small,
        cv::Size(static_cast<int>(background_img_.cols*map_pub_scale_), 
                 static_cast<int>(background_img_.rows*map_pub_scale_)));
    //Publish the map image
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), 
        "bgr8", background_copy_small).toImageMsg();
    map_viz_pub_.publish(img_msg);
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
    setw(width) << "[ROS] map_path: " << 
    (top_down_map_params.map_path == "" ? "dynamic" : top_down_map_params.map_path) << endl <<
    setw(width) << "[ROS] map_resolution: " << filter_params.fixed_scale << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] particle_count: " << particle_count << endl <<
    setw(width) << "[ROS] filter_pos_cov: " << filter_params.pos_cov << endl <<
    setw(width) << "[ROS] filter_theta_cov: " << filter_params.theta_cov << endl <<
    setw(width) << "[ROS] filter_regularization: " << filter_params.regularization << endl <<
    setw(width) << "[ROS] filter_force_on_map: " << filter_params.force_on_map << endl <<
    setw(width) << "[ROS] filter_scale_log_min: " << filter_params.scale_log_min << endl <<
    setw(width) << "[ROS] filter_scale_log_max: " << filter_params.scale_log_max << endl <<
    setw(width) << "[ROS] conf_factor: " << conf_factor_ << endl <<
    setw(width) << "[ROS] out_of_bounds_const: " << top_down_map_params.out_of_bounds_const << endl <<
    setw(width) << "[ROS] target_uncertainty_m: " << target_uncertainty_m_ << endl <<
    setw(width) << "[ROS] range_scale_min: " << range_scale_min_ << endl <<
    setw(width) << "[ROS] range_scale_max: " << range_scale_max_ << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] init_pos_px_x: " << filter_params.init_pos_px_x << endl <<
    setw(width) << "[ROS] init_pos_px_y: " << filter_params.init_pos_px_y << endl <<
    setw(width) << "[ROS] init_pos_px_cov: " << filter_params.init_pos_px_cov << endl <<
    setw(width) << "[ROS] init_pos_m_x: " << filter_params.init_pos_m_x << endl <<
    setw(width) << "[ROS] init_pos_m_y: " << filter_params.init_pos_m_y << endl <<
    setw(width) << "[ROS] init_pos_deg_theta: " << filter_params.init_pos_deg_theta << endl <<
    setw(width) << "[ROS] init_pos_deg_cov: " << filter_params.init_pos_deg_cov << endl <<
    "[ROS] ===============================" << endl <<
    setw(width) << "[ROS] map_frame: " << map_frame_ << endl <<
    setw(width) << "[ROS] map_viz_frame: " << map_viz_frame_ << endl <<
    setw(width) << "[ROS] use_motion_prior: " << use_motion_prior_ << endl <<
    setw(width) << "[ROS] svg_origin_x: " << svg_origin_x << endl <<
    setw(width) << "[ROS] svg_origin_y: " << svg_origin_y << endl <<
    setw(width) << "[ROS] map_pub_scale: " << map_pub_scale_ << endl <<
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
  for (int class_id : class_params.flattened_to_class) {
    if (class_params.exclusivity[class_id]) {
      params.exclusive_classes.push_back(class_params.class_to_flattened[class_id]);
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

  nh_.param<float>("conf_factor", conf_factor_, 1);

  std::string tmp_ext_buf;
  nh_.getParam("init_pos_px_x", tmp_ext_buf);
  if (tmp_ext_buf == "none") {
    filter_params.init_pos_px_x = -1;
    filter_params.init_pos_px_y = -1;
  } else {
    nh_.param<float>("init_pos_px_x", filter_params.init_pos_px_x, -1);
    nh_.param<float>("init_pos_px_y", filter_params.init_pos_px_y, -1);
  }
  nh_.param<float>("init_pos_px_cov", filter_params.init_pos_px_cov, -1);

  constexpr float inf = std::numeric_limits<float>::infinity();
  tmp_ext_buf = "";
  nh_.getParam("init_pos_m_x", tmp_ext_buf);
  if (tmp_ext_buf == "none") {
    filter_params.init_pos_m_x = inf;
    filter_params.init_pos_m_y = inf;
  } else {
    nh_.param<float>("init_pos_m_x", filter_params.init_pos_m_x, inf);
    nh_.param<float>("init_pos_m_y", filter_params.init_pos_m_y, inf);
  }
  tmp_ext_buf = "";
  nh_.getParam("init_pos_deg_theta", tmp_ext_buf);
  if (tmp_ext_buf == "none") {
    filter_params.init_pos_deg_theta = inf;
    filter_params.init_pos_deg_cov = 10;
  } else {
    nh_.param<float>("init_pos_deg_theta", filter_params.init_pos_deg_theta, inf);
    nh_.param<float>("init_pos_deg_cov", filter_params.init_pos_deg_cov, 10);
  }

  nh_.param<bool>("filter_force_on_map", filter_params.force_on_map, false);
  filter_params.fixed_scale = map_params.resolution;
  nh_.param<float>("filter_scale_log_min", filter_params.scale_log_min, -0.1);
  nh_.param<float>("filter_scale_log_max", filter_params.scale_log_max, 1);

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
      int best_cls = 255;
      float best_cls_score = -std::numeric_limits<float>::infinity();
      float worst_cls_score = std::numeric_limits<float>::infinity();
      int cls_id = 0;
      for (auto cls : classes) {
        if (cls(idx, idy) >= best_cls_score) {
          best_cls_score = cls(idx, idy);
          best_cls = cls_id;
        }
        if (cls(idx, idy) < worst_cls_score) {
          worst_cls_score = cls(idx, idy);
        }
        ++cls_id;
      }

      if (best_cls_score == worst_cls_score) {
        //All scores are the same, unknown class
        map_mat.at<uint8_t>(idy, idx) = 255;
      } else {
        map_mat.at<uint8_t>(idy, idx) = unflatten_lut_[best_cls];
      }
    }
  }

  color_lut_.ind2Color(map_mat, img);
}

//Debug function
void TopDownRender::publishLocalMap(int h, int w, Eigen::Vector2f center, float res, const std_msgs::Header &header, TopDownMap *map) {
  std::vector<Eigen::ArrayXXf> classes;
  for (int i=0; i<map->numClasses(); i++) {
    Eigen::ArrayXXf cls(h, w);
    classes.push_back(cls);
  }
  Eigen::ArrayXXc mask(h, w);
  map->getLocalMap(center, 0, res, classes, mask);

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

void TopDownRender::publishPoseEst(const std_msgs::Header &header) {
  Eigen::Matrix4f cov;
  filter_->computeMeanCov(cov);

  float scale = filter_->scale();
  float scale_2 = scale*scale;
  if (std::max(cov(0,0), cov(1,1))/scale_2 > std::pow(target_uncertainty_m_, 2) && 
      current_range_scale_ < range_scale_max_) 
  {
    //cov big, expand local region
    current_range_scale_ += 0.05;
  } else if (current_range_scale_ > range_scale_min_) {
    //we gucci, shrink to refine
    current_range_scale_ -= 0.02;
  }

  if (filter_->numParticles() < 1) {
    // Haven't converged yet
    return;
  }

  //get mean likelihood state
  Eigen::Vector4f ml_state;
  filter_->meanLikelihood(ml_state);

  if (cov(3,3) < 0.003*ml_state[3] && !filter_->isScaleFrozen()) {
    //freeze scale
    ROS_INFO_STREAM("\033[36m" << "[XView] Fixed Scale: " << ml_state[3] << "\033[0m");
    filter_->freezeScale();
  }

  //Only publish if converged to unimodal dist
  if (cov(0,0)/scale_2 < 40 && cov(1,1)/scale_2 < 40 && cov(2,2) < 0.5 && filter_->scale() > 0) {
    is_converged_ = true;
  }

  if (is_converged_) {
    geometry_msgs::PoseWithCovarianceStamped pose;
    pose.header = header;
    pose.header.frame_id = map_frame_;

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
    published_pose_ = true;
  }

  geometry_msgs::TransformStamped map_svg_transform;
  map_svg_transform.header.stamp = header.stamp;
  map_svg_transform.header.frame_id = map_frame_;
  map_svg_transform.child_frame_id = map_viz_frame_;
  map_svg_transform.transform.translation.x = (background_img_.size().width/2-map_center_.x)/scale;
  map_svg_transform.transform.translation.y = -(background_img_.size().height/2-map_center_.y)/scale;
  map_svg_transform.transform.translation.z = -2;
  map_svg_transform.transform.rotation.x = 1; //identity rot
  tf2_broadcaster_->sendTransform(map_svg_transform);
}

void TopDownRender::updateFilter(std::vector<Eigen::ArrayXXf> &top_down, 
                                 std::vector<Eigen::ArrayXXf> &top_down_geo, float res,
                                 Eigen::Affine3f &motion_prior, const std_msgs::Header &header) {
  auto start = std::chrono::high_resolution_clock::now();
  //Project 3d prior to 2d
  Eigen::Vector2f motion_priort = motion_prior.translation().head<2>();
  Eigen::Vector3f proj_rot = motion_prior.rotation() * Eigen::Vector3f::UnitX();
  float motion_priora = std::atan2(proj_rot[1], proj_rot[0]);
  ROS_INFO_STREAM("\033[36m" << "[XView] Motion Prior trans:" << motion_priort.transpose() << 
      ", rot: " << motion_priora << "\033[0m");
  filter_->propagate(motion_priort, motion_priora);

  filter_->update(top_down, top_down_geo, res);
  auto stop = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  ROS_INFO_STREAM("\033[36m" << "[XView] Filter update " << dur.count() << " ms" << "\033[0m");

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
  map_viz_pub_.publish(img_msg);
}

void TopDownRender::pcCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
  if (!use_motion_prior_) {
    takeStep(cloud_msg, {});
    return;
  }

  // Loop starting with most recent
  for (auto mp_it = motion_prior_buf_.rbegin(); mp_it != motion_prior_buf_.rend(); ++mp_it) {
    if (cloud_msg->header.stamp == (*mp_it)->header.stamp) {
      takeStep(cloud_msg, *mp_it);
      motion_prior_buf_.erase(motion_prior_buf_.begin(), mp_it.base());
      break;
    }
  }
}

void TopDownRender::motionPriorCallback(
    const geometry_msgs::PoseStamped::ConstPtr &motion_prior) 
{
  if (!published_pose_ && filter_->numParticles() > 0) {
    // Publish initial position
    publishPoseEst(motion_prior->header);
  }

  if (last_prior_pose_.matrix().isIdentity()) {
    // Initialize the last prior pose so we are measuring relative motion
    Eigen::Affine3d motion_prior_eig;
    tf::poseMsgToEigen(motion_prior->pose, motion_prior_eig);
    last_prior_pose_ = motion_prior_eig.cast<float>();
  }

  if (use_motion_prior_) {
    motion_prior_buf_.push_back(motion_prior);
  }
}

void TopDownRender::takeStep(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
                             const geometry_msgs::PoseStamped::ConstPtr& motion_prior) {
  ROS_INFO_STREAM("\033[36m" << "[XView] Received Point Cloud" << "\033[0m");
  if (!map_->haveMap()) {
    ROS_WARN_STREAM("[XView] No map received yet");
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

  ROS_INFO_STREAM("\033[36m" << "[XView] Starting Render" << "\033[0m");
  renderer_->renderSemanticTopDown(cloud_ptr, current_range_scale_, 2*M_PI/100, top_down);
  //renderer_->renderGeometricTopDown(cloud, current_range_scale_, 2*M_PI/100, top_down_geo);

  //convert pointcloud header to ROS header
  publishSemanticTopDown(top_down, cloud_msg->header);
  publishGeometricTopDown(top_down_geo, cloud_msg->header);

  auto stop = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  ROS_INFO_STREAM("\033[36m" << "[XView] Render took " << dur.count() << " ms" << "\033[0m");
  
  //Compute delta motion
  Eigen::Affine3d motion_prior_eig = Eigen::Affine3d::Identity();
  if (motion_prior) {
    tf::poseMsgToEigen(motion_prior->pose, motion_prior_eig);
  }
  Eigen::Affine3f delta_pose = last_prior_pose_.inverse() * motion_prior_eig.cast<float>();
  last_prior_pose_ = motion_prior_eig.cast<float>();

  //publishLocalMap(50, 50, Eigen::Vector2f(575/2.64, 262/2.64), 1, img_header);
  updateFilter(top_down, top_down_geo, current_range_scale_, delta_pose, cloud_msg->header);
  publishPoseEst(cloud_msg->header);

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

void TopDownRender::aerialMapCallback(const grid_map_msgs::GridMap::ConstPtr &map) {
  if (map->info.header.stamp.toNSec() <= last_map_stamp_) return;
  if (map->info.length_x <= 0 || map->info.length_y <= 0) return;

  ROS_INFO_STREAM("Got new map");
  //Convert to cv and rotate
  cv::Mat map_img;
  grid_map::GridMapComp::toImage(*map, {"semantics", "", "char"}, map_img);
  cv::rotate(map_img, map_img, cv::ROTATE_90_CLOCKWISE);

  color_lut_.ind2Color(map_img, background_img_);

  Eigen::Vector2i map_loc_eig(-map->info.pose.position.x, -map->info.pose.position.y);
  map_loc_eig *= filter_->scale();
  map_loc_eig += Eigen::Vector2i(map_img.size().width/2, map_img.size().height/2);

  map_center_ = cv::Point(map_loc_eig[0], map_img.size().height - map_loc_eig[1]);
  filter_->updateMap(map_img, map_loc_eig);
  last_map_stamp_ = map->info.header.stamp.toNSec();
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
