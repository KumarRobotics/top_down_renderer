#include "top_down_render/top_down_render.h"

TopDownRender::TopDownRender(ros::NodeHandle &nh) {
  nh_ = nh;
}

void TopDownRender::initialize() {
  pc_sub_ = nh_.subscribe<pcl::PointCloud<pcl::PointXYZRGB>>("pc", 10, &TopDownRender::pcCallback, this);
	it_ = new image_transport::ImageTransport(nh_);
	img_pub_ = it_->advertise("img", 1);
	scan_pub_ = it_->advertise("scan", 1);
	map_pub_ = it_->advertise("map_max", 1);
  
  //Fill org_pc_
  for (size_t xi=0; xi<100; xi++) {
    std::vector<pt_q_type> org_pc_x;
    for (size_t yi=0; yi<100; yi++) {
      pt_q_type q;
      org_pc_x.push_back(q);
    }
    org_pc_.push_back(org_pc_x);
  }

  flatten_lut_ = cv::Mat::zeros(256, 1, CV_8UC1);
  flatten_lut_.at<uint8_t>(0) = 1;   //ground
  flatten_lut_.at<uint8_t>(1) = 1;   //sidewalk
  flatten_lut_.at<uint8_t>(2) = 2;   //building
  flatten_lut_.at<uint8_t>(3) = 2;   //wall

  flatten_lut_.at<uint8_t>(7) = 0;   //vegetation
  flatten_lut_.at<uint8_t>(8) = 0;   //terrain
  flatten_lut_.at<uint8_t>(13) = 1;  //car
  flatten_lut_.at<uint8_t>(14) = 1;  //truck
  flatten_lut_.at<uint8_t>(15) = 1;  //bus

  color_lut_ = cv::Mat::ones(256, 1, CV_8UC3)*255;
  color_lut_.at<cv::Vec3b>(0) = cv::Vec3b(255,255,255); //unlabeled
  color_lut_.at<cv::Vec3b>(1) = cv::Vec3b(255,0,0);     //ground
  color_lut_.at<cv::Vec3b>(2) = cv::Vec3b(0,0,255);     //building
  color_lut_.at<cv::Vec3b>(3) = cv::Vec3b(0,255,0);     //veg
  color_lut_.at<cv::Vec3b>(4) = cv::Vec3b(255,255,0);   //car

  std::string map_path;
  nh_.getParam("map", map_path);
  background_img_ = cv::imread(map_path+".png", cv::IMREAD_COLOR);
  map_ = new TopDownMap(map_path+".svg", color_lut_, 4, 5.9);
  filter_ = new ParticleFilter(1000, background_img_.size().width/5.9, background_img_.size().height/5.9, map_);
}

void TopDownRender::renderTopDown(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud, 
									                pcl::PointCloud<pcl::Normal>::Ptr& normals,	
									                float side_length, Eigen::ArrayXXc &img) {
  size_t img_size = img.cols();

  //Generate bins of points
  for (size_t idx=0; idx<cloud->height*cloud->width; idx++) {
    auto pt = cloud->points[idx];
    if (pt.x == 0 && pt.y == 0) continue;

    int x_ind = std::round(pt.x/side_length)+img_size/2;
    int y_ind = std::round(pt.y/side_length)+img_size/2;
    if (x_ind >= 0 && x_ind < img_size && y_ind >= 0 && y_ind < img_size) {
      PointXYZClassNormal pt(cloud->points[idx], normals->points[idx]);
      org_pc_[x_ind][y_ind].push(pt);
    }
  }
  //Look at each bin, pick representative class
  for (size_t xi=0; xi<img_size; xi++) {
    for (size_t yi=0; yi<img_size; yi++) {
      //Look at top 10
      unsigned int num_floor = 0;
      unsigned int num_total = 0;
      std::map<uint32_t, unsigned int> class_cnt;
      for (size_t idx=0; org_pc_[xi][yi].size()>0; idx++) {
        auto pt = org_pc_[xi][yi].top();
        if (pt.pt_normal.normal_z > 0.9) num_floor++; 
        num_total++;

        if (idx < 5) {
          //Look at classification for top 5
          class_cnt[*reinterpret_cast<int*>(&pt.pt_xyz.rgb)]++;
        }
        org_pc_[xi][yi].pop();
      }
      uint32_t best_class = 255;
      unsigned int best_class_cnt = 0;
      for (auto x : class_cnt) {
        if (x.second > best_class_cnt) {
          best_class_cnt = x.second;
          best_class = x.first;
        }
      }

      if (num_floor >= num_total*0.9 && num_total > 1) {
        img(img_size-1-yi,xi) = 1;
      } else {
        img(img_size-1-yi,xi) = best_class&0xff;
      }
      //Clear queue
      pt_q_type().swap(org_pc_[xi][yi]);
    }
  }
}

void TopDownRender::publishTopDown(cv::Mat& top_down_img, std_msgs::Header &header) {
  cv::Mat top_down_multichannel, top_down_color;
  cv::cvtColor(top_down_img, top_down_multichannel, cv::COLOR_GRAY2BGR);
  cv::LUT(top_down_multichannel, color_lut_, top_down_color);
  
	//Convert to ROS and publish
	sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", top_down_color).toImageMsg();
  img_msg->header = header;
	scan_pub_.publish(img_msg);
}

//Really a debug function, should not be called in normal operation.  Very slow
void TopDownRender::publishHeatMap(Eigen::ArrayXXc &top_down, float local_res, float heatmap_res, cv::Rect roi, std_msgs::Header &header) {
  //Tests
  cv::Mat likelihood = cv::Mat::zeros(roi.width/heatmap_res, roi.height/heatmap_res, CV_32FC1);
  Eigen::ArrayXXc cls(top_down.rows(), top_down.cols());
  Eigen::ArrayXXc best_cls(top_down.rows(), top_down.cols());
  float best_cls_val = 1;
  for (int x=roi.x; x<roi.x+roi.width; x+=heatmap_res) { 
    for (int y=roi.y; y<roi.y+roi.height; y+=heatmap_res) { 
      float best_val = 1;
      for (float theta=0; theta<2*3.14; theta+=0.2) {
        Eigen::Vector2f center(x, y);

        map_->getLocalMap(center, theta, 1, cls);
        Eigen::ArrayXXc diff = cls.cwiseNotEqual(top_down).cast<uint8_t>() * top_down;
        float lh = static_cast<float>(diff.count())/top_down.count();
        if (lh < best_val) {
          best_val = lh;
        }
        if (lh < best_cls_val) {
          best_cls_val = lh;
          best_cls = cls;
        }
      }
      likelihood.at<float>(likelihood.size().width-1-(y-roi.y)/heatmap_res, (x-roi.x)/heatmap_res) = best_val;
      ROS_INFO_STREAM(x << ", " << y);
    }
  }
  cv::Mat likelihood_scaled, likelihood_char, heatmap, overlay;  
  cv::resize(likelihood, likelihood_scaled, cv::Size(roi.width*map_->scale(), roi.height*map_->scale()), 0, 0, cv::INTER_NEAREST);
  likelihood_scaled.convertTo(likelihood_char, CV_8UC1, 3*255);
  cv::applyColorMap(likelihood_char, heatmap, cv::COLORMAP_JET);

  roi.x *= map_->scale();
  roi.y = background_img_.size().height - roi.height*map_->scale()-roi.y*map_->scale();
  roi.width *= map_->scale();
  roi.height *= map_->scale();
  cv::addWeighted(background_img_(roi), 0.3, heatmap, 0.7, 0, overlay);

	//Convert to ROS and publish
	sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", overlay).toImageMsg();
  img_msg->header = header;
	img_pub_.publish(img_msg);

  cv::Mat map_mat(best_cls.cols(), best_cls.rows(), CV_8UC1, (void*)best_cls.data());
  cv::Mat map_multichannel, map_color;
  cv::cvtColor(map_mat, map_multichannel, cv::COLOR_GRAY2BGR);
  cv::LUT(map_multichannel, color_lut_, map_color);

	//Convert to ROS and publish
	img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", map_color).toImageMsg();
  img_msg->header = header;
	map_pub_.publish(img_msg);
}

void TopDownRender::updateFilter(Eigen::ArrayXXc &top_down, std_msgs::Header &header) {
  filter_->update(top_down);
  filter_->propagate();

  cv::Mat background_copy = background_img_.clone();
  filter_->visualize(background_copy);

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
  Eigen::ArrayXXc top_down(50,50);
  top_down.setZero();
	renderTopDown(cloud, normals, 1, top_down);
  cv::Mat top_down_img(top_down.cols(), top_down.rows(), CV_8UC1, (void*)top_down.data());
  cv::LUT(top_down_img, flatten_lut_, top_down_img); //remap classes

  //convert pointcloud header to ROS header
  std_msgs::Header img_header;
	pcl_conversions::fromPCL(cloud->header, img_header);

  auto stop = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  ROS_INFO_STREAM("Render took " << dur.count() << " ms");

  //publishHeatMap(top_down, 1, 2, cv::Rect(70, 20, 100, 100), img_header);
  updateFilter(top_down, img_header);
  publishTopDown(top_down_img, img_header);

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
