#include <ros/ros.h>
#include "top_down_render/top_down_render.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "top_down_render");
  ros::NodeHandle nh("~");

  try {
    TopDownRender node(nh);
    node.initialize();
    ros::spin();
  } catch (const std::exception& e) {
    ROS_ERROR("%s: %s", nh.getNamespace().c_str(), e.what());
  }
  return 0;
}
