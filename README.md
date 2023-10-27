# top_down_renderer

Code for the semantic crossview SLAM algorithm detailed in 
[Any Way You Look at It: Semantic Crossview Localization and Mapping With LiDAR](https://ieeexplore.ieee.org/document/9361130)

## Installation
- rangenet_inf
- grid_map
- ros-noetic-octomap-msgs (for phoenix)


## Getting started

### Subscriptions
* `/os_node/rofl_odom/pose [geometry_msgs/PoseStamped]`
* `/os_node/segmented_point_cloud [sensor_msgs/PointCloud2]`
* `/titan/asoom/map [grid_map_msgs/GridMap]`

### Publications
* `/top_down_render/pose_est [geometry_msgs/PoseWithCovarianceStamped]`
  
