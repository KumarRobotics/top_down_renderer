#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import rospy
import subprocess
import tf2_ros
import tf2_geometry_msgs
import pyrosbag as prb
import thread
import time
import os
import signal
import pickle
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float32
from math import radians, cos, sin, asin, sqrt

def haversine(coord1, coord2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [coord1[1], coord1[0], coord2[1], coord2[0]])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    return c * 6371 * 1000

#janky way of shared memory without pointers
#so sue me
gps_hist = {'pos': [], 'times': []}
loc_hist = {'pos': [], 'times': []}
scale = -1
initial_time = None
last_pos = np.array([0,0,0])
total_dist = 0.
lock = thread.allocate_lock()

class BenchmarkLoc:
    def __init__(self):
        self.image_origin_gps_ = np.array([48.9803654, 8.3877372])
        #m per unit deg (approx.)
        self.gps_scale_ = np.array([0., 0.])
        self.gps_scale_[0] = haversine(self.image_origin_gps_, self.image_origin_gps_ + np.array([0.001, 0]))*1000
        self.gps_scale_[1] = haversine(self.image_origin_gps_, self.image_origin_gps_ + np.array([0, 0.001]))*1000
        print(self.gps_scale_)

        self.tf_buffer_ = tf2_ros.Buffer(rospy.Duration(1000.0))
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_)

        self.gps_sub_ = rospy.Subscriber('/kitti/oxts/gps/fix', NavSatFix, self.gps_cb)

    def gps_cb(self, gps_msg):
        global initial_time, last_pos, total_dist
        lat_lon = np.array([gps_msg.latitude, gps_msg.longitude])
        lat_lon -= self.image_origin_gps_
        pos_m = np.flip(lat_lon * self.gps_scale_)
        if initial_time is None:
            initial_time = gps_msg.header.stamp
            last_pos = pos_m
        else:
            if (gps_msg.header.stamp - initial_time).to_sec() < 70:
                total_dist += np.linalg.norm(pos_m - last_pos)
                last_pos = pos_m
                print(total_dist)


if __name__ == '__main__':
    rospy.init_node('benchmark_loc')
    bl = BenchmarkLoc()
    rospy.spin()
