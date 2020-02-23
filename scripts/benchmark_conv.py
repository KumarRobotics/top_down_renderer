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
convergence_start = None
lock = thread.allocate_lock()

class LaunchManager:
    def __init__(self, bag_path):
        self.bag_path_ = bag_path

    def start(self):
        global gps_hist, loc_hist, lock, convergence_start

        overall_data = []

        start_t = 0
        while True:
            print('STARTING RUN')
            loc = subprocess.Popen(["roslaunch /home/ian/catkin/surveying_ws/src/top_down_render/launch/top_down_render_kitti.launch"],
                                 shell=True,preexec_fn=os.setsid)
            time.sleep(5)
            bag = subprocess.Popen(["rosbag play --start " + str(start_t) + " " + self.bag_path_],
                                 shell=True,preexec_fn=os.setsid)

            while bag.poll() is None:
                #record some of traj after convergence
                if len(loc_hist['pos']) > 200:
                    os.killpg(os.getpgid(bag.pid), signal.SIGTERM)
                time.sleep(1)
            print('killing mapper')
            #kill localizer/mapper process
            os.killpg(os.getpgid(loc.pid), signal.SIGTERM)

            lock.acquire()
            print('DONE')
            run = {'gps': gps_hist, 'loc': loc_hist, 'start': start_t, 'scale': scale}
            overall_data.append(run)
            pickle.dump(overall_data, open('kitti_runs.pkl', 'wb'))

            gps_hist = {'pos': [], 'times': []}
            loc_hist = {'pos': [], 'times': []}
            convergence_start = None
            lock.release()

            start_t += 30
            if start_t > 850:
                break

class BenchmarkLoc:
    def __init__(self):
        global convergence_start
        self.image_origin_gps_ = np.array([48.9803654, 8.3877372])
        #m per unit deg (approx.)
        self.gps_scale_ = np.array([0., 0.])
        self.gps_scale_[0] = haversine(self.image_origin_gps_, self.image_origin_gps_ + np.array([0.001, 0]))*1000
        self.gps_scale_[1] = haversine(self.image_origin_gps_, self.image_origin_gps_ + np.array([0, 0.001]))*1000
        print(self.gps_scale_)
        convergence_start = None

        self.tf_buffer_ = tf2_ros.Buffer(rospy.Duration(1000.0))
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_)

        self.gps_sub_ = rospy.Subscriber('/kitti/oxts/gps/fix', NavSatFix, self.gps_cb)
        self.loc_sub_ = rospy.Subscriber('/subt/global_pose', PoseStamped, self.loc_cb)
        self.sem_loc_sub_ = rospy.Subscriber('/subt/sem_pose', PoseWithCovarianceStamped, self.sem_loc_cb)
        self.scale_sub_ = rospy.Subscriber('/top_down_render/scale', Float32, self.scale_cb)

    def gps_cb(self, gps_msg):
        global convergence_start
        lat_lon = np.array([gps_msg.latitude, gps_msg.longitude])
        lat_lon -= self.image_origin_gps_
        pos_m = np.flip(lat_lon * self.gps_scale_)
        if convergence_start is not None:
            if gps_msg.header.stamp > convergence_start:
                lock.acquire()
                gps_hist['pos'].append(pos_m)
                gps_hist['times'].append(gps_msg.header.stamp.to_nsec())
                lock.release()

    def loc_cb(self, loc_msg):
        global convergence_start
        trans = self.tf_buffer_.lookup_transform('world', loc_msg.header.frame_id, loc_msg.header.stamp)
        loc_trans = tf2_geometry_msgs.do_transform_pose(loc_msg, trans)
        pos = np.array([loc_trans.pose.position.x, loc_trans.pose.position.y])
        #if convergence_start is not None:
        #    if loc_msg.header.stamp > convergence_start:
        #        lock.acquire()
        #        loc_hist['pos'].append(pos)
        #        loc_hist['times'].append(loc_msg.header.stamp.to_nsec())
        #        lock.release()

    def sem_loc_cb(self, sem_loc_msg):
        global convergence_start
        pos = np.array([sem_loc_msg.pose.pose.position.x, sem_loc_msg.pose.pose.position.y])
        if convergence_start is None:
            print('convergence detected')
            convergence_start = sem_loc_msg.header.stamp
        if convergence_start is not None:
            lock.acquire()
            loc_hist['pos'].append(pos)
            loc_hist['times'].append(sem_loc_msg.header.stamp.to_nsec())
            lock.release()

    def scale_cb(self, scale_msg):
        global scale
        lock.acquire()
        scale = scale_msg.data
        lock.release()

if __name__ == '__main__':
    #lm = LaunchManager('/media/ian/SSD1/tmp_datasets/KITTI/kitti_2011_10_03_drive_0027_synced.bag')
    #lm = LaunchManager('/media/ian/SSD1/tmp_datasets/ucity/*.bag')
    #lm = LaunchManager('/media/ian/SSD1/tmp_datasets/ucity2/*.bag')
    #lm = LaunchManager('/media/ian/HDD1/DrivingDatasets/4_9_20_Morgantown_3/morgtown_2*.bag')
    lm = LaunchManager('/media/ian/SSD1/tmp_datasets/KITTI/kitti0.bag')
    thread.start_new_thread(lm.start, ())

    rospy.init_node('benchmark_loc')
    bl = BenchmarkLoc()
    rospy.spin()
