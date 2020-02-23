#!/usr/bin/env python3

# Process bag with gps and images and output folder for use by ODM

import rosbag
from cv_bridge import CvBridge
import numpy as np
import cv2
import utm

class PreprocSurvey:
    def __init__(self, bag_path):
        self.bag_ = rosbag.Bag(bag_path, 'r')
        print("Loaded bag")

    def proc(self):
        gps_stamps = np.array([])
        positions = np.empty([0,3])
        init_altitude = None
        for topic, msg, t in self.bag_.read_messages():
            if topic == '/ublox/fix':
                if init_altitude is None:
                    init_altitude = msg.altitude
                
                rel_alt = msg.altitude - init_altitude
                if rel_alt < 10:
                    continue
                
                gps_stamps = np.concatenate([gps_stamps, np.array([t.to_sec()])])
                positions = np.vstack([positions, np.array([msg.longitude, msg.latitude, rel_alt])])
        print("Loaded GPS")
                
        gps_file = open('geo.txt', 'w')
        gps_file.write('EPSG:4326\n')
        bridge = CvBridge()

        last_pos_utm = np.array([0,0])
        for topic, msg, t in self.bag_.read_messages():
            if topic == '/ovc/rgb/image_color/compressed':
                closest_ind = np.argmin(np.abs(t.to_sec() - gps_stamps))
                if np.abs(t.to_sec() - gps_stamps[closest_ind]) < 0.2:
                    pos = positions[closest_ind]
                    pos_utm = np.array(utm.from_latlon(pos[1], pos[0])[:2])
                    if np.linalg.norm(pos_utm - last_pos_utm) > 5:
                        filename = f'images/{msg.header.stamp.to_nsec()}.jpg'
                        gps_file.write(f'{filename} {pos[0]} {pos[1]} {pos[2]}\n')
                        img = bridge.compressed_imgmsg_to_cv2(msg)
                        cv2.imwrite(filename, img)
                        last_pos_utm = pos_utm
                        print(filename)
        gps_file.close()



if __name__ == '__main__':
    ps = PreprocSurvey('/media/ian/ResearchSSD/grace_quarters/2021-11-01/gq_pad_coverage1_2021-10-30-21-00-48.bag')
    ps.proc()
