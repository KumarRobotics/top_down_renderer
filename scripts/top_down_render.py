#!/usr/bin/env python3

import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import time
import matplotlib.pyplot as plt
import pcl
import pcl.pcl_visualization

class TopDownRender():
    def __init__(self):
        self.pc_sub_ = rospy.Subscriber('pc', PointCloud2, self.pc_cb)
        self.fig_ = plt.figure()
        #self.img_ = plt.imshow(np.zeros([31, 31, 3]))
        self.img_ = plt.imshow(np.zeros([64, 1024, 3]))
        plt.show()

    def proj_top_down(self, pc, cell_size, cell_num):
        x_centers = np.tile(np.linspace(-cell_num*cell_size/2, cell_num*cell_size/2, cell_num+1), [cell_num+1, 1])
        y_centers = x_centers.copy().T
        x_centers = x_centers.flatten()
        y_centers = y_centers.flatten()
        
        dx = np.abs(pc['x'][:, None]-x_centers[None, :])
        dy = np.abs(pc['y'][:, None]-y_centers[None, :])
        dist = np.max([dx, dy], axis=0)
        
        top_down_img = np.zeros([cell_num+1, cell_num+1, 3], dtype=np.uint8)
        
        start = time.perf_counter()
        for cell in range(dx.shape[1]):
            cell_pc_ind = np.argwhere(dist[:, cell] < cell_size/2)
            cell_pc = pc[cell_pc_ind]
            if cell_pc.size > 0:
                # class is majority vote of top 50 points in cell
                section = np.max([-50, -cell_pc['z'].size])
                top_ind = np.argpartition(cell_pc['z'][:,0], section)[section:]
                top_ind_full = cell_pc_ind[top_ind]
                colors = np.stack([pc[top_ind_full]['r'][:,0], pc[top_ind_full]['g'][:,0], pc[top_ind_full]['b'][:,0]]).T
                colors, nums = np.unique(colors, axis=0, return_counts=True) 

                top_down_img[int(cell/(cell_num+1)), int(cell%(cell_num+1)), :] = colors[np.argmax(nums)]
        rospy.loginfo(time.perf_counter()-start)

        return top_down_img

    def organize(self, pc_np):
        pc_org = np.zeros([64, pc_np['x'].shape[0]//64, 3], dtype=np.float32)
        pc_org[:,:,0] = np.reshape(pc_np['r'], pc_org.shape[:-1], 'F')
        pc_org[:,:,1] = np.reshape(pc_np['g'], pc_org.shape[:-1], 'F')
        pc_org[:,:,2] = np.reshape(pc_np['b'], pc_org.shape[:-1], 'F')

        #pc_org[1::4,:,:] = np.roll(pc_org[1::4,:,:], -6, axis=1)
        #pc_org[2::4,:,:] = np.roll(pc_org[2::4,:,:], -12, axis=1)
        #pc_org[3::4,:,:] = np.roll(pc_org[3::4,:,:], -18, axis=1)

        #return np.reshape(pc_org, [pc_org.shape[0]*pc_org.shape[1], 3])
        return pc_org

    def pc_cb(self, pc_msg):
        pc_np = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
        pc_np = ros_numpy.point_cloud2.split_rgb_field(pc_np)

        #img = self.proj_top_down(pc_np, 1, 30)
        #self.img_.set_data(img.astype(np.float32)/255)

        pc_org = self.organize(pc_np)
        #cloud = pcl.PointCloud()
        #cloud.from_array_org(pc_org)

        #ne = cloud.make_IntegralImageNormalEstimation()
        #ne.set_NormalEstimation_Method_AVERAGE_3D_GRADIENT()
        #ne.set_MaxDepthChange_Factor(0.1)
        #ne.set_NormalSmoothingSize(10.0)
        #normals = ne.compute()

        #normals_np = normals.to_array()
        #print(normals_np.shape)

        #viewer = pcl.pcl_visualization.PCLVisualizering()
        #viewer.SetBackgroundColor(0,0,0.5)
        #viewer.AddPointCloud(cloud)
        #viewer.AddPointCloudNormals(cloud, normals, 1, 1, b'normals')

        #while True:
        #    if viewer.WasStopped():
        #        break
        #    viewer.SpinOnce()

        self.img_.set_data(pc_org.astype(np.float32)/255)
        plt.draw()

if __name__ == '__main__':
    rospy.init_node('top_down_render')
    top_down_render = TopDownRender()
    rospy.spin()
