from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt

import roslib
import sys
import rospy
import tf as ros_tf
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from deepslam_model import *
from deepslam_dataloader import *
#from average_gradients import *

parser = argparse.ArgumentParser(description='deepslam TensorFlow implementation.')

parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

class deepslam:
    def __init__(self):

        '''Initialize ros publisher, ros subscriber'''
        self.image_pub = rospy.Publisher("/undeepvo/image",Image,queue_size=10)
        self.bridge = CvBridge()
        self.image_sub_left = rospy.Subscriber("/kitti/left_color_image",Image,self.callback_left)    

        '''Initialize refresh parameters '''
        self.is_left_in  = False
        self.is_start = False
        self.test_num = 0

        '''Initialize network for the VO estimation'''
        params = deepslam_parameters(
            height=args.input_height,
            width=args.input_width,
            batch_size=10,
            sequence_size=10,
            num_threads=1,
            num_epochs=1,
            full_summary=False)

        left  = tf.placeholder(tf.float32, [10, args.input_height, args.input_width, 3])
        left_next  = tf.placeholder(tf.float32, [10, args.input_height, args.input_width, 3])
        self.model = DeepslamModel(params, "test", left, left_next, None, None)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coordinator)

        # RESTORE
        restore_path = args.checkpoint_path.split(".")[0]
        train_saver.restore(self.sess, args.checkpoint_path)

        br = ros_tf.TransformBroadcaster()
        br.sendTransform((0,0,0),
                         ros_tf.transformations.quaternion_from_euler(0,0,0),
                         rospy.Time.now(),
                         "/undeepvo/Current",
                         "/undeepvo/World")

    def callback_left(self,data):
        input_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#        cv2.imshow("Image window", input_image)
#        cv2.waitKey(3)

        original_height, original_width, num_channels = input_image.shape

        input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
        if self.is_start == True:
            self.img_left = self.img_left_next
        self.img_left_next = np.expand_dims(input_image.astype(np.float32) / 255, axis=0)
        self.is_left_in = True


    def refresh(self):
        "Check new sequence is comming"
        if self.is_start == True:
            if self.is_left_in == True:
                self.test_simple()
                self.is_left_in = False
        else: 
            if self.is_left_in == True:
                self.is_start = True
                self.is_left_in = False


    def test_simple(self):

        if self.test_num == 0:
            self.batch_left = self.img_left
            self.batch_left_next = self.img_left_next
        else:
            self.batch_left = self.batch_left[:self.test_num+1,:,:,:]
            self.batch_left_next = self.batch_left_next[:self.test_num+1,:,:,:]

        if self.test_num<10:
            num = 9 - self.test_num
            left_tile = np.tile(self.img_left, (num, 1,1,1))
            left_next_tile = np.tile(self.img_left_next, (num, 1,1,1))
            self.batch_left = np.concatenate([self.batch_left,left_tile],axis=0)
            self.batch_left_next = np.concatenate([self.batch_left_next,left_next_tile],axis=0)
            self.test_num = self.test_num +1
        else:
            self.batch_left = np.concatenate([self.batch_left[1:,:,:,:],self.img_left],axis=0)
            self.batch_left_next = np.concatenate([self.batch_left_next[1:,:,:,:],self.img_left_next],axis=0)

        """Test function."""
        [tran, rot] = self.sess.run([self.model.tran_est, self.model.rot_est], feed_dict={self.model.img_cur: self.batch_left, self.model.img_next: self.batch_left_next})


        #Publish R and t
        print("publish R and t")
        tran = tran[self.test_num-1,:].squeeze()
        rot  = rot[self.test_num-1,:].squeeze()
        print(tran)
        print(rot)
        br = ros_tf.TransformBroadcaster()
        br.sendTransform(tran,
                         ros_tf.transformations.quaternion_from_euler(rot[0],rot[1],rot[2]),
                         rospy.Time.now(),
                         "/undeepvo/Current",
                         "/undeepvo/World")


def main(_):


    #init rospy
    rospy.init_node('deepslam', anonymous=True)
    rate = rospy.Rate(100) # 100hz

    ic = deepslam()
    while not rospy.is_shutdown():
        ic.refresh()
        rate.sleep()

if __name__ == '__main__':
    tf.app.run()
