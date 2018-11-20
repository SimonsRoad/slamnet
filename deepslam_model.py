
from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.layers import Lambda

from models import *
from transformers import *
from projective_transformer import *
from projective_transformer_inv import *

deepslam_parameters = namedtuple('parameters',
                        'height, width, '
                        'batch_size, '
                        'sequence_size,'
                        'num_threads, '
                        'num_epochs, '
                        'full_summary')

class DeepslamModel(object):
    """deepslam model"""

    def __init__(self, params, mode, img_cur, img_next, poses_cur, poses_next, cam_params, reuse_variables=None, model_index=0):

        self.params = params
        self.mode = mode
        self.img_cur = img_cur
        self.img_next = img_next
        self.poses_cur = poses_cur
        self.poses_next = poses_next
        self.model_collection = ['model_' + str(model_index)]
        self.reuse_variables = reuse_variables
        self.rnn_batch_size = int(params.batch_size/params.sequence_size)

        self.modelnets = Models(img_cur.get_shape().as_list(), reuse_variables)

        self.modelnets.build_depth_encoder()
        self.modelnets.build_depth_decoder()
        self.modelnets.build_depth_variances_decoder()
        self.modelnets.build_depth_variances_decoder2()
        self.modelnets.build_pose_encoder()
        self.modelnets.build_pose_decoder()
        self.modelnets.build_slam_architecture()
        self.modelnets.build_pose_variances_decoder()

        self.depthmap1, self.depthmap2, self.tran, self.rot, self.unc, self.img_var1, self.depth_var1, self.img_var2, self.depth_var2, self.img_var3, self.depth_var3, self.img_var4, self.depth_var4, self.pose_unc = self.build_model(self.img_cur, self.img_next)

        if self.mode == 'test':
            return
        
        self.focal_length1 = cam_params[:,0]*params.width/cam_params[:,5]
        self.focal_length2 = cam_params[:,0]*params.height/cam_params[:,4]
        self.c0 = cam_params[:,1]*params.width/cam_params[:,5]
        self.c1 = cam_params[:,2]*params.height/cam_params[:,4]

        self.build_outputs()
        self.build_losses()
        self.build_summaries()     

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        gx = tf.pad(gx, [[0, 0], [0, 0], [0, 1], [0, 0]], "CONSTANT")
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        gy = tf.pad(gy, [[0, 0], [0, 1], [0, 0], [0, 0]], "CONSTANT")
        return gy

    def get_disparity_smoothness(self, disp, img):

        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

        smoothness_x = tf.abs(disp_gradients_x) * weights_x 
        smoothness_y = tf.abs(disp_gradients_y) * weights_y

        return smoothness_x + smoothness_y

    def build_model(self,img1,img2):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):

            # encoders
            [d1_conv1,d1_conv2,d1_conv3,d1_conv4,d1_conv5,d1_conv6,d1_conv7] = self.modelnets.depth_encoder_model(img1)
            [d2_conv1,d2_conv2,d2_conv3,d2_conv4,d2_conv5,d2_conv6,d2_conv7] = self.modelnets.depth_encoder_model(img2)            
            [p1_conv1,p1_conv2,p1_conv3,p1_conv4,p1_conv5,p1_conv6,p1_conv7] = self.modelnets.pose_encoder_model([img1,img2])

            # SLAM Nets
            [depth1_conv7, depth2_conv7, pose_conv7] = self.modelnets.slam_model(d1_conv7, d2_conv7, p1_conv7)
            skips_depth1 = [d1_conv1,d1_conv2,d1_conv3,d1_conv4,d1_conv5,d1_conv6,depth1_conv7]
            skips_depth2 = [d2_conv1,d2_conv2,d2_conv3,d2_conv4,d2_conv5,d2_conv6,depth2_conv7]
            skips_pose = [p1_conv1,p1_conv2,p1_conv3,p1_conv4,p1_conv5,p1_conv6,pose_conv7]

            # decoders
            [disp1,_,_,_,_,_,_,_,_,_,_,_] = self.modelnets.depth_decoder_model(skips_depth1) 
            disp1.set_shape(disp1._keras_shape)
            [disp2,_,_,_,_,_,_,_,_,_,_,_] = self.modelnets.depth_decoder_model(skips_depth2) 
            disp2.set_shape(disp2._keras_shape)

            [img_var1, depth_var1] = self.modelnets.depth_var_decoder_model(skips_depth1) 
            img_var1.set_shape(img_var1._keras_shape)
            depth_var1.set_shape(depth_var1._keras_shape)
            [img_var2, depth_var2] = self.modelnets.depth_var_decoder_model(skips_depth2) 
            img_var2.set_shape(img_var2._keras_shape)
            depth_var2.set_shape(depth_var2._keras_shape)

            [img_var3, depth_var3] = self.modelnets.depth_var_decoder_model2(skips_depth1) 
            img_var3.set_shape(img_var3._keras_shape)
            depth_var3.set_shape(depth_var3._keras_shape)
            [img_var4, depth_var4] = self.modelnets.depth_var_decoder_model2(skips_depth2) 
            img_var4.set_shape(img_var4._keras_shape)
            depth_var4.set_shape(depth_var4._keras_shape)

            [trans, rot, unc] = self.modelnets.pose_decoder_model(skips_pose)

            pose_unc = self.modelnets.pose_var_decoder_model(skips_pose)

        return disp1, disp2, trans, rot, unc, img_var1, depth_var1, img_var2, depth_var2, img_var3, depth_var3, img_var4, depth_var4, pose_unc

    def build_outputs(self):

        # generate base images & depthmap
        self.img_base = tf.tile(self.img_cur[:1,:,:,:], [self.params.batch_size,1,1,1])
        self.depthmap_base = tf.tile(self.depthmap1[:1,:,:,:], [self.params.batch_size,1,1,1])

        # generate acc_rot & acc_tran
        M_delta = compose_matrix(self.rot_est,self.tran_est)
        for i in range(self.params.batch_size):
            if i==0:
                est = M_delta[0:1,:,:]
                M_est = est
            else:
                est = tf.matmul(est,M_delta[i:i+1,:,:])
                M_est = concatenate([M_est,est],axis=0)
        self.rot_acc,self.tran_acc = decompose_matrix(M_est)

        # generate k+1 th image
        self.img_est = projective_transformer(self.img_cur, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2, self.rot_est, self.tran_est)

        # generate k+n th image
        self.img_n_est = projective_transformer(self.img_base, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2, self.rot_acc, self.tran_acc)

        # generate k+1 th depth image
        self.depth_est = projective_transformer(self.depthmap1, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2, self.rot_est, self.tran_est)

        # generate k+n th depth image
        self.depth_n_est = projective_transformer(self.depthmap_base, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2, self.rot_acc, self.tran_acc)

        # DISPARITY SMOOTHNESS
        self.depth1_smoothness  = self.get_disparity_smoothness(self.depthmap1,  self.img_cur)
        self.depth2_smoothness  = self.get_disparity_smoothness(self.depthmap2,  self.img_next)


    def compute_temporal_loss(self, img_syn, img, data_uncertainty, pose_uncertainty):

        _num_batch    = img.shape[0]
        _height       = img.shape[1]
        _width        = img.shape[2]
        _num_channels = img.shape[3]

        # make uncertainty matrix (Q)
        L = tf.contrib.distributions.fill_triangular(pose_uncertainty)
        Lt = tf.transpose(L,perm=[0, 2, 1])
        self.Q = tf.matmul(L,Lt)

        # make residual uncertainty
        img_diffs = [tf.reshape(tf.abs(img_syn[i] - img),[_num_batch,-1]) for i in range(7)]
        res_diffs = [tf.abs(img_diffs[0]-img_diffs[i]) for i in range(7)]

        diffs_part = tf.stack([10.0*res_diffs[1],10.0*res_diffs[2],10.0*res_diffs[3],10.0*res_diffs[4],10.0*res_diffs[5],10.0*res_diffs[6]], axis=2)
        tmp = tf.reduce_sum(tf.matmul(diffs_part,self.Q)*diffs_part,2)
        res_uncertainty = tf.reshape(tmp, [_num_batch,_height,_width,_num_channels])

        res_u_norm = res_uncertainty + data_uncertainty
        res_u_norm = Lambda(lambda x: 0.00001 + x)(res_u_norm)
        res_u_plus = Lambda(lambda x: 1.0 + x)(res_u_norm)

        # dist
        diffs = tf.square(img_syn[0] - img)
        dist = tf.divide(diffs, res_u_norm) + tf.log(res_u_plus)

        loss = tf.reduce_sum(dist,3, keepdims=True)

        return loss

    def build_losses(self):
        
        # PHOTOMETRIC REGISTRATION (temporal loss)
        self.image_dists = self.compute_temporal_loss(self.img_est, self.img_next, self.img_var1+self.img_var2, self.unc)
        self.image_n_dists = self.compute_temporal_loss(self.img_n_est, self.img_next, self.img_var3+self.img_var4, self.pose_unc)
        self.image_loss  = tf.reduce_mean(self.image_dists + self.image_n_dists)

        self.depth_dists = self.compute_temporal_loss(self.depth_est, self.depthmap2, self.depth_var1+self.depth_var2, self.unc)
        self.depth_n_dists = self.compute_temporal_loss(self.depth_n_est, self.depthmap2, self.depth_var3+self.depth_var4, self.pose_unc)
        self.depth_loss  = tf.reduce_mean(self.depth_dists + self.depth_n_dists)

        self.depth_smoothness_loss = tf.reduce_mean(self.depth1_smoothness + self.depth2_smoothness)

        self.total_loss = tf.reduce_mean((self.image_dists + self.image_n_dists) + 0.01*(self.depth_dists + self.depth_n_dists) + 0.01*(self.depth1_smoothness + self.depth2_smoothness))
 
        self.poses_txt = concatenate([self.tran,self.rot],axis=0)

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            tf.summary.scalar('image_loss', self.image_loss, collections=self.model_collection)
            tf.summary.scalar('depth_loss', self.depth_loss, collections=self.model_collection)
            tf.summary.scalar('depth_smoothness_loss', self.depth_smoothness_loss, collections=self.model_collection)
            tf.summary.image('img_cur',  self.img_cur,   max_outputs=3, collections=self.model_collection)
            tf.summary.image('img_next',  self.img_next,   max_outputs=3, collections=self.model_collection)
            tf.summary.image('depth',  self.depthmap1,   max_outputs=3, collections=self.model_collection)
            tf.summary.image('img_est',  self.img_est[0],   max_outputs=3, collections=self.model_collection)
            tf.summary.image('img_n_est',  self.img_n_est[0],   max_outputs=3, collections=self.model_collection)

            txtPredictions = tf.Print(tf.as_string(self.Q),[tf.as_string(self.Q)], message='predictions', name='txtPredictions')
            tf.summary.text('predictions', txtPredictions, collections=self.model_collection)

