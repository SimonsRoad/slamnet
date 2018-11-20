
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

class MappingModel(object):
    """mapping model"""

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

        self.modelnets = Models(img_cur.get_shape().as_list(), self.rnn_batch_size, params.sequence_size, reuse_variables)

        self.modelnets.build_depth_architecture()
        self.modelnets.build_pose_architecture()
#        self.modelnets.build_localization_architecture()
        self.modelnets.build_mapping_architecture()

        [self.depthmap, self.depthmap_unc_i, self.depthmap_unc_d, self.tran, self.rot, self.pose_unc, self.tran_est, self.rot_est, self.pose_unc_est] = self.build_model(self.img_cur, self.img_next)

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

            # depth
            disp1 = self.modelnets.depth_model(img1)
            [disp1[i].set_shape(disp1[i]._keras_shape) for i in range(12)]
            disp2 = self.modelnets.depth_model(img2)
            [disp2[i].set_shape(disp2[i]._keras_shape) for i in range(12)]

            self.depth_prev = disp1[0]
            self.unc_i_prev = disp1[4]
            self.unc_d_prev = disp1[8] 

            # pose
            [trans, rot, unc, _, _] = self.modelnets.pose_model([img1,img2])

            # mapping
            imgs_for_mapping = concatenate([img1, img2, disp1[0], disp2[0], disp1[4], disp2[4], disp1[8], disp2[8]],axis=3)
            poses = concatenate([trans,rot,unc],axis=1)
            [disp_est, unc_i, unc_d, tran_est, rot_est, unc_est] = self.modelnets.mapping_model([imgs_for_mapping, poses])
            
            disp_est.set_shape(disp_est._keras_shape)
            unc_i.set_shape(unc_i._keras_shape)
            unc_d.set_shape(unc_d._keras_shape)

            tran_est.set_shape(tran_est._keras_shape)
            rot_est.set_shape(rot_est._keras_shape)
            unc_est.set_shape(unc_est._keras_shape)

        return disp_est, unc_i, unc_d, trans, rot, unc, tran_est, rot_est, unc_est

    def build_outputs(self):

        # generate depthmap1 and depthmap2 from depthmap
        self.depthmap1 = concatenate([self.depth_prev[:1,:,:,:],self.depthmap[:-1,:,:,:]],axis=0)
        self.depthmap2 = self.depthmap        
        self.depthmap1_unc_i = concatenate([self.unc_i_prev[:1,:,:,:],self.depthmap_unc_i[:-1,:,:,:]],axis=0)
        self.depthmap1_unc_d = concatenate([self.unc_d_prev[:1,:,:,:],self.depthmap_unc_d[:-1,:,:,:]],axis=0)
        self.depthmap2_unc_i = self.depthmap_unc_i
        self.depthmap2_unc_d = self.depthmap_unc_d

        # generate base images & depthmap
        self.img_base = tf.tile(self.img_cur[:1,:,:,:], [self.params.batch_size,1,1,1])
        self.depthmap_base = tf.tile(self.depthmap1[:1,:,:,:], [self.params.batch_size,1,1,1])
        self.depthmap_base_unc_i = tf.tile(self.unc_i_prev[:1,:,:,:], [self.params.batch_size,1,1,1])
        self.depthmap_base_unc_d = tf.tile(self.unc_d_prev[:1,:,:,:], [self.params.batch_size,1,1,1])

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
        self.depth_smoothness  = self.get_disparity_smoothness(self.depthmap1,  self.img_cur)


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
        self.image_dists = self.compute_temporal_loss(self.img_est, self.img_next, self.depthmap1_unc_i, self.pose_unc)
        self.image_n_dists = self.compute_temporal_loss(self.img_n_est, self.img_next, self.depthmap_base_unc_i, self.pose_unc_est)
        self.image_loss  = tf.reduce_mean(self.image_dists + self.image_n_dists)

        self.depth_dists = self.compute_temporal_loss(self.depth_est, self.depthmap2, self.depthmap1_unc_d + self.depthmap2_unc_d, self.pose_unc)
        self.depth_n_dists = self.compute_temporal_loss(self.depth_n_est, self.depthmap2, self.depthmap_base_unc_d + self.depthmap2_unc_d, self.pose_unc_est)
        self.depth_loss  = tf.reduce_mean(self.depth_dists + self.depth_n_dists)

        self.depth_smoothness_loss = tf.reduce_mean(self.depth_smoothness)

        self.total_loss = tf.reduce_mean((self.image_dists + self.image_n_dists) + 0.01*(self.depth_dists + self.depth_n_dists) + 0.01*self.depth_smoothness) 
        self.poses_txt = concatenate([self.tran,self.rot],axis=0)

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            tf.summary.scalar('image_loss', self.image_loss, collections=self.model_collection)
            tf.summary.scalar('depth_loss', self.depth_loss, collections=self.model_collection)
            tf.summary.scalar('depth_smoothness_loss', self.depth_smoothness_loss, collections=self.model_collection)
            tf.summary.image('img_cur',  self.img_cur,   max_outputs=3, collections=self.model_collection)
            tf.summary.image('img_next',  self.img_next,   max_outputs=3, collections=self.model_collection)
            tf.summary.image('depth_prev',  self.depth_prev,   max_outputs=3, collections=self.model_collection)
            tf.summary.image('depth',  self.depthmap1,   max_outputs=3, collections=self.model_collection)
            tf.summary.image('img_est',  self.img_est[0],   max_outputs=3, collections=self.model_collection)
            tf.summary.image('img_n_est',  self.img_n_est[0],   max_outputs=3, collections=self.model_collection)

            txtPredictions = tf.Print(tf.as_string(self.Q),[tf.as_string(self.Q)], message='predictions', name='txtPredictions')
            tf.summary.text('predictions', txtPredictions, collections=self.model_collection)

