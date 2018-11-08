
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

class LocalizationModel(object):
    """localization model"""

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
        self.modelnets.build_mapping_architecture()
        self.modelnets.build_localization_architecture()

        [self.depthmap1, self.depthmap2, self.depthmap1_unc, self.depthmap2_unc, self.tran, self.rot, self.unc, self.tran_est, self.rot_est, self.unc_est] = self.build_model(self.img_cur, self.img_next)

        if self.mode == 'test':
            return
        
        self.focal_length1 = cam_params[:,0]*params.width/cam_params[:,5]
        self.focal_length2 = cam_params[:,0]*params.height/cam_params[:,4]
        self.c0 = cam_params[:,1]*params.width/cam_params[:,5]
        self.c1 = cam_params[:,2]*params.height/cam_params[:,4]

        self.build_outputs()
        self.build_losses()
        self.build_summaries()     



    def build_model(self,img1,img2):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):

            #depth
            disp1 = self.modelnets.depth_model(img1)
            [disp1[i].set_shape(disp1[i]._keras_shape) for i in range(8)]
            disp2 = self.modelnets.depth_model(img2)
            [disp2[i].set_shape(disp2[i]._keras_shape) for i in range(8)] 

            #pose
            [trans, rot, unc] = self.modelnets.pose_model([img1,img2])
            out_trans, out_rot, out_unc = trans, rot, unc

            #mapping
            [disp1_est,disp1_unc_est] = self.modelnets.mapping_model([disp1[0],disp1[4]])
            [disp2_est,disp2_unc_est] = self.modelnets.mapping_model([disp2[0],disp2[4]])

            #localization
            trans = tf.reshape(trans,[self.rnn_batch_size,self.params.sequence_size,3])
            rot = tf.reshape(rot,[self.rnn_batch_size,self.params.sequence_size,3])
            unc = tf.reshape(unc,[self.rnn_batch_size,self.params.sequence_size,21])

            [trans_est, rot_est, unc_est] = self.modelnets.localization_model([trans,rot,unc])

            trans_est.set_shape(trans_est._keras_shape)
            rot_est.set_shape(rot_est._keras_shape)
            unc_est.set_shape(unc_est._keras_shape)

            trans_est = tf.reshape(trans_est, [-1, 3])
            rot_est = tf.reshape(rot_est, [-1, 3])
            unc_est = tf.reshape(unc_est, [-1, 21])

        return disp1_est, disp1_unc_est, disp2_est, disp2_unc_est, out_trans, out_rot, out_unc, trans_est, rot_est, unc_est

    def build_outputs(self):
        # create base images & depthmap
        self.img_base = tf.tile(self.img_cur[:1,:,:,:], [self.params.batch_size,1,1,1])
        self.depthmap_base = tf.tile(self.depthmap1[:1,:,:,:], [self.params.batch_size,1,1,1])

        # create rot_part & tran_part
        M = compose_matrix(self.rot_est,self.tran_est)
        for i in range(self.params.batch_size):
            if i==0:
                est = M[0:1,:,:]
                M_est = est
            else:
                est = tf.matmul(tf.matrix_inverse(M[i-1:i,:,:]),M[i:i+1,:,:])
                M_est = concatenate([M_est,est],axis=0)
        self.rot_part,self.tran_part = decompose_matrix(M_est)

#        # create acc_rot & acc_tran
#        M_delta = compose_matrix(self.rot_est,self.tran_est)
#        for i in range(self.params.batch_size):
#            if i==0:
#                est = M_delta[0:1,:,:]
#                M_est = est
#            else:
#                est = tf.matmul(est,M_delta[i:i+1,:,:])
#                M_est = concatenate([M_est,est],axis=0)
#        self.rot_acc,self.tran_acc = decompose_matrix(M_est)

        # generate k+1 th image
        self.plus0 = projective_transformer(self.img_cur, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2, self.rot, self.tran)
        self.plus1 = projective_transformer(self.img_base, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2, self.rot_est, self.tran_est)
        self.plus2 = projective_transformer(self.img_cur, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2, self.rot_part, self.tran_part)

        # generate k+1 th depth image
        self.depthplus0 = projective_transformer(self.depthmap1, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2, self.rot, self.tran)
        self.depthplus1 = projective_transformer(self.depthmap_base, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2, self.rot_est, self.tran_est)
        self.depthplus2 = projective_transformer(self.depthmap1, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2, self.rot_part, self.tran_part)



    def compute_temporal_loss(self, img_syn, img, uncertainty):

        # make uncertainty matrix (Q)
        L = tf.contrib.distributions.fill_triangular(uncertainty)
        Lt = tf.transpose(L,perm=[0, 2, 1])
        self.Q = tf.matmul(L,Lt)

        # make residual uncertainty
        img_diffs = [tf.reshape(tf.abs(img_syn[i] - img),[self.params.batch_size,-1]) for i in range(7)]
        res_diffs = [tf.reduce_mean(tf.abs(img_diffs[0]-img_diffs[i]),1) for i in range(7)]

        diffs_part = tf.stack([10.0*res_diffs[1],10.0*res_diffs[2],10.0*res_diffs[3],10.0*res_diffs[4],10.0*res_diffs[5],10.0*res_diffs[6]], axis=1)
        diffs_part = tf.expand_dims(diffs_part,2)
        diffs_part_t = tf.transpose(diffs_part,perm=[0, 2, 1])
        res_uncertainty = tf.matmul(tf.matmul(diffs_part_t,self.Q),diffs_part)

        res_u_norm = tf.norm(res_uncertainty,axis=[1,2])
#        res_u_norm = Lambda(lambda x: 0.000001 + x)(res_u_norm)
        res_u_plus = Lambda(lambda x: 1.0 + x)(res_u_norm)

        # dist
        diffs = tf.reduce_mean(tf.reshape(tf.square(img_syn[0] - img),[self.params.batch_size,-1]),1)
        dist = tf.divide(diffs, res_u_norm) + tf.log(res_u_plus)

        return tf.reduce_mean(dist)

    def build_losses(self):
        
        # PHOTOMETRIC REGISTRATION (temporal loss)
        self.l1_plus0 = self.compute_temporal_loss(self.plus0, self.img_next, self.unc) 
        self.l1_plus2 = self.compute_temporal_loss(self.plus2, self.img_next, self.unc) 
        self.l1_plus1 = self.compute_temporal_loss(self.plus1, self.img_next, self.unc_est)

        self.l1_depthplus0 = self.compute_temporal_loss(self.depthplus0, self.depthmap2, self.unc) 
        self.l1_depthplus2 = self.compute_temporal_loss(self.depthplus2, self.depthmap2, self.unc) 
        self.l1_depthplus1 = self.compute_temporal_loss(self.depthplus1, self.depthmap2, self.unc_est)

        self.image_loss_temporal = self.l1_plus0 + self.l1_plus1 + self.l1_plus2
        self.depth_loss_temporal = self.l1_depthplus0 + self.l1_depthplus1 + self.l1_depthplus2
        self.total_loss = self.image_loss_temporal + 0.1*self.depth_loss_temporal 
        self.poses_txt = concatenate([self.tran,self.tran_est],axis=0)

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            tf.summary.scalar('image_loss_temporal', self.image_loss_temporal, collections=self.model_collection)
            tf.summary.scalar('depth_loss_temporal', self.depth_loss_temporal, collections=self.model_collection)
            tf.summary.image('img_next',  self.img_next,   max_outputs=3, collections=self.model_collection)
            tf.summary.image('depth',  self.depthmap1,   max_outputs=3, collections=self.model_collection)
            tf.summary.image('depthplus0',  self.depthplus0[0],   max_outputs=3, collections=self.model_collection)
            tf.summary.image('plus1',  self.plus1[0],   max_outputs=3, collections=self.model_collection)

            txtPredictions = tf.Print(tf.as_string(self.Q),[tf.as_string(self.Q)], message='predictions', name='txtPredictions')
            tf.summary.text('predictions', txtPredictions, collections=self.model_collection)

