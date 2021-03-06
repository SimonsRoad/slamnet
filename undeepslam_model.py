
from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.layers import Lambda
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Cropping2D, Dense, Flatten, Input, Reshape, LSTM, GRU, TimeDistributed
from keras.models import Model

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


        self.build_depth_architecture()
        self.build_pose_architecture()
        self.build_localization_architecture()

        [self.depthmap1, self.depthmap2, self.tran, self.rot, self.unc, self.tran_est, self.rot_est, self.unc_est] = self.build_model(self.img_cur, self.img_next)

        if self.mode == 'test':
            return
        
        self.focal_length1 = cam_params[:,0]*params.width/cam_params[:,5]
        self.focal_length2 = cam_params[:,0]*params.height/cam_params[:,4]
        self.c0 = cam_params[:,1]*params.width/cam_params[:,5]
        self.c1 = cam_params[:,2]*params.height/cam_params[:,4]

        

        self.build_outputs()
        self.build_losses()
        self.build_summaries()     

    def get_variance(self, x):
        variance = self.conv(x, 3, 3, 1, activation='softplus')
        variance = Lambda(lambda x: 0.01 + x)(variance)
        return variance

    def get_depth(self, x):
        depth = self.conv(x, 1, 3, 1, activation='sigmoid')
        depth = Lambda(lambda x: 80.0 * x)(depth)
        return depth

    @staticmethod
    def conv(input, channels, kernel_size, strides, activation='elu'):

        return Conv2D(channels, kernel_size=kernel_size, strides=strides, padding='same', activation=activation)(input)

    @staticmethod
    def deconv(input, channels, kernel_size, scale):

        output =  Conv2DTranspose(channels, kernel_size=kernel_size, strides=scale, padding='same')(input)
        output_shape = output._keras_shape
        output.set_shape(output_shape)
        return output
    @staticmethod
    def maxpool(input, kernel_size):
        
        return MaxPooling2D(pool_size=kernel_size, strides=2, padding='same', data_format=None)(input)

    def conv_block(self, input, channels, kernel_size):
        conv1 = self.conv(input, channels, kernel_size, 1)

        conv2 = self.conv(conv1, channels, kernel_size, 2)

        return conv2

    def deconv_block(self, input, channels, kernel_size, skip):

        deconv1 = self.deconv(input, channels, kernel_size, 2)
        if skip is not None:
            s = skip.shape

            if  s[1] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:-1,:,:])(deconv1)
            if  s[2] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:,:-1,:])(deconv1)

            concat1 = concatenate([deconv1, skip], 3)
        else:
            concat1 = deconv1

        iconv1 = self.conv(concat1, channels, kernel_size, 1)
        return iconv1

    def build_depth_architecture(self):

        with tf.variable_scope('depth_model',reuse=self.reuse_variables):

            input = Input(batch_shape=self.img_cur.get_shape().as_list())

            # encoder
            conv1 = self.conv_block(input, 32, 7)

            conv2 = self.conv_block(conv1, 64, 5)

            conv3 = self.conv_block(conv2, 128, 3)


            conv4 = self.conv_block(conv3, 256, 3)

            conv5 = self.conv_block(conv4, 512, 3)

            conv6 = self.conv_block(conv5, 512, 3)

            conv7 = self.conv_block(conv6, 512, 3)

            skip1 = conv1

            skip2 = conv2
        
            skip3 = conv3

            skip4 = conv4

            skip5 = conv5

            skip6 = conv6


            # decoder1
            deconv7 = self.deconv_block(conv7, 512, 3, skip6)

            deconv6 = self.deconv_block(deconv7, 512, 3, skip5)

            deconv5 = self.deconv_block(deconv6, 256, 3, skip4)
            
            deconv4 = self.deconv_block(deconv5, 128, 3, skip3)
            disp4 = self.get_depth(deconv4)

            deconv3 = self.deconv_block(deconv4, 64, 3, skip2)
            disp3 = self.get_depth(deconv3)

            deconv2 = self.deconv_block(deconv3, 32, 3, skip1)
            disp2 = self.get_depth(deconv2)

            deconv1 = self.deconv_block(deconv2, 16, 3, None)

            s = self.img_cur.shape
            if  s[1] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:-1,:,:])(deconv1)
            if  s[2] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:,:-1,:])(deconv1)

            disp1 = self.get_depth(deconv1)

            # decoder2

            deconv4_2 = self.deconv_block(deconv5, 128, 3, skip3)
            disp4_2 = self.get_variance(deconv4_2)

            deconv3_2 = self.deconv_block(deconv4_2, 64, 3, skip2)
            disp3_2 = self.get_variance(deconv3_2)

            deconv2_2 = self.deconv_block(deconv3_2, 32, 3, skip1)
            disp2_2 = self.get_variance(deconv2_2)

            deconv1_2 = self.deconv_block(deconv2_2, 16, 3, None)

            s = self.img_cur.shape
            if  s[1] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_2)
            if  s[2] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_2)

            disp1_2 = self.get_variance(deconv1_2)

            disp_est  = [disp1, disp2, disp3, disp4, disp1_2, disp2_2, disp3_2, disp4_2]

            self.depth_model = Model(input, disp_est)

    def build_pose_architecture(self):
        
        with tf.variable_scope('pose_model',reuse=self.reuse_variables):
            input1 = Input(batch_shape=self.img_cur.get_shape().as_list())

            input2 = Input(batch_shape=self.img_cur.get_shape().as_list())

            input = concatenate([input1,input2], axis=3)

            conv1 = self.conv(input, 16, 7, 2, activation='relu')

            conv2 = self.conv(conv1, 32, 5, 2, activation='relu')

            conv3 = self.conv(conv2, 64, 3, 2, activation='relu')

            conv4 = self.conv(conv3, 128, 3, 2, activation='relu')

            conv5 = self.conv(conv4, 256, 3, 2, activation='relu')

            conv6 = self.conv(conv5, 256, 3, 2, activation='relu')

            conv7 = self.conv(conv6, 512, 3, 2, activation='relu')

            dim = np.prod(conv7.shape[1:])

            flat1 = Lambda(lambda x: tf.reshape(x, [-1, dim]))(conv7)

            # translation
            fc1_tran = Dense(512, input_shape=(dim,))(flat1)

            fc2_tran = Dense(512, input_shape=(512,))(fc1_tran)

            fc3_tran = Dense(3, input_shape=(512,))(fc2_tran)

            # rotation
            fc1_rot = Dense(512, input_shape=(dim,))(flat1)

            fc2_rot = Dense(512, input_shape=(512,))(fc1_rot)

            fc3_rot = Dense(3, input_shape=(512,))(fc2_rot)

            # pose uncertainty
            fc1_unc = Dense(512, input_shape=(dim,))(flat1)

            fc2_unc = Dense(512, input_shape=(512,))(fc1_unc)

            fc3_unc = Dense(21, input_shape=(512,))(fc2_unc)#, activation = 'softplus'

            self.pose_model = Model([input1,input2], [fc3_tran, fc3_rot, fc3_unc])

    def build_localization_architecture(self):
        
        with tf.variable_scope('localization_model',reuse=self.reuse_variables):
            input1 = Input(batch_shape=(self.rnn_batch_size,self.params.sequence_size,3))

            input2 = Input(batch_shape=(self.rnn_batch_size,self.params.sequence_size,3))

            input3 = Input(batch_shape=(self.rnn_batch_size,self.params.sequence_size,21))

            input = concatenate([input1,input2,input3], axis=2)
        
            lstm_1 = LSTM(1024, batch_input_shape = (self.rnn_batch_size,self.params.sequence_size,27), stateful=True, return_sequences=True)(input)
    
            lstm_2 = LSTM(1024, stateful=True, return_sequences=True)(lstm_1)


            fc1_tran = TimeDistributed(Dense(512, input_shape=(1024,), activation='relu'))(lstm_2)

            fc2_tran = TimeDistributed(Dense(128, input_shape=(512,), activation='relu'))(fc1_tran)

            fc3_tran = TimeDistributed(Dense(3, input_shape=(128,)))(fc2_tran)

            fc1_rot = TimeDistributed(Dense(512, input_shape=(1024,), activation='relu'))(lstm_2)

            fc2_rot = TimeDistributed(Dense(128, input_shape=(512,), activation='relu'))(fc1_rot)

            fc3_rot = TimeDistributed(Dense(3, input_shape=(128,)))(fc2_rot)

            fc1_unc = TimeDistributed(Dense(512, input_shape=(1024,), activation='relu'))(lstm_2)

            fc2_unc = TimeDistributed(Dense(128, input_shape=(512,), activation='relu'))(fc1_unc)

            fc3_unc = TimeDistributed(Dense(21, input_shape=(128,)))(fc2_unc)

            self.localization_model = Model([input1,input2,input3], [fc3_tran, fc3_rot, fc3_unc])


    def build_model(self,img1,img2):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):

            disp1 = self.depth_model(img1)
            [disp1[i].set_shape(disp1[i]._keras_shape) for i in range(8)]
            disp2 = self.depth_model(img2)
            [disp2[i].set_shape(disp2[i]._keras_shape) for i in range(8)] 

            [trans, rot, unc] = self.pose_model([img1,img2])
            out_trans, out_rot, out_unc = trans, rot, unc
            self.tran_zero = trans[:1,:]
            self.rot_zero = rot[:1,:]

            trans = tf.reshape(trans,[self.rnn_batch_size,self.params.sequence_size,3])
            rot = tf.reshape(rot,[self.rnn_batch_size,self.params.sequence_size,3])
            unc = tf.reshape(unc,[self.rnn_batch_size,self.params.sequence_size,21])

            [trans_est, rot_est, unc_est] = self.localization_model([trans,rot,unc])

            trans_est.set_shape(trans_est._keras_shape)
            rot_est.set_shape(rot_est._keras_shape)
            unc_est.set_shape(unc_est._keras_shape)

            trans_est = tf.reshape(trans_est, [-1, 3])
            rot_est = tf.reshape(rot_est, [-1, 3])
            unc_est = tf.reshape(unc_est, [-1, 21])

        return disp1, disp2, out_trans, out_rot, out_unc, trans_est, rot_est, unc_est

    def build_outputs(self):
        # create base images & depthmap
        self.img_base = tf.tile(self.img_cur[:1,:,:,:], [self.params.batch_size,1,1,1])
        self.depthmap_base = tf.tile(self.depthmap1[0][:1,:,:,:], [self.params.batch_size,1,1,1])

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
        self.plus0 = projective_transformer(self.img_cur, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2[0], self.rot, self.tran)
        self.plus1 = projective_transformer(self.img_base, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2[0], self.rot_est, self.tran_est)
        self.plus2 = projective_transformer(self.img_cur, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2[0], self.rot_part, self.tran_part)

        # generate k+1 th depth image
        self.depthplus0 = projective_transformer(self.depthmap1[0], self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2[0], self.rot, self.tran)
        self.depthplus1 = projective_transformer(self.depthmap_base, self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2[0], self.rot_est, self.tran_est)
        self.depthplus2 = projective_transformer(self.depthmap1[0], self.focal_length1, self.focal_length2, self.c0, self.c1, self.depthmap2[0], self.rot_part, self.tran_part)



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

    def compute_abs_losses(self,r_gt,t_gt,r_est,t_est):
        with tf.variable_scope('losses', reuse=self.reuse_variables):

            # Generate poses
            M_gt  = compose_matrix(r_gt,t_gt)
            M_est = compose_matrix(r_est,t_est)

            # Compute dists
            r,t = decompose_matrix(tf.matmul(tf.matrix_inverse(M_gt),M_est))
            dist = concatenate([r,t],axis=1)
            dist = tf.expand_dims(dist,2)
            dist_t = tf.transpose(dist,perm=[0,2,1])

            # TOTAL LOSS
            return tf.reduce_mean(tf.sqrt(tf.matmul(dist_t,dist)))

    def build_losses(self):
        
        # PHOTOMETRIC REGISTRATION (temporal loss)
        self.l1_plus0 = self.compute_temporal_loss(self.plus0, self.img_next, self.unc) 
        self.l1_plus2 = self.compute_temporal_loss(self.plus2, self.img_next, self.unc) 
        self.l1_plus1 = self.compute_temporal_loss(self.plus1, self.img_next, self.unc_est)

        self.l1_depthplus0 = self.compute_temporal_loss(self.depthplus0, self.depthmap2[0], self.unc) 
        self.l1_depthplus2 = self.compute_temporal_loss(self.depthplus2, self.depthmap2[0], self.unc) 
        self.l1_depthplus1 = self.compute_temporal_loss(self.depthplus1, self.depthmap2[0], self.unc_est)

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
            tf.summary.image('depth',  self.depthmap1[0],   max_outputs=3, collections=self.model_collection)
            tf.summary.image('depthplus0',  self.depthplus0[0],   max_outputs=3, collections=self.model_collection)
            tf.summary.image('plus1',  self.plus1[0],   max_outputs=3, collections=self.model_collection)

            txtPredictions = tf.Print(tf.as_string(self.Q),[tf.as_string(self.Q)], message='predictions', name='txtPredictions')
            tf.summary.text('predictions', txtPredictions, collections=self.model_collection)

