
from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.layers import Lambda
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Cropping2D, Dense, Flatten, Input, Reshape, LSTM, GRU, TimeDistributed
from keras.models import Model

from transformers import *

deepslam_parameters = namedtuple('parameters',
                        'height, width, '
                        'batch_size, '
                        'sequence_size,'
                        'num_threads, '
                        'num_epochs, '
                        'full_summary')

class DeepslamModel(object):
    """deepslam model"""

    def __init__(self, params, mode, img_cur, img_next, poses_cur, poses_next, reuse_variables=None, model_index=0):

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
        self.build_slam_architecture()
#        self.build_slam_architecture_addon()

        [self.tran_est, self.rot_est, self.unc_est] = self.build_model(self.img_cur, self.img_next)#

        if self.mode == 'test':
            return

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

    def build_slam_architecture(self):
        
        with tf.variable_scope('slam_model',reuse=self.reuse_variables):
            input1 = Input(batch_shape=(self.rnn_batch_size,self.params.sequence_size,3))

            input2 = Input(batch_shape=(self.rnn_batch_size,self.params.sequence_size,3))

            input3 = Input(batch_shape=(self.rnn_batch_size,self.params.sequence_size,21))

            input = concatenate([input1,input2,input3], axis=2)
        
            lstm_1 = LSTM(1024, batch_input_shape = (self.rnn_batch_size,self.params.sequence_size,27), stateful=True, return_sequences=True)(input)
    
            lstm_2 = LSTM(1024, stateful=True, return_sequences=True)(lstm_1)#batch_input_shape = (self.rnn_batch_size,self.params.sequence_size,27), 


            fc1_tran = TimeDistributed(Dense(512, input_shape=(1024,), activation='relu'))(lstm_2)

            fc2_tran = TimeDistributed(Dense(128, input_shape=(512,), activation='relu'))(fc1_tran)

            fc3_tran = TimeDistributed(Dense(3, input_shape=(128,)))(fc2_tran)

            fc1_rot = TimeDistributed(Dense(512, input_shape=(1024,), activation='relu'))(lstm_2)

            fc2_rot = TimeDistributed(Dense(128, input_shape=(512,), activation='relu'))(fc1_rot)

            fc3_rot = TimeDistributed(Dense(3, input_shape=(128,)))(fc2_rot)

            fc1_unc = TimeDistributed(Dense(512, input_shape=(1024,), activation='relu'))(lstm_2)

            fc2_unc = TimeDistributed(Dense(128, input_shape=(512,), activation='relu'))(fc1_unc)

            fc3_unc = TimeDistributed(Dense(21, input_shape=(128,)))(fc2_unc)

            self.slam_model = Model([input1,input2,input3], [fc3_tran, fc3_rot, fc3_unc])

    def build_slam_architecture_addon(self):
        
        with tf.variable_scope('slam_model_addon',reuse=self.reuse_variables):

            input1 = Input(batch_shape=(self.rnn_batch_size,self.params.sequence_size,3))

            input2 = Input(batch_shape=(self.rnn_batch_size,self.params.sequence_size,3))

            input3 = Input(batch_shape=(self.rnn_batch_size,self.params.sequence_size,21))

            input = concatenate([input1,input2,input3], axis=2)
        
            lstm_1 = LSTM(1024, batch_input_shape = (self.rnn_batch_size,self.params.sequence_size,27), stateful=True, return_sequences=True)(input)
    
            lstm_2 = LSTM(1024, batch_input_shape = (self.rnn_batch_size,self.params.sequence_size,27), stateful=True, return_sequences=True)(lstm_1)

            fc1_unc = TimeDistributed(Dense(512, input_shape=(1024,), activation='relu'))(lstm_2)

            fc2_unc = TimeDistributed(Dense(128, input_shape=(512,), activation='relu'))(fc1_unc)

            fc3_unc = TimeDistributed(Dense(21, input_shape=(128,)))(fc2_unc)

            self.slam_model_addon = Model([input1,input2,input3], fc3_unc)


    def build_model(self,img1,img2):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):

            [trans, rot, unc] = self.pose_model([img1,img2])
            self.tran_zero = trans[:1,:]
            self.rot_zero = rot[:1,:]

            trans = tf.reshape(trans,[self.rnn_batch_size,self.params.sequence_size,3])
            rot = tf.reshape(rot,[self.rnn_batch_size,self.params.sequence_size,3])
            unc = tf.reshape(unc,[self.rnn_batch_size,self.params.sequence_size,21])

            [trans_est, rot_est, unc_est] = self.slam_model([trans,rot,unc])
#            unc_est = self.slam_model_addon([trans,rot,unc])

            trans_est.set_shape(trans_est._keras_shape)
            rot_est.set_shape(rot_est._keras_shape)
            unc_est.set_shape(unc_est._keras_shape)

            trans_est = tf.reshape(trans_est, [-1, 3])
            rot_est = tf.reshape(rot_est, [-1, 3])
            unc_est = tf.reshape(unc_est, [-1, 21])

        return trans_est, rot_est, unc_est



    def compute_SE3_estimates(self):
        in_poses = tf.reshape(self.poses_cur,[self.rnn_batch_size,self.params.sequence_size,6])
        rot_init = tf.reshape(in_poses[:,:1,:3],[self.rnn_batch_size,3])
        tran_init = tf.reshape(in_poses[:,:1,3:],[self.rnn_batch_size,3])
        M_init = compose_matrix(rot_init,tran_init)
        M_init = tf.expand_dims(M_init,1)
        M_init_tile = tf.tile(M_init, tf.stack([1,self.params.sequence_size,1,1]))
        M_init = tf.reshape(M_init_tile,[-1,4,4]) 
        M_delta = compose_matrix(self.rot_est,self.tran_est)
        M_est = tf.matmul(M_init,M_delta)
        
        M_prev = M_delta[1:,:,:]
        eye = batch_identity = tf.eye(4, batch_shape=[1])
        M_prev = concatenate([eye,M_prev],axis=0)
        M_prev_inv = tf.matrix_inverse(M_prev)
        M_delta_est = tf.matmul(M_prev_inv,M_delta)        

        return M_est, M_delta_est

    def compute_SE3_gt(self):
        M_cur = compose_matrix(self.poses_cur[:,:3], self.poses_cur[:,3:])
        M_cur_inv = tf.matrix_inverse(M_cur)
        M_next = compose_matrix(self.poses_next[:,:3],self.poses_next[:,3:])
        M_delta = tf.matmul(M_cur_inv,M_next)
        M_gt  = compose_matrix(self.poses_next[:,:3],self.poses_next[:,3:])

        return M_gt, M_delta


    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            
            # Generate Q
            L = tf.contrib.distributions.fill_triangular(self.unc_est)
            Lt = tf.transpose(L,perm=[0, 2, 1])
            self.Q = tf.matmul(L,Lt)


            # Generate poses
            M_est, M_delta_est = self.compute_SE3_estimates()
            M_gt, M_delta  = self.compute_SE3_gt()#compose_matrix(self.poses_next[:,:3],self.poses_next[:,3:])

#            # Compute delta dists
#            r,t = decompose_matrix(tf.matmul(tf.matrix_inverse(M_delta),M_delta_est))
#            dist_delta = concatenate([r,t],axis=1)
#            dist_delta = tf.expand_dims(dist_delta,2)
#            dist_delta_t = tf.transpose(dist_delta,perm=[0,2,1])
#            self.dist_sum= tf.reduce_mean(tf.sqrt(tf.matmul(dist_delta_t,dist_delta)))

            # Compute dists
            r,t = decompose_matrix(tf.matmul(tf.matrix_inverse(M_gt),M_est))
            dist = concatenate([r,t],axis=1)
            dist = tf.expand_dims(dist,2)
            dist_t = tf.transpose(dist,perm=[0,2,1])

            # Compute mdist
            res_Q_norm = tf.norm(self.Q,axis=[1,2])
#            res_Q_norm = Lambda(lambda x: 1.0 + x)(res_Q_norm)
            mdist = tf.matmul(tf.matmul(dist_t,tf.matrix_inverse(self.Q)),dist) + tf.log(res_Q_norm)
#            mdist = tf.matmul(dist_t,dist)# + tf.log(res_Q_norm) 

            # TOTAL LOSS
            self.dist_sum= tf.reduce_mean(tf.sqrt(tf.matmul(dist_t,dist)))
            self.total_loss = tf.reduce_mean(mdist)
            self.poses_txt = dist[-1,:]#[poses_est[-1,:],poses_gt[-1,:]]#[poses_gt,poses_est]

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            tf.summary.scalar('mean_dist', self.dist_sum, collections=self.model_collection)
            tf.summary.image('img_cur', self.img_cur,  max_outputs=3, collections=self.model_collection)
            tf.summary.image('img_next',  self.img_next,   max_outputs=3, collections=self.model_collection)

            txtPredictions = tf.Print(tf.as_string(self.Q),[tf.as_string(self.Q)], message='predictions', name='txtPredictions')
            tf.summary.text('predictions', txtPredictions, collections=self.model_collection)

