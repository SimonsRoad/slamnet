
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.layers import Lambda
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Cropping2D, Dense, Flatten, Input, Reshape, LSTM, GRU, TimeDistributed
from keras.models import Model

class Models(object):
    """models for deepslam"""
    def __init__(self, img_shape, reuse_variables=None):

        self.img_shape = img_shape
        self.reuse_variables = reuse_variables

    def get_variance1(self, x):
        variance = self.conv(x, 3, 3, 1, activation='softplus')
#        variance = Lambda(lambda x: 0.01 + x)(variance)
        return variance

    def get_variance2(self, x):
        variance = self.conv(x, 1, 3, 1, activation='softplus')
#        variance = Lambda(lambda x: 0.01 + x)(variance)
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

    def build_depth_encoder(self):
        with tf.variable_scope('depth_encoder_model',reuse=self.reuse_variables):
            input = Input(batch_shape=self.img_shape)

            # encoder
            conv1 = self.conv_block(input, 32, 7)

            conv2 = self.conv_block(conv1, 64, 5)

            conv3 = self.conv_block(conv2, 128, 3)


            conv4 = self.conv_block(conv3, 256, 3)

            conv5 = self.conv_block(conv4, 512, 3)

            conv6 = self.conv_block(conv5, 512, 3)

            conv7 = self.conv_block(conv6, 512, 3)

            est  = [conv1, conv2, conv3, conv4, conv5, conv6, conv7]

            self.depth_encoder_model = Model(input, est)
    def build_depth_decoder(self):
        with tf.variable_scope('depth_decoder_model',reuse=self.reuse_variables):

            skip1 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/2,self.img_shape[2]/2,32])

            skip2 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/4,self.img_shape[2]/4,64])
        
            skip3 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/8,self.img_shape[2]/8,128])

            skip4 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/16,self.img_shape[2]/16,256])

            skip5 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/32,self.img_shape[2]/32,512])

            skip6 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/64,self.img_shape[2]/64,512])

            conv7 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/128,self.img_shape[2]/128,512])

            input = [skip1,skip2,skip3,skip4,skip5,skip6,conv7]

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

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:-1,:,:])(deconv1)
            if  s[2] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:,:-1,:])(deconv1)

            disp1 = self.get_depth(deconv1)

            # decoder2

            deconv4_2 = self.deconv_block(deconv5, 128, 3, skip3)
            disp4_2 = self.get_variance1(deconv4_2)

            deconv3_2 = self.deconv_block(deconv4_2, 64, 3, skip2)
            disp3_2 = self.get_variance1(deconv3_2)

            deconv2_2 = self.deconv_block(deconv3_2, 32, 3, skip1)
            disp2_2 = self.get_variance1(deconv2_2)

            deconv1_2 = self.deconv_block(deconv2_2, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_2)
            if  s[2] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_2)

            disp1_2 = self.get_variance1(deconv1_2)

            # decoder3

            deconv4_3 = self.deconv_block(deconv5, 128, 3, skip3)
            disp4_3 = self.get_variance2(deconv4_3)

            deconv3_3 = self.deconv_block(deconv4_3, 64, 3, skip2)
            disp3_3 = self.get_variance2(deconv3_3)

            deconv2_3 = self.deconv_block(deconv3_3, 32, 3, skip1)
            disp2_3 = self.get_variance2(deconv2_3)

            deconv1_3 = self.deconv_block(deconv2_3, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1_3 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_3)
            if  s[2] % 2 != 0:
                deconv1_3 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_3)

            disp1_3 = self.get_variance2(deconv1_3)

            disp_est  = [disp1, disp2, disp3, disp4, disp1_2, disp2_2, disp3_2, disp4_2, disp1_3, disp2_3, disp3_3, disp4_3]


            self.depth_decoder_model = Model(input, disp_est)

    def build_depth_variances_decoder(self):
        with tf.variable_scope('depth_var_decoder_model',reuse=self.reuse_variables):

            skip1 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/2,self.img_shape[2]/2,32])

            skip2 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/4,self.img_shape[2]/4,64])
        
            skip3 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/8,self.img_shape[2]/8,128])

            skip4 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/16,self.img_shape[2]/16,256])

            skip5 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/32,self.img_shape[2]/32,512])

            skip6 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/64,self.img_shape[2]/64,512])

            conv7 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/128,self.img_shape[2]/128,512])

            input = [skip1,skip2,skip3,skip4,skip5,skip6,conv7]

            # decoder1
            deconv7 = self.deconv_block(conv7, 512, 3, skip6)

            deconv6 = self.deconv_block(deconv7, 512, 3, skip5)

            deconv5 = self.deconv_block(deconv6, 256, 3, skip4)
            
            deconv4 = self.deconv_block(deconv5, 128, 3, skip3)

            deconv3 = self.deconv_block(deconv4, 64, 3, skip2)

            deconv2 = self.deconv_block(deconv3, 32, 3, skip1)

            deconv1 = self.deconv_block(deconv2, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:-1,:,:])(deconv1)
            if  s[2] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:,:-1,:])(deconv1)

            disp1 = self.get_variance1(deconv1)

            # decoder2

            deconv4_2 = self.deconv_block(deconv5, 128, 3, skip3)

            deconv3_2 = self.deconv_block(deconv4_2, 64, 3, skip2)

            deconv2_2 = self.deconv_block(deconv3_2, 32, 3, skip1)

            deconv1_2 = self.deconv_block(deconv2_2, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_2)
            if  s[2] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_2)

            disp1_2 = self.get_variance2(deconv1_2)

            disp_est  = [disp1, disp1_2]

            self.depth_var_decoder_model = Model(input, disp_est)

    def build_depth_variances_decoder2(self):
        with tf.variable_scope('depth_var_decoder_model2',reuse=self.reuse_variables):

            skip1 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/2,self.img_shape[2]/2,32])

            skip2 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/4,self.img_shape[2]/4,64])
        
            skip3 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/8,self.img_shape[2]/8,128])

            skip4 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/16,self.img_shape[2]/16,256])

            skip5 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/32,self.img_shape[2]/32,512])

            skip6 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/64,self.img_shape[2]/64,512])

            conv7 = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/128,self.img_shape[2]/128,512])

            input = [skip1,skip2,skip3,skip4,skip5,skip6,conv7]

            # decoder1
            deconv7 = self.deconv_block(conv7, 512, 3, skip6)

            deconv6 = self.deconv_block(deconv7, 512, 3, skip5)

            deconv5 = self.deconv_block(deconv6, 256, 3, skip4)
            
            deconv4 = self.deconv_block(deconv5, 128, 3, skip3)

            deconv3 = self.deconv_block(deconv4, 64, 3, skip2)

            deconv2 = self.deconv_block(deconv3, 32, 3, skip1)

            deconv1 = self.deconv_block(deconv2, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:-1,:,:])(deconv1)
            if  s[2] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:,:-1,:])(deconv1)

            disp1 = self.get_variance1(deconv1)

            # decoder2

            deconv4_2 = self.deconv_block(deconv5, 128, 3, skip3)

            deconv3_2 = self.deconv_block(deconv4_2, 64, 3, skip2)

            deconv2_2 = self.deconv_block(deconv3_2, 32, 3, skip1)

            deconv1_2 = self.deconv_block(deconv2_2, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_2)
            if  s[2] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_2)

            disp1_2 = self.get_variance2(deconv1_2)

            disp_est  = [disp1, disp1_2]

            self.depth_var_decoder_model2 = Model(input, disp_est)

    def build_depth_architecture(self):
        with tf.variable_scope('depth_model',reuse=self.reuse_variables):
            input = Input(batch_shape=self.img_shape)
            
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

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:-1,:,:])(deconv1)
            if  s[2] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:,:-1,:])(deconv1)

            disp1 = self.get_depth(deconv1)

            # decoder2

            deconv4_2 = self.deconv_block(deconv5, 128, 3, skip3)
            disp4_2 = self.get_variance1(deconv4_2)

            deconv3_2 = self.deconv_block(deconv4_2, 64, 3, skip2)
            disp3_2 = self.get_variance1(deconv3_2)

            deconv2_2 = self.deconv_block(deconv3_2, 32, 3, skip1)
            disp2_2 = self.get_variance1(deconv2_2)

            deconv1_2 = self.deconv_block(deconv2_2, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_2)
            if  s[2] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_2)

            disp1_2 = self.get_variance1(deconv1_2)

            # decoder3

            deconv4_3 = self.deconv_block(deconv5, 128, 3, skip3)
            disp4_3 = self.get_variance2(deconv4_3)

            deconv3_3 = self.deconv_block(deconv4_3, 64, 3, skip2)
            disp3_3 = self.get_variance2(deconv3_3)

            deconv2_3 = self.deconv_block(deconv3_3, 32, 3, skip1)
            disp2_3 = self.get_variance2(deconv2_3)

            deconv1_3 = self.deconv_block(deconv2_3, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1_3 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_3)
            if  s[2] % 2 != 0:
                deconv1_3 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_3)

            disp1_3 = self.get_variance2(deconv1_3)

            disp_est  = [disp1, disp2, disp3, disp4, disp1_2, disp2_2, disp3_2, disp4_2, disp1_3, disp2_3, disp3_3, disp4_3]

            self.depth_model = Model(input, disp_est)
    
    def build_pose_encoder(self):
        with tf.variable_scope('pose_encoder_model',reuse=self.reuse_variables):
            input1 = Input(batch_shape=self.img_shape)

            input2 = Input(batch_shape=self.img_shape)

            input = concatenate([input1,input2], axis=3)

            conv1 = self.conv(input, 16, 7, 2, activation='relu')

            conv2 = self.conv(conv1, 32, 5, 2, activation='relu')

            conv3 = self.conv(conv2, 64, 3, 2, activation='relu')

            conv4 = self.conv(conv3, 128, 3, 2, activation='relu')

            conv5 = self.conv(conv4, 256, 3, 2, activation='relu')

            conv6 = self.conv(conv5, 256, 3, 2, activation='relu')

            conv7 = self.conv(conv6, 512, 3, 2, activation='relu')

            self.pose_encoder_model = Model([input1,input2], conv7)

    def build_pose_decoder(self):
        with tf.variable_scope('pose_decoder_model',reuse=self.reuse_variables):

            input = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/128,self.img_shape[2]/128,512])

            dim = np.prod(input.shape[1:])

            flat1 = Lambda(lambda x: tf.reshape(x, [-1, dim]))(input)

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

            self.pose_decoder_model = Model(input, [fc3_tran, fc3_rot, fc3_unc])

    def build_pose_variances_decoder(self):
        with tf.variable_scope('pose_var_decoder_model',reuse=self.reuse_variables):

            input = Input(batch_shape=[self.img_shape[0],self.img_shape[1]/128,self.img_shape[2]/128,512])

            dim = np.prod(input.shape[1:])

            flat1 = Lambda(lambda x: tf.reshape(x, [-1, dim]))(input)

            # pose uncertainty
            fc1_unc = Dense(512, input_shape=(dim,))(flat1)

            fc2_unc = Dense(512, input_shape=(512,))(fc1_unc)

            fc3_unc = Dense(21, input_shape=(512,))(fc2_unc)#, activation = 'softplus'

            self.pose_var_decoder_model = Model(input, fc3_unc)

    def build_pose_architecture(self):
        
        with tf.variable_scope('pose_model',reuse=self.reuse_variables):
            input1 = Input(batch_shape=self.img_shape)

            input2 = Input(batch_shape=self.img_shape)

            input = concatenate([input1,input2], axis=3)

            conv1 = self.conv(input, 16, 7, 2, activation='relu')

            conv2 = self.conv(conv1, 32, 5, 2, activation='relu')

            conv3 = self.conv(conv2, 64, 3, 2, activation='relu')

            conv4 = self.conv(conv3, 128, 3, 2, activation='relu')

            conv5 = self.conv(conv4, 256, 3, 2, activation='relu')

            conv6 = self.conv(conv5, 256, 3, 2, activation='relu')

            conv7 = self.conv(conv6, 512, 3, 2, activation='relu')

            skip1 = conv1

            skip2 = conv2


            # decoder1

            deconv3 = self.deconv_block(conv3, 32, 3, skip2)
            
            deconv2 = self.deconv_block(deconv3, 32, 3, skip1)

            deconv1 = self.deconv_block(deconv2, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:-1,:,:])(deconv1)
            if  s[2] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:,:-1,:])(deconv1)

            img_var = self.get_variance1(deconv1)


            # decoder2

            deconv3_2 = self.deconv_block(conv3, 32, 3, skip2)
            
            deconv2_2 = self.deconv_block(deconv3_2, 32, 3, skip1)

            deconv1_2 = self.deconv_block(deconv2_2, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_2)
            if  s[2] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_2)

            depth_var = self.get_variance2(deconv1_2)


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

            self.pose_model = Model([input1,input2], [fc3_tran, fc3_rot, fc3_unc, img_var, depth_var])

    def build_localization_architecture(self):
        
        with tf.variable_scope('localization_model',reuse=self.reuse_variables):
            input1 = Input(batch_shape=(1,self.img_shape[0],3))

            input2 = Input(batch_shape=(1,self.img_shape[0],3))

            input3 = Input(batch_shape=(1,self.img_shape[0],21))

            input = concatenate([input1,input2,input3], axis=2)
        
            lstm_1 = LSTM(1024, batch_input_shape = (1,self.img_shape[0],27), stateful=True, return_sequences=True)(input)
    
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

    def build_mapping_architecture(self):
        
        with tf.variable_scope('mapping_model',reuse=self.reuse_variables):
            input1 = Input(batch_shape=[self.img_shape[0],self.img_shape[1],self.img_shape[2],16])

            input2 = Input(batch_shape=[self.img_shape[0],27])
        
            # encoder
            conv1 = self.conv_block(input1, 32, 7)

            conv2 = self.conv_block(conv1, 64, 5)

            conv3 = self.conv_block(conv2, 128, 3)

            conv4 = self.conv_block(conv3, 256, 3)

            conv5 = self.conv_block(conv4, 256, 3)

            conv6 = self.conv_block(conv5, 256, 3)

            conv7 = self.conv_block(conv6, 256, 3)

            skip1 = conv1

            skip2 = conv2

            skip3 = conv3
            
            skip4 = conv4

            skip5 = conv5

            skip6 = conv6


            # RNN
            dim = np.prod(conv7.shape[1:])
            flat1 = Lambda(lambda x: tf.reshape(x, [1, self.img_shape[0], dim]))(conv7)
            pose_flat = Lambda(lambda x: tf.reshape(x, [1, self.img_shape[0], 27]))(input2)
            flat2 = concatenate([flat1,pose_flat],axis=2)
            dim2 = dim + pose_flat.shape[2]

            lstm_1 = LSTM(512, batch_input_shape = (1, self.img_shape[0], dim2), stateful=True, return_sequences=True)(flat2)

            lstm_2 = LSTM(int(dim), stateful=True, return_sequences=True)(lstm_1)

            unflat1 = Lambda(lambda x: tf.reshape(x, conv7.shape))(lstm_2)

            # decoder1
            deconv7 = self.deconv_block(unflat1, 256, 3, skip6)

            deconv6 = self.deconv_block(deconv7, 256, 3, skip5)

            deconv5 = self.deconv_block(deconv6, 256, 3, skip4)

            deconv4 = self.deconv_block(deconv5, 128, 3, skip3)

            deconv3 = self.deconv_block(deconv4, 64, 3, skip2)

            deconv2 = self.deconv_block(deconv3, 32, 3, skip1)

            deconv1 = self.deconv_block(deconv2, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:-1,:,:])(deconv1)
            if  s[2] % 2 != 0:
                deconv1 = Lambda(lambda x: x[:,:,:-1,:])(deconv1)

            disp = self.get_depth(deconv1)

            # decoder2

            deconv4_2 = self.deconv_block(deconv5, 128, 3, skip3)

            deconv3_2 = self.deconv_block(deconv4_2, 64, 3, skip2)

            deconv2_2 = self.deconv_block(deconv3_2, 32, 3, skip1)

            deconv1_2 = self.deconv_block(deconv2_2, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_2)
            if  s[2] % 2 != 0:
                deconv1_2 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_2)

            unc_i = self.get_variance1(deconv1_2)

            # decoder3

            deconv4_3 = self.deconv_block(deconv5, 128, 3, skip3)

            deconv3_3 = self.deconv_block(deconv4_3, 64, 3, skip2)

            deconv2_3 = self.deconv_block(deconv3_3, 32, 3, skip1)

            deconv1_3 = self.deconv_block(deconv2_3, 16, 3, None)

            s = self.img_shape
            if  s[1] % 2 != 0:
                deconv1_3 = Lambda(lambda x: x[:,:-1,:,:])(deconv1_3)
            if  s[2] % 2 != 0:
                deconv1_3 = Lambda(lambda x: x[:,:,:-1,:])(deconv1_3)

            unc_d = self.get_variance2(deconv1_3)


            dim3 = np.prod(conv7.shape[1:])

            unflat2 = Lambda(lambda x: tf.reshape(x, [-1, dim3]))(lstm_2)

            # translation
            fc1_tran = Dense(512, input_shape=(dim,))(unflat2)

            fc2_tran = Dense(512, input_shape=(512,))(fc1_tran)

            fc3_tran = Dense(3, input_shape=(512,))(fc2_tran)

            # rotation
            fc1_rot = Dense(512, input_shape=(dim,))(unflat2)

            fc2_rot = Dense(512, input_shape=(512,))(fc1_rot)

            fc3_rot = Dense(3, input_shape=(512,))(fc2_rot)

            # pose uncertainty
            fc1_unc = Dense(512, input_shape=(dim,))(unflat2)

            fc2_unc = Dense(512, input_shape=(512,))(fc1_unc)

            fc3_unc = Dense(21, input_shape=(512,))(fc2_unc)

            self.mapping_model = Model([input1,input2], [disp, unc_i, unc_d, fc3_tran, fc3_rot, fc3_unc])
