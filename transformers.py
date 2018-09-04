from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.layers import Lambda
from keras.layers import concatenate

def rpy_to_matrix(r, p, y):

    _num_batch = r.shape[0]    

    zeros = tf.zeros([_num_batch], tf.float32)
    ones = tf.ones([_num_batch], tf.float32)

    yawMatrix = tf.stack([tf.cos(y), -tf.sin(y), zeros,
    tf.sin(y), tf.cos(y), zeros,
    zeros, zeros, ones], axis=1)
    yawMatrix = tf.reshape(yawMatrix, [-1, 3, 3])

    pitchMatrix = tf.stack([tf.cos(p), zeros, tf.sin(p),
    zeros, ones, zeros,
    -tf.sin(p), zeros, tf.cos(p)], axis=1)
    pitchMatrix = tf.reshape(pitchMatrix, [-1, 3, 3])

    rollMatrix = tf.stack([ones, zeros, zeros,
    zeros, tf.cos(r), -tf.sin(r),
    zeros, tf.sin(r), tf.cos(r)], axis=1)
    rollMatrix = tf.reshape(rollMatrix, [-1, 3, 3])

    R = tf.matmul(tf.matmul(yawMatrix, pitchMatrix), rollMatrix)
    return R

def matrix_to_rpy(R):

    _num_batch = R.shape[0]  

    r = tf.atan2(R[:,2,1],R[:,2,2])
    p = tf.atan2(-R[:,2,0],tf.sqrt(tf.square(R[:,2,1])+tf.square(R[:,2,2])))
    y = tf.atan2(R[:,1,0],R[:,0,0])

    return r, p ,y


def compose_matrix(rot, trans):
    
    _num_batch = rot.shape[0]  

    zeros = tf.zeros([_num_batch,1,3], tf.float32)
    ones = tf.ones([_num_batch,1,1], tf.float32)
    zero_one = concatenate([zeros,ones],axis=2)

    R = rpy_to_matrix(rot[:,0],rot[:,1],rot[:,2])
    trans = tf.expand_dims(trans,2)

    M = concatenate([R,trans],axis=2)
    M = concatenate([M,zero_one], axis=1)

    return M

def decompose_matrix(M):

    r,p,y = matrix_to_rpy(M[:,:3,:3])
    rot = tf.stack([r,p,y],axis=1)
    trans = M[:,:3,1]

    return rot, trans
