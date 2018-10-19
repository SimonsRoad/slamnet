from __future__ import absolute_import, division, print_function
import numpy as np
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

    sy = tf.sqrt(tf.square(R[:,0,0])+tf.square(R[:,1,0]))

    def f1(rot,sp): 
        r = tf.atan2(rot[2,1],rot[2,2])
        p = tf.atan2(-rot[2,0],sp)
        y = tf.atan2(rot[1,0],rot[0,0]) 
        return r, p ,y
    def f2(rot,sp): 
        r = tf.atan2(-rot[1,2], rot[1,1])
        p = tf.atan2(-rot[2,0], sp)
        y = tf.zeros([],tf.float32)
        return r, p ,y

    res_r = tf.zeros([1,1],tf.float32)
    res_p = tf.zeros([1,1],tf.float32)
    res_y = tf.zeros([1,1],tf.float32)
    for i in range(_num_batch):
        r,p,y = tf.cond(tf.less(sy[i],1e-6), lambda: f2(R[i,:,:],sy[i]), lambda: f1(R[i,:,:],sy[i]))
        r = tf.expand_dims(r,0)
        r = tf.expand_dims(r,1)
        p = tf.expand_dims(p,0)
        p = tf.expand_dims(p,1)
        y = tf.expand_dims(y,0)
        y = tf.expand_dims(y,1) 
        res_r = concatenate([res_r,r],axis=0)
        res_p = concatenate([res_p,p],axis=0)
        res_y = concatenate([res_y,y],axis=0)
    return res_r[1:,:], res_p[1:,:], res_y[1:,:]

      

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
    rot = concatenate([r,p,y],axis=1)
    trans = M[:,:3,3]
    return rot, trans

def np_rpy_to_matrix(r, p, y):

    yawMatrix = [np.cos(y), -np.sin(y), 0,
    np.sin(y), np.cos(y), 0,
    0, 0, 1]
    yawMatrix = np.reshape(yawMatrix, [3, 3])

    pitchMatrix = [np.cos(p), 0, np.sin(p),
    0, 1, 0,
    -np.sin(p), 0, np.cos(p)]
    pitchMatrix = np.reshape(pitchMatrix, [3, 3])

    rollMatrix = [1,0, 0,
    0, np.cos(r), -np.sin(r),
    0, np.sin(r), np.cos(r)]
    rollMatrix = np.reshape(rollMatrix, [3, 3])

    R = np.matmul(np.matmul(yawMatrix, pitchMatrix), rollMatrix)
    return R

def np_matrix_to_rpy(R):


    sy = np.sqrt(np.square(R[0,0])+np.square(R[1,0]))

    if sy<1e-6:
        r = np.arctan2(-R[1,2], R[1,1])
        p = np.arctan2(-R[2,0], sy)
        y = 0
    else:
        r = np.arctan2(R[2,1],R[2,2])
        p = np.arctan2(-R[2,0],sy)
        y = np.arctan2(R[1,0],R[0,0]) 

    return r,p,y

def np_compose_matrix(rot, trans): 

    zero_one = np.array([[0,0,0,1]])

    R = np_rpy_to_matrix(rot[0],rot[1],rot[2])
    trans = np.expand_dims(trans,1)

    M = np.concatenate([R,trans],axis=1)
    M = np.concatenate([M,zero_one], axis=0)

    return M

def np_decompose_matrix(M):

    r,p,y = np_matrix_to_rpy(M[:3,:3])
    rot = np.array([r,p,y])
    trans = M[:3,3]
    return rot, trans
