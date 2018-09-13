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


#    pred = tf.less(sy,1e-6)
#    r = Lambda( lambda x: tf.cond(x[0], lambda: tf.atan2(x[3],x[4]), lambda: tf.atan2(-x[2], x[1])) )([pred, R[:,1,1],R[:,1,2],R[:,2,1],R[:,2,2]])
#    p = tf.atan2(-R[:,2,0], sy)
#    y = Lambda( lambda x: tf.cond(x[0], lambda: tf.atan2(x[2],x[1]), lambda: tf.zeros([1],tf.float32)) )([pred, R[:,0,0],R[:,1,0]])
    

    res_r = tf.zeros([1,1],tf.float32)
    res_p = tf.zeros([1,1],tf.float32)
    res_y = tf.zeros([1,1],tf.float32)
    for i in range(_num_batch):
        r,p,y = tf.cond(tf.less(sy[i],1e-6), lambda: f2(R[i,:,:],sy[i]), lambda: f1(R[i,:,:],sy[i]))
#        print(singular)
#        if singular:
#            r = tf.atan2(-R[i,1,2], R[i,1,1])
#            p = tf.atan2(-R[i,2,0], sy[i])
#            y = tf.zeros([1],tf.float32)
#        else:
#            r = tf.atan2(R[i,2,1],R[i,2,2])
#            p = tf.atan2(R[i,2,0],sy[i]) 
#            y = tf.atan2(R[i,1,0],R[i,0,0])
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
