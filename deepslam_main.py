

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras import backend as K

from mapping_model import *
from localization_model import *
from deepslam_model import *
from deepslam_dataloader import *
from average_gradients import *

parser = argparse.ArgumentParser(description='DeepSLAM TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--train_mode',                      type=str,   help='localization, mapping or slam', default='slam')
parser.add_argument('--model_name',                type=str,   help='model name', default='deepslam')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)

parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)

parser.add_argument('--batch_size',                type=int,   help='batch size', default=5)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=30)
parser.add_argument('--sequence_size',             type=int,   help='size of sequence', default=5)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)

parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)

parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--prev_checkpoint_path',        type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')

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

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    seq_nums = [i.split(' ')[0] for i in lines]
    f.close()
    return len(lines), seq_nums

def train(params):
    """Training loop."""

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples, seq_nums = count_text_lines(args.filenames_file)
        
        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        seq_nums = np.reshape(seq_nums, (steps_per_epoch,int(params.batch_size)))
        seq_per_epoch = seq_nums[:,0]

        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
#        values = [args.learning_rate, args.learning_rate, args.learning_rate]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        opt_step = tf.train.AdamOptimizer(learning_rate)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        # Load dataset
        dataloader = DeepslamDataloader(args.data_path, args.filenames_file, params, args.mode)
        dataset = dataloader.dataset
        iterator = dataset.make_initializable_iterator()
        image, next_image, poses, next_poses, cam_params = iterator.get_next()
        init_op = iterator.initializer
        image.set_shape( [params.batch_size, params.height, params.width, 3])
        next_image.set_shape( [params.batch_size,params.height, params.width, 3])
        poses.set_shape( [params.batch_size,6]) 
        next_poses.set_shape( [params.batch_size,6])   
        cam_params.set_shape( [params.batch_size,6])

        # split for each gpu
        image_splits  = tf.split(image,  args.num_gpus, 0)
        next_image_splits = tf.split(next_image, args.num_gpus, 0)
        poses_splits = tf.split(poses, args.num_gpus, 0)
        next_poses_splits = tf.split(next_poses, args.num_gpus, 0)
        cam_params_splits = tf.split(cam_params, args.num_gpus, 0)

        tower_grads  = []
        tower_losses = []
        reuse_variables = True
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):
                    
                    if args.train_mode == 'mapping':
                        model = MappingModel(params, args.mode, image_splits[i], next_image_splits[i], poses_splits[i], next_poses_splits[i], cam_params_splits[i], reuse_variables, i)
                    elif args.train_mode == 'localization':
                        model = LocalizationModel(params, args.mode, image_splits[i], next_image_splits[i], poses_splits[i], next_poses_splits[i], cam_params_splits[i], reuse_variables, i)
                    elif args.train_mode == 'slam':
                        model = DeepslamModel(params, args.mode, image_splits[i], next_image_splits[i], poses_splits[i], next_poses_splits[i], cam_params_splits[i], reuse_variables, i)

                    loss = model.total_loss
                    tower_losses.append(loss)

                    reuse_variables = True

                    if args.prev_checkpoint_path != '':
                        
                        if args.train_mode == 'mapping':
                            train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mapping_model')# + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='depth_model')
                        elif args.train_mode == 'localization':
                            train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='localization_model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pose_model')
                        elif args.train_mode == 'slam':
                            train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='localization_model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mapping_model')

                        grads = opt_step.compute_gradients(loss,var_list=train_vars)
                    else:
                        grads = opt_step.compute_gradients(loss)

                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver()

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(init_op)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()


        # LOAD CHECKPOINT IF SET
        if args.prev_checkpoint_path != '':
            if args.train_mode == 'mapping':
                load_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pose_model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='depth_model')
            elif args.train_mode == 'localization':
                load_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pose_model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='depth_model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mapping_model')
            elif args.train_mode == 'slam':
                load_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pose_model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='depth_model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mapping_model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='localization_model')
            
            loader = tf.train.Saver(load_vars)
            loader.restore(sess, args.prev_checkpoint_path)

        if args.checkpoint_path != '':
            restore_path = args.checkpoint_path.split(".")[0]
            train_saver.restore(sess, args.checkpoint_path)

            if args.retrain:
                sess.run(global_step.assign(0))


        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            idx = step % steps_per_epoch
            if idx ==0:
                sess.run(init_op)

            model.modelnets.mapping_model.reset_states()  


            if step % 100 == 0:
                _, loss_value, summary_str, poses_txt = sess.run([apply_gradient_op, total_loss,summary_op,model.poses_txt])
                summary_writer.add_summary(summary_str, global_step=step)
                print(poses_txt)
            else:
                _, loss_value = sess.run([apply_gradient_op, total_loss])

            duration = time.time() - before_op_time
            if step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
            if step and (step+1) % (steps_per_epoch*5) == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

def test(params):
    """Test function."""


def main(_):

    params = deepslam_parameters(
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        sequence_size=args.sequence_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        full_summary=args.full_summary)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)

if __name__ == '__main__':
    tf.app.run()
