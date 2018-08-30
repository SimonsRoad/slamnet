
"""Deepslam data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf

def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int64])

class DeepslamDataloader(object):
    """deepslam dataloader"""

    def __init__(self, data_path, filenames_file, params, dataset, mode):
        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode

#        self.seq_nums = None
        self.image_batch  = None
        self.next_image_batch  = None
        self.cam_params_batch = None
        self.poses_batch = None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values
        
        # we load only one image for test, except if we trained a stereo model
        if mode == 'test':
            image_path  = tf.string_join([self.data_path, split_line[1]])
            image_o  = self.read_image(image_path)
        else:
#            self.seq_nums = tf.string_to_number(split_line[0])
            image_path  = tf.string_join([self.data_path, split_line[1]])
            next_image_path  = tf.string_join([self.data_path, split_line[2]])
            cam_params = tf.string_to_number(split_line[3:10])
            height_o = tf.string_to_number(split_line[10])
            width_o = tf.string_to_number(split_line[11])
            poses = tf.string_to_number(split_line[12:])

            image_o  = self.read_image(image_path)
            next_image_o  = self.read_image(next_image_path)

            # set cam_params shape
            cam_params = tf.reshape(cam_params, [7])
            cam_params = tf.expand_dims(cam_params,0)
            h_tensor = tf.expand_dims(tf.cast(tf.constant([self.params.height]), tf.float32),0)
            w_tensor = tf.expand_dims(tf.cast(tf.constant([self.params.width]), tf.float32),0)
            cam_params = tf.squeeze(tf.concat([cam_params, h_tensor/height_o, w_tensor/width_o],1))

            # set poses shape
            poses = tf.reshape(poses,[6])


        if mode == 'train':
            # set image shape
            image_o.set_shape( [self.params.height, self.params.width, 3])
            next_image_o.set_shape( [self.params.height, self.params.width, 3])
 
            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * params.batch_size
            self.image_batch, self.next_image_batch, self.cam_params_batch, self.poses_batch = tf.train.batch([image_o, next_image_o, cam_params, poses], params.batch_size)

        elif mode == 'test':
            self.image_batch = image_o
            self.image_batch.set_shape( [1, None, None, 3])

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')
        
        image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height    = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image  =  image[:crop_height,:,:]

        image  = tf.image.convert_image_dtype(image,  tf.float32) 
        image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image
