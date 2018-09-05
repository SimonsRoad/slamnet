
"""Deepslam data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf

def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int64])

class DeepslamDataloader(object):
    """deepslam dataloader"""

    def __init__(self, data_path, filenames_file, params, mode):
        self.data_path = data_path
        self.params = params
        self.mode = mode

        self.dataset = None

        dataset = tf.data.TextLineDataset(filenames_file)
        split_line = dataset.map(lambda string: tf.string_split([string]).values)

        # we load only one image for test, except if we trained a stereo model
        if mode == 'test':
            image_path  = tf.string_join([self.data_path, split_line[1]])
            image_o  = self.read_image(image_path)
        else:
            self.dataset = split_line.map(self.process_data)

        if mode == 'train':
            self.dataset = self.dataset.batch(self.params.batch_size)
            self.dataset = self.dataset.prefetch(1)

        elif mode == 'test':
            self.image_batch = image_o
            self.image_batch.set_shape( [1, None, None, 3])

    def process_data(self, strings):
        image_path  = tf.string_join([self.data_path, strings[1]])
        img_cur  = self.read_image(image_path)
        next_image_path  = tf.string_join([self.data_path, strings[2]])
        img_next  = self.read_image(next_image_path)
        poses = strings[3:]
        poses = tf.string_to_number(poses)
        poses = tf.reshape(poses, [6])
        return img_cur, img_next, poses

    def read_image(self, image_path):
        path_length = string_length_tf(image_path)[0]
        image = tf.image.decode_png(tf.read_file(image_path))
        image  = tf.image.convert_image_dtype(image,  tf.float32) 
        image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)
        image.set_shape( [self.params.height, self.params.width, 3])
        return image
