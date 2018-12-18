
"""Deepslam data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf

class DeepslamDataloader(object):
    """deepslam dataloader"""

    def __init__(self, data_path, filenames_file, params, mode):
#        print(tf.__version__)
        self.data_path = data_path
        self.params = params
        self.mode = mode

        self.dataset = None

        dataset = tf.data.TextLineDataset(filenames_file)

#        dataset = dataset.apply(tf.contrib.data.sliding_window_batch(self.params.batch_size))
        dataset = dataset.window(self.params.batch_size, 1, 1, True).flat_map(lambda x: x.batch(self.params.batch_size))
        dataset = dataset.shuffle(buffer_size=40000, reshuffle_each_iteration=True)
        dataset = dataset.apply(tf.data.experimental.unbatch())

        split_line = dataset.map(lambda string: tf.string_split([string]).values)

        # we load only one image for test, except if we trained a stereo model
        if mode == 'test':
            image_path  = tf.string_join([self.data_path, split_line[1]])
            image_o  = self.read_image(image_path)
        else:
            self.dataset = split_line.map(self.process_data)

        if mode == 'train':
#            self.dataset = self.dataset.apply(tf.contrib.data.sliding_window_batch(self.params.batch_size))
#            self.dataset = self.dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
#            self.dataset_img_cur = self.dataset.img_cur.window(self.params.batch_size, 1, 1, True).flat_map(lambda x: x.batch(self.params.batch_size))

            self.dataset = self.dataset.batch(self.params.batch_size)
            self.dataset = self.dataset.filter(self.check_batch_indices)
            self.dataset = self.dataset.prefetch(1)

        elif mode == 'test':
            self.image_batch = image_o
            self.image_batch.set_shape( [1, None, None, 3])

    def check_batch_indices(self, indices,im1,im2,cp):
        all_elems_equal = tf.reduce_all(tf.equal(indices, indices))
        return all_elems_equal

    def process_data(self, strings):

        seq_index = cam_params = strings[0]
        seq_index = tf.string_to_number(seq_index)
        seq_index = tf.reshape(seq_index, [1])
 
        image_path  = tf.string_join([self.data_path, strings[1]])
        img_cur  = self.read_image(image_path)
        next_image_path  = tf.string_join([self.data_path, strings[2]])
        img_next  = self.read_image(next_image_path)

        cam_params = strings[3:9]
        cam_params = tf.string_to_number(cam_params)
        cam_params = tf.reshape(cam_params, [6])

        return seq_index, img_cur, img_next, cam_params

    def read_image(self, image_path):
        image = tf.image.decode_png(tf.read_file(image_path))
        image  = tf.image.convert_image_dtype(image,  tf.float32) 
        image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)
        image.set_shape( [self.params.height, self.params.width, 3])
        return image
