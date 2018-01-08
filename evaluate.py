from model import LSGAN_large, LSGAN_origin, LSGAN_dense
from data_helper import ImageHandler
import tensorflow as tf
import numpy as np

evaluate_period = 100

if __name__ == '__main__':
    # Define image handler
    handler = ImageHandler(dataset_name = 'mnist', resize_length = 32)

    # Define config object
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    
    # ----------------------------------------------------------------------------
    # Test on original LSGAN
    # ----------------------------------------------------------------------------
    with tf.Graph().as_default():
        with tf.defive('/GPU:0'):
            # Define network
            z_placeholder = tf.placeholder(tf.float32, [None, 100])
            img_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1])
            model = LSGAN_large()
            model.build(z_placeholder, img_placeholder)

