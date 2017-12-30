import tensorflow.contrib.slim as slim
import tensorlayer as tl
import tensorflow as tf
import numpy as np

class GAS_Discriminator(object):
    """
        The small discriminator of Generative Auxiliary Strategy

        n_filter = 32: 713MB
        n_filter = 16: 457MB
    """
    
    def __init__(self, ph):
        self.network = None
        self.logit = self.build(ph).outputs
        
    def add_layer(self, network, n_filter, with_bn = True, width_multiplier = 1, name = "layer"):
        """
            From the normal block of DCGAN
        """
        network = self.inception_conv(network, n_filter, width_multiplier, sc = name + 'conv_ds')
        if with_bn == True:
            network = tl.layers.BatchNormLayer(network, name = name +'batchnorm_layer')
        network = tf.nn.relu(network.outputs)
        network = tl.layers.InputLayer(network, name = name + 'aligned_layer_2')
        network = tl.layers.MaxPool2d(network, name = name + 'maxpool')
        return network

    def build(self, ph, base_filter=16):
        """
            Get network
        """
        with tf.name_scope('discriminator'):
            self.network = tl.layers.InputLayer(ph)
            self.network = self.add_layer(self.network, n_filter = base_filter, with_bn = False, name = '1')
            self.network = self.add_layer(self.network, n_filter = base_filter * (2 ** 1), name = '2')
            self.network = self.add_layer(self.network, n_filter = base_filter * (2 ** 2), name = '3')
            self.network = self.add_layer(self.network, n_filter = base_filter * (2 ** 3), name = '4')
            self.network = tl.layers.FlattenLayer(self.network)
            self.network = tl.layers.DenseLayer(self.network, n_units = 1)
            return self.network

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.device('/device:GPU:0'):
            # Define network
            ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
            model = Model(ph)
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.1
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                while True:
                    sess.run(model.logit, feed_dict={
                        ph: np.random.random([32, 224, 224, 3])
                    })