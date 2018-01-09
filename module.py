from ops import depthwise_separable_conv2d, inception_conv2d, inception_conv2d_with_pwconv
import tensorflow.contrib.slim as slim
import tensorlayer as tl
import tensorflow as tf
import numpy as np

class GAS(object):
    """
        The small discriminator of Generative Auxiliary Strategy

        n_filter = 32: 713MB
        n_filter = 16: 457MB
    """
    pass    

class Discriminator_large(object):
    """
        The small discriminator of Generative Auxiliary Strategy

        n_filter = 32: 713MB
        n_filter = 16: 457MB
    """
    
    def __init__(self, base_filter=32):
        self.base_filter = base_filter

    def add_layer(self, network, n_filter, with_bn = True, width_multiplier = 4, name = "layer"):
        """
            From the normal block of DCGAN
        """
        network = tl.layers.Conv2d(network, n_filter=n_filter, name = name + 'conv2d')
        if with_bn == True:
            network = tl.layers.BatchNormLayer(network, name = name +'batchnorm_layer')
        network = tf.nn.relu(network.outputs)
        network = tl.layers.InputLayer(network, name = name + 'aligned_layer_2')
        network = tl.layers.MaxPool2d(network, name = name + 'maxpool')
        return network

    def build(self, ph, reuse = False):
        """
            Get network
        """
        with tf.variable_scope('discriminator', reuse = reuse):
            tl.layers.set_name_reuse(reuse)
            if type(ph) == tf.Tensor:
                self.network = tl.layers.InputLayer(ph)
            else:
                self.network = ph
            self.network = self.add_layer(self.network, n_filter = self.base_filter, with_bn = False, name = '1')
            self.network = self.add_layer(self.network, n_filter = self.base_filter * (2 ** 1), name = '2')
            self.network = self.add_layer(self.network, n_filter = self.base_filter * (2 ** 2), name = '3')
            self.network = self.add_layer(self.network, n_filter = self.base_filter * (2 ** 3), name = '4')
            self.logits = tl.layers.FlattenLayer(self.network)
            self.network = tl.layers.DenseLayer(self.logits, n_units = 1, act = tf.nn.sigmoid)
            return self.network

class Discriminator_small(object):
    """
        The small discriminator of Generative Auxiliary Strategy

        n_filter = 32: 713MB
        n_filter = 16: 457MB
    """
    
    def __init__(self, base_filter=32):
        self.base_filter = base_filter

    def add_layer(self, network, n_filter, with_bn = True, width_multiplier = 4, name = "layer"):
        """
            From the normal block of DCGAN
        """
        network = inception_conv2d_with_pwconv(network, n_filter, width_multiplier, sc = name + 'conv_ds')
        if with_bn == True:
            network = tl.layers.BatchNormLayer(network, name = name +'batchnorm_layer')
        network = tf.nn.relu(network.outputs)
        network = tl.layers.InputLayer(network, name = name + 'aligned_layer_2')
        network = tl.layers.MaxPool2d(network, name = name + 'maxpool')
        return network

    def build(self, ph, reuse = False):
        """
            Get network
        """
        
        with tf.variable_scope('discriminator', reuse = reuse):
            tl.layers.set_name_reuse(reuse)
            if type(ph) == tf.Tensor:
                self.network = tl.layers.InputLayer(ph)
            else:
                self.network = ph
            self.network = self.add_layer(self.network, n_filter = self.base_filter, with_bn = False, name = '1')
            self.network = self.add_layer(self.network, n_filter = self.base_filter * (2 ** 1), name = '2')
            self.network = self.add_layer(self.network, n_filter = self.base_filter * (2 ** 2), name = '3')
            self.network = self.add_layer(self.network, n_filter = self.base_filter * (2 ** 3), name = '4')
            self.logits = tl.layers.FlattenLayer(self.network)
            self.network = tl.layers.DenseLayer(self.logits, n_units = 1, act = tf.nn.sigmoid)
            return self.network

class Discriminator_inception(object):
    """
        The small discriminator of Generative Auxiliary Strategy

        n_filter = 32: 713MB
        n_filter = 16: 457MB
    """
    
    def __init__(self, base_filter=32):
        self.base_filter = base_filter

    def add_layer(self, network, n_filter, with_bn = True, width_multiplier = 4, name = "layer"):
        """
            From the normal block of DCGAN
        """
        network = inception_conv2d(network, n_filter, width_multiplier, sc = name + 'conv_ds')
        if with_bn == True:
            network = tl.layers.BatchNormLayer(network, name = name +'batchnorm_layer')
        network = tf.nn.relu(network.outputs)
        network = tl.layers.InputLayer(network, name = name + 'aligned_layer_2')
        network = tl.layers.MaxPool2d(network, name = name + 'maxpool')
        return network

    def build(self, ph, reuse = False):
        """
            Get network
        """
        
        with tf.variable_scope('discriminator', reuse = reuse):
            tl.layers.set_name_reuse(reuse)
            if type(ph) == tf.Tensor:
                self.network = tl.layers.InputLayer(ph)
            else:
                self.network = ph
            self.network = self.add_layer(self.network, n_filter = self.base_filter, with_bn = False, name = '1')
            self.network = self.add_layer(self.network, n_filter = self.base_filter * (2 ** 1), name = '2')
            self.network = self.add_layer(self.network, n_filter = self.base_filter * (2 ** 2), name = '3')
            self.network = self.add_layer(self.network, n_filter = self.base_filter * (2 ** 3), name = '4')
            self.logits = tl.layers.FlattenLayer(self.network)
            self.network = tl.layers.DenseLayer(self.logits, n_units = 1, act = tf.nn.sigmoid)
            return self.network

class Discriminator_dense(object):
    """
        The small discriminator of Generative Auxiliary Strategy (DenseNet version)

        n_filter = 32: 713MB
        n_filter = 16: 457MB
    """
    def __init__(self):
        pass

    def add_layer(self, name, l, growthRate = 16, width_multiplier = 1):
        shape = l.outputs.get_shape().as_list()
        in_channel = shape[3]
        with tf.variable_scope(name) as scope:
            c = tl.layers.BatchNormLayer(l, act = tf.nn.relu, name = name +'_layer_batchnorm_layer')
            c = inception_conv2d(c, growthRate, width_multiplier, sc = name + '_layer_incep_conv')
            l = tl.layers.ConcatLayer([c, l], concat_dim = 3, name = name + "_layer_concat_layer")
            return l

    def add_transition(self, name, l, width_multiplier = 1):
        """
            * Input type is layer
            * Output type is layer
        """
        shape = l.outputs.get_shape().as_list()
        in_channel = shape[3]
        with tf.variable_scope(name) as scope:
            l = tl.layers.BatchNormLayer(l, act = tf.nn.relu, name = name +'_transition_batchnorm_layer')
            l = inception_conv2d(l, 4, width_multiplier, sc = name + '_transition_incep_conv')
            l = tf.layers.average_pooling2d(l.outputs, 2, 2)
            l = tl.layers.InputLayer(l, name = name + "_transition_avgpool_aligned")
            return l

    def add_block(self, network, n_filter, name = "block"):
        """
            * Input type is layer
            * Output type is layer
        """
        with tf.variable_scope(name) as scope:
            l = network
            for i in range(3):
                l = self.add_layer('dense_layer.{}'.format(i), l)
            l = self.add_transition('transition1', l)
            return l

    def build(self, ph, base_filter=16, reuse = False):
        """
            Get network
        """
        with tf.variable_scope('discriminator', reuse = reuse):
            tl.layers.set_name_reuse(reuse)
            if type(ph) == tf.Tensor:
                self.network = tl.layers.InputLayer(ph)
            else:
                self.network = ph
            self.network = inception_conv2d_with_pwconv(self.network, base_filter, 1, sc = '1')
            self.network = self.add_block(self.network, n_filter = base_filter, name = '2')
            self.network = self.add_block(self.network, n_filter = base_filter, name = '3')
            self.network = self.add_block(self.network, n_filter = base_filter, name = '4')
            self.network = tl.layers.FlattenLayer(self.network)
            self.network = tl.layers.DenseLayer(self.network, n_units = 1, act = tf.nn.sigmoid)
            return self.network

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.device('/device:GPU:0'):
            # Define network
            ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
            model = Discriminator()
            model.build(ph)
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.1
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                print(type(model))
                while True:
                    sess.run(model.logits, feed_dict={
                        ph: np.random.random([32, 28, 28, 1])
                    })
