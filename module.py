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

    def depthwise_separable_conv(self, inputs, num_pwc_filters, width_multiplier, sc, downsample=False):
        """ 
            Helper function to build the depth-wise separable convolution layer.
        """
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1
        depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=_stride, depth_multiplier=1, kernel_size=[3, 3], scope=sc+'/depthwise_conv')
        bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn, num_pwc_filters, kernel_size=[1, 1], scope=sc+'/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
        return bn

    def _inception_conv(self, inputs, num_pwc_filters, width_multiplier, sc, downsample=False):
        """
            alpha is the shrink scale
        """
        branch1 = tl.layers.Conv2d(inputs, n_filter = num_pwc_filters // 2, filter_size=(1, 1), strides=(1, 1), name = sc + '_inception_branch1')
        branch2 = tl.layers.Conv2d(inputs, n_filter = num_pwc_filters // 2, filter_size=(1, 1), strides=(1, 1), name = sc + '_inception_branch2')
        branch2 = self.depthwise_separable_conv(branch2.outputs, num_pwc_filters // 2, width_multiplier, sc, downsample)
        branch2 = tl.layers.InputLayer(branch2, name = sc + 'inception_aligned_layer')    
        network = tl.layers.ConcatLayer([branch1, branch2], concat_dim = -1, name = sc + '_inception_concat')
        return network
        
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