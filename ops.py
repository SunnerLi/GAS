import tensorflow.contrib.slim as slim
import tensorlayer as tl
import tensorflow as tf

def depthwise_separable_conv2d(inputs, num_pwc_filters, width_multiplier, sc, downsample=False):
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

def inception_conv2d(inputs, num_pwc_filters, width_multiplier, sc, downsample=False):
    branch1 = tl.layers.Conv2d(inputs, n_filter = num_pwc_filters // 2, filter_size=(1, 1), strides=(1, 1), name = sc + '_inception_branch1')
    branch2 = tl.layers.Conv2d(inputs, n_filter = num_pwc_filters // 2, filter_size=(1, 1), strides=(1, 1), name = sc + '_inception_branch2_1')
    branch2 = tl.layers.Conv2d(inputs, n_filter = num_pwc_filters // 2, filter_size=(3, 3), strides=(1, 1), name = sc + '_inception_branch2_2')
    network = tl.layers.ConcatLayer([branch1, branch2], concat_dim = -1, name = sc + '_inception_concat')
    return network

def inception_conv2d_with_pwconv(inputs, num_pwc_filters, width_multiplier, sc, downsample=False):
    branch1 = tl.layers.Conv2d(inputs, n_filter = num_pwc_filters // 2, filter_size=(1, 1), strides=(1, 1), name = sc + '_inception_branch1')
    branch2 = tl.layers.Conv2d(inputs, n_filter = num_pwc_filters // 2, filter_size=(1, 1), strides=(1, 1), name = sc + '_inception_branch2_1')
    branch2 = depthwise_separable_conv2d(branch2.outputs, num_pwc_filters // 2, width_multiplier, sc, downsample)
    branch2 = tl.layers.InputLayer(branch2, name = sc + 'inception_aligned_layer')    
    network = tl.layers.ConcatLayer([branch1, branch2], concat_dim = -1, name = sc + '_inception_concat')
    return network