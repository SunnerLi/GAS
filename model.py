from module import Discriminator_dense, Discriminator_large, Discriminator_small, Discriminator_inception
import tensorlayer as tl
import tensorflow as tf
import numpy as np

class Generator(object):
    def __init__(self):
        pass

    def dense(self, x, n_unit):
        W = tf.Variable(tf.truncated_normal([int(x.get_shape()[-1]), n_unit]))
        b = tf.Variable(tf.truncated_normal([n_unit]))
        return tf.add(tf.matmul(x, W), b)

    def build(self, ph, name = "generator"):
        """
            The structure of generator is as the same as the original LSGAN
        """
        initializer = tf.truncated_normal_initializer(stddev=0.02)
        with tf.variable_scope(name):
            # conv1
            fc1 = tf.contrib.layers.fully_connected(inputs=ph, num_outputs=7*7*128, activation_fn=tf.nn.relu, \
                                                    normalizer_fn=tf.contrib.layers.batch_norm,\
                                                    weights_initializer=initializer,scope="g_fc1")
            fc1 = tf.reshape(fc1, shape=[32, 7, 7, 128])

            # conv2
            conv1 = tf.contrib.layers.conv2d(fc1, num_outputs=4*64, kernel_size=5, stride=1, padding="SAME",    \
                                            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                            weights_initializer=initializer,scope="g_conv1")
            conv1 = tf.reshape(conv1, shape=[32,14,14,64])

            # conv3
            conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=4*32, kernel_size=5, stride=1, padding="SAME", \
                                            activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm, \
                                            weights_initializer=initializer,scope="g_conv2")

            conv2 = tf.reshape(conv2, shape=[32,28,28,32])

            # conv4
            conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=1, kernel_size=5, stride=1, padding="SAME", \
                                            activation_fn=tf.nn.tanh,scope="g_conv3")
            return conv3

class LSGAN(object):
    def __init__(self):
        pass

    def build(self, z_ph, img_ph):
        self.gen_imgs = self.generator.build(z_ph)
        true_logits = self.discriminator.build(img_ph).outputs
        fake_logits = self.discriminator.build(self.gen_imgs, reuse=True).outputs

        # Define loss
        self.dis_loss = tf.reduce_sum(tf.square(true_logits - 1) + tf.square(fake_logits)) / 2
        self.gen_loss = tf.reduce_sum(tf.square(fake_logits - 1)) / 2

        # Define optimizer
        # gen_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        # dis_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        gen_variable = [v for v in tf.trainable_variables() if 'generator' in v.name]
        dis_variable = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

        gen_optimizer = tf.train.RMSPropOptimizer(0.001)
        dis_optimizer = tf.train.RMSPropOptimizer(0.001)
        gen_gradient = gen_optimizer.compute_gradients(self.gen_loss, gen_variable)
        dis_gradient = dis_optimizer.compute_gradients(self.dis_loss, dis_variable)
        self.gen_train_op = gen_optimizer.apply_gradients(gen_gradient)
        self.dis_train_op = dis_optimizer.apply_gradients(dis_gradient)

class LSGAN_large(LSGAN):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator_large()

class LSGAN_small(LSGAN):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator_small()

class LSGAN_inception_32(LSGAN):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator_inception(base_filter = 32)

class LSGAN_dense(LSGAN):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator_dense()

class LSGAN_inception_16(LSGAN):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator_inception(base_filter = 16)

class LSGAN_inception_8(LSGAN):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator_inception(base_filter = 8)