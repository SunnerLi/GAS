from module import Discriminator, Discriminator_Dense, Discriminator_Large
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
        # true_logits = tf.reduce_mean(true_logits)
        # fake_logits = tf.reduce_mean(fake_logits)
        # self.dis_loss = tf.square(true_logits - 1) + tf.square(fake_logits)
        # self.gen_loss = tf.square(fake_logits - 1)
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
        self.discriminator = Discriminator_Large()

class LSGAN_origin(LSGAN):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()

class LSGAN_dense(LSGAN):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator_Dense()

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.device('/device:GPU:0'):
            # Define network
            z_ph = tf.placeholder(tf.float32, shape=[None, 100])
            img_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
            model = LSGAN_origin()
            model.build(z_ph, img_ph)
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.1
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                print(type(model.discriminator))
                while True:
                    sess.run(model.gen_train_op, feed_dict={
                        z_ph: np.random.random([32, 100]),
                        img_ph: np.random.random([32, 28, 28, 1])
                    })
