from module import Discriminator, Discriminator_Dense
import tensorlayer as tl
import tensorflow as tf
import numpy as np

class Generator(object):
    def __init__(self, ph):
        self.ph = ph

    def build(self, name = "generator"):
        def lrelu(x, th=0.2):
            return tf.maximum(th * x, x)

        with tf.variable_scope(name) as scope:
            # 1st hidden layer
            dense1 = tf.layers.dense(self.ph, units = 7*7*128)
            reshape1 = tf.reshape(dense1, [tf.shape(dense1)[0], 7, 7, 128])
            lrelu1 = lrelu(tf.layers.batch_normalization(reshape1), 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(lrelu1, 64, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = lrelu(tf.layers.batch_normalization(conv2), 0.2)

            # output layer
            conv3 = tf.layers.conv2d_transpose(lrelu2, 1, [4, 4], strides=(2, 2), padding='same')
            o = tf.nn.tanh(conv3)

            return o


class LSGAN(object):
    def __init__(self):
        self.z_ph = None
        self.img_ph = None
        self.generator = None
        self.true_discriminator = None
        self.fake_discriminator = None

    def build(self):
        self.gen_imgs = self.generator.build()
        true_logits = self.true_discriminator.build().outputs
        fake_logits = self.fake_discriminator.build(reuse=True).outputs

        # Define loss
        self.dis_loss = tf.reduce_sum(tf.square(true_logits - 1) + tf.square(fake_logits))
        self.gen_loss = tf.reduce_sum(tf.square(fake_logits - 1))

        # Define optimizer
        gen_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        gen_variable = [v for v in tf.trainable_variables() if 'generator' in v.name]
        dis_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        gen_optimizer = tf.train.AdamOptimizer()
        dis_optimizer = tf.train.AdamOptimizer()
        gen_gradient = gen_optimizer.compute_gradients(self.gen_loss, gen_variable)
        dis_gradient = dis_optimizer.compute_gradients(self.dis_loss, dis_variable)
        self.gen_train_op = gen_optimizer.apply_gradients(gen_gradient)
        self.dis_train_op = dis_optimizer.apply_gradients(dis_gradient)

class LSGAN_origin(LSGAN):
    def __init__(self, z_ph, img_ph):
        self.z_ph = z_ph
        self.img_ph = img_ph
        self.generator = Generator(self.z_ph)
        self.true_discriminator = Discriminator(self.img_ph)
        self.fake_discriminator = Discriminator(self.img_ph)

class LSGAN_dense(LSGAN):
    def __init__(self, z_ph, img_ph):
        self.z_ph = z_ph
        self.img_ph = img_ph
        self.generator = Generator(self.z_ph)
        self.discriminator = Discriminator_Dense(self.img_ph)


if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.device('/device:GPU:0'):
            # Define network
            z_ph = tf.placeholder(tf.float32, shape=[None, 1, 1, 100])
            img_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
            model = LSGAN_origin(z_ph, img_ph)

            model.build()
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.1
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                while True:
                    sess.run(model.gen_imgs, feed_dict={
                        ph: np.random.random([32, 100])
                    })