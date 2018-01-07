from module import Discriminator, Discriminator_Dense, Discriminator_Large
import tensorlayer as tl
import tensorflow as tf
import numpy as np

class Generator(object):
    def __init__(self):
        pass

    def lrelu(self, x, th=0.2):
        return tf.maximum(th * x, x) 

    def dense(self, x, n_unit):
        W = tf.Variable(tf.truncated_normal([int(x.get_shape()[-1]), n_unit]))
        b = tf.Variable(tf.truncated_normal([n_unit]))
        return tf.add(tf.matmul(x, W), b)

    def build(self, ph, name = "generator"):
        

        with tf.variable_scope(name) as scope:
            # 1st hidden layer
            dense1 = self.dense(ph, 7*7*128)
            reshape1 = tf.reshape(dense1, [tf.shape(dense1)[0], 7, 7, 128])
            lrelu1 = self.lrelu(tf.layers.batch_normalization(reshape1), 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(lrelu1, 64, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = self.lrelu(tf.layers.batch_normalization(conv2), 0.2)

            # output layer
            conv3 = tf.layers.conv2d_transpose(lrelu2, 1, [4, 4], strides=(2, 2), padding='same')
            o = tf.nn.tanh(conv3)

            return o

class LSGAN(object):
    def __init__(self):
        pass

    def build(self, z_ph, img_ph):
        self.gen_imgs = self.generator.build(z_ph)
        true_logits = self.discriminator.build(img_ph).outputs
        fake_logits = self.discriminator.build(self.gen_imgs, reuse=True).outputs

        # Define loss
        self.dis_loss = tf.reduce_sum(tf.square(true_logits - 1) + tf.square(fake_logits))
        self.gen_loss = tf.reduce_sum(tf.square(fake_logits - 1))

        # Define optimizer
        # gen_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        # dis_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        gen_variable = [v for v in tf.trainable_variables() if 'generator' in v.name]
        dis_variable = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

        gen_optimizer = tf.train.AdamOptimizer()
        dis_optimizer = tf.train.AdamOptimizer()
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
