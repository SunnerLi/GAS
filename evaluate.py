from model import LSGAN_large, LSGAN_origin, LSGAN_dense
from data_helper import ImageHandler, generateNoice
from record import saveGeneratedBatch
import tensorflow as tf
import numpy as np
import pickle

# Training constant
evaluate_period = 100
epochs = 1000
batch_size = 32

# Loss file
large_gen_loss_file = 'gen_loss_large.pickle'
large_dis_loss_file = 'dis_loss_large.pickle'

if __name__ == '__main__':
    # Define image handler
    handler = ImageHandler(dataset_name = 'mnist', resize_length = 28)

    # Define config object
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    
    # ----------------------------------------------------------------------------
    # Test on original LSGAN
    # ----------------------------------------------------------------------------
    gen_loss_list = []
    dis_loss_list = []
    with tf.Graph().as_default():
        with tf.device('/GPU:0'):
            # Define network
            z_placeholder = tf.placeholder(tf.float32, [None, 100])
            img_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1])
            model = LSGAN_large()
            model.build(z_placeholder, img_placeholder)
            with tf.Session(config=config) as sess:
                # Train and evaluate
                sess.run(tf.global_variables_initializer())
                for i in range(epochs):
                    feed_dict = {
                        z_placeholder: generateNoice(batch_size, 100),
                        img_placeholder: handler.getBatchImage()
                    }
                    sess.run([model.dis_train_op, model.gen_train_op], feed_dict=feed_dict)
                    if i % evaluate_period == 0:
                        gen_loss, dis_loss = sess.run([model.gen_loss, model.dis_loss], feed_dict=feed_dict)
                        gen_loss_list.append(gen_loss)
                        dis_loss_list.append(dis_loss)
                        print('Iter: ', i, '\tgenerator loss: ', gen_loss, '\tdiscriminator loss: ', dis_loss)
                        gen_image = sess.run([model.gen_imgs], feed_dict=feed_dict)
                        saveGeneratedBatch(gen_image[0], 8, i)
        # Store
        with open(large_gen_loss_file, 'wb') as f:
            pickle.dump(gen_loss_list, f)
        with open(large_dis_loss_file, 'wb') as f:
            pickle.dump(dis_loss_list, f)
