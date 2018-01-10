from data_helper import ImageHandler, generateNoice
from record import saveGeneratedBatch
from model import *
import tensorflow as tf
import numpy as np
import pickle

# Training constant
evaluate_period = 250
epochs = 1000
batch_size = 32

def train(model, gen_loss_file_name, dis_loss_file_name, img_save_folder):
    # Define image handler & object
    handler = ImageHandler(dataset_name = 'mnist', resize_length = 28)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    gen_loss_list = []
    dis_loss_list = []
    
    # Define network
    z_placeholder = tf.placeholder(tf.float32, [None, 100])
    img_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1])
            
    model.build(z_placeholder, img_placeholder)
    with tf.Session(config=config) as sess:
        # Train and evaluate
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            sess.run([model.dis_train_op], feed_dict={
                z_placeholder: generateNoice(batch_size, 100).astype(np.float32),
                img_placeholder: handler.getBatchImage()
            })
            for j in range(4):  # Should update generator to avoid mode collapse
                sess.run([model.gen_train_op], feed_dict={
                    z_placeholder: generateNoice(batch_size, 100).astype(np.float32),
                })
            if i % evaluate_period == 0:
                feed_dict = {
                    z_placeholder: generateNoice(batch_size, 100).astype(np.float32),
                    img_placeholder: handler.getBatchImage()
                }
                gen_loss, dis_loss = sess.run([model.gen_loss, model.dis_loss], feed_dict=feed_dict)
                gen_loss_list.append(gen_loss)
                dis_loss_list.append(dis_loss)
                print('Iter: ', i, '\tgenerator loss: ', gen_loss, '\tdiscriminator loss: ', dis_loss)
                gen_image = sess.run([model.gen_imgs], feed_dict=feed_dict)
                saveGeneratedBatch(gen_image[0], 8, i, output_dir=img_save_folder)
    # Store
    with open(gen_loss_file_name, 'wb') as f:
        pickle.dump(gen_loss_list, f)
    with open(dis_loss_file_name, 'wb') as f:
        pickle.dump(dis_loss_list, f)

if __name__ == '__main__':
    # ----------------------------------------------------------------------------
    # Test on original LSGAN
    # ----------------------------------------------------------------------------
    with tf.Graph().as_default():
        with tf.device('/GPU:0'):
            model = LSGAN_large()
            train(model, 'gen_loss_large.pickle', 'dis_loss_large.pickle', './lsgan_large_img')
            tl.layers.clear_layers_name()

    # ----------------------------------------------------------------------------
    # Test on GAS (inception + point-wise conv)
    # ----------------------------------------------------------------------------
    with tf.Graph().as_default():
        with tf.device('/GPU:0'):
            model = LSGAN_small()
            train(model, 'gen_loss_small.pickle', 'dis_loss_small.pickle', './lsgan_small_img')
            tl.layers.clear_layers_name()

    # ----------------------------------------------------------------------------
    # Test on GAS (inception only, kernel = 32)
    # ----------------------------------------------------------------------------
    with tf.Graph().as_default():
        with tf.device('/GPU:0'):
            model = LSGAN_inception_32()
            train(model, 'gen_loss_in32.pickle', 'dis_loss_in32.pickle', './lsgan_inception_32_img')
            tl.layers.clear_layers_name()

    # ----------------------------------------------------------------------------
    # Test on GAS (dense + inception + point-wise conv)
    # ----------------------------------------------------------------------------
    with tf.Graph().as_default():
        with tf.device('/GPU:0'):
            model = LSGAN_dense()
            train(model, 'gen_loss_dense.pickle', 'dis_loss_dense.pickle', './lsgan_dense_img')
            tl.layers.clear_layers_name()

    # ----------------------------------------------------------------------------
    # Test on GAS (inception only, kernel = 16)
    # ----------------------------------------------------------------------------
    with tf.Graph().as_default():
        with tf.device('/GPU:0'):
            model = LSGAN_inception_16()
            train(model, 'gen_loss_in16.pickle', 'dis_loss_in16.pickle', './lsgan_inception_16_img')
            tl.layers.clear_layers_name()

    # ----------------------------------------------------------------------------
    # Test on GAS (inception only, kernel = 8)
    # ----------------------------------------------------------------------------
    with tf.Graph().as_default():
        with tf.device('/GPU:0'):
            model = LSGAN_inception_8()
            train(model, 'gen_loss_in8.pickle', 'dis_loss_in8.pickle', './lsgan_inception_8_img')
            tl.layers.clear_layers_name()
