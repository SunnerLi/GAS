from matplotlib import pyplot as plt
from utils import get_img, save_img
import numpy as np
import pickle
import os

# File & folder name 
gen_pickle_name_list = ['gen_loss_large', 'gen_loss_small', 'gen_loss_in32', 
    'gen_loss_dense', 'gen_loss_in16', 'gen_loss_in8']
dis_pickle_name_list = ['dis_loss_large', 'dis_loss_small', 'dis_loss_in32',
    'dis_loss_dense', 'dis_loss_in16', 'dis_loss_in8']
output_folder = './output/'
img_folder_list = ['lsgan_large_img', 'lsgan_small_img', 'lsgan_inception_32_img',
    'lsgan_dense_img', 'lsgan_inception_16_img', 'lsgan_inception_8_img']

# Label
labels = ['Large', 'Small', 'Inception_32', 'Dense', 'Inception_16', 'Inception_8']

# object
gen_loss_list = []
dis_loss_list = []

if __name__ == '__main__':
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # -------------------------------------------------------------------------------
    # Loss file
    # -------------------------------------------------------------------------------
    if len(gen_pickle_name_list) != len(dis_pickle_name_list):
        print("the number of file name isn't equal...")
        exit()
    for i in range(len(gen_pickle_name_list)):
        with open(gen_pickle_name_list[i] + '.pickle', 'rb') as f:
            gen_loss_list.append(pickle.load(f))
        with open(dis_pickle_name_list[i] + '.pickle', 'rb') as f:
            dis_loss_list.append(pickle.load(f))

    # -------------------------------------------------------------------------------
    # Draw discriminator loss
    # -------------------------------------------------------------------------------
    plt.figure(0)
    for i in range(len(labels)):
        idx = np.asarray(range(len(dis_loss_list[i]))) * 200
        plt.plot(idx, dis_loss_list[i], '-o', label=labels[i])
    plt.legend()
    plt.savefig(output_folder + 'discriminator_loss.png')
    plt.gca().clear()

    # -------------------------------------------------------------------------------
    # Draw generator loss
    # -------------------------------------------------------------------------------
    plt.figure(0)
    for i in range(len(labels)):
        idx = np.asarray(range(len(gen_loss_list[i]))) * 200
        plt.plot(idx, gen_loss_list[i], '-o', label=labels[i])
    plt.legend()
    plt.savefig(output_folder + 'generator_loss.png')
    plt.gca().clear()

    # -------------------------------------------------------------------------------
    # Draw (3 * 2) plot
    # -------------------------------------------------------------------------------
    # Image order:
    #
    #   Large        Small       Inception_32
    #   Dense     Inception_16    Inception_8
    # 
    img_name_list = os.listdir(img_folder_list[0])
    num_img = len(img_name_list)
    for i in range(num_img):
        result_img = None
        for j in range(2):
            row = None
            for k in range(3):
                img = get_img(img_folder_list[j*3+k] + '/' + img_name_list[i])
                if row is None:
                    row = img
                else:
                    row = np.concatenate((row, img), axis=1)
            if result_img is None:
                result_img = row
            else:
                result_img = np.concatenate((result_img, row), axis=0)
        save_img(output_folder + img_name_list[i], result_img)
