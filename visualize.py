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
    """
    if len(gen_pickle_name_list) != len(dis_pickle_name_list):
        print("the number of file name isn't equal...")
        exit()
    for i in range(len(gen_pickle_name_list)):
        with open(gen_pickle_name_list[i] + '.pickle', 'rb') as f:
            gen_loss_list.append(pickle.load(f))
        with open(dis_pickle_name_list[i] + '.pickle', 'rb') as f:
            dis_loss_list.append(pickle.load(f))
    #print dis_loss_list
    """
    gen_loss_list = [[3.9850731, 5.2694607, 3.9990358, 2.678813, 4.9761009, 10.643662, 7.9860086, 7.7833447, 14.692568, 12.315845], [8.3235435, 2.3021585e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3.9995375, 5.735745, 8.3344269, 3.9079924, 2.8897741, 6.2019825, 6.4326687, 2.0114141, 4.95469, 15.746188], [4.0000248, 3.99719, 3.9979784, 3.9979918, 3.9979908, 3.9979928, 3.9979947, 3.9979937, 3.9979975, 3.9979994], [4.0005827, 4.2677031, 3.083617, 4.5677752, 4.7255445, 2.5669613, 5.9051499, 2.4773955, 3.2656775, 9.8660784], [4.0006051, 4.7379575, 3.9286952, 4.0103893, 4.2286038, 4.1773257, 4.0260468, 4.1013517, 3.9127953, 2.8878026]]
    dis_loss_list = [[7.9975076, 5.3601418, 7.5370717, 7.2687149, 4.9018345, 3.5204906, 4.2886934, 4.7916803, 8.3172426, 3.9153376], [15.024375, 15.999998, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0], [7.9964557, 6.565815, 7.2593861, 5.9563131, 6.8626628, 3.69504, 3.4604151, 8.5461082, 5.0192304, 4.3967938], [8.0001202, 8.000001, 8.000001, 8.000001, 8.000001, 8.000001, 8.000001, 8.000001, 8.000001, 8.000001], [7.9998865, 7.7692046, 8.1805267, 8.0408697, 7.9545918, 7.7954245, 7.1426668, 7.2224016, 6.2955341, 5.3442998], [7.9994273, 7.6801462, 8.0139771, 8.0394135, 8.096734, 8.0504322, 7.8956299, 8.0275307, 8.2115822, 8.2166672]]

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
