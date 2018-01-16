from utils import save_img
import scipy.misc, numpy as np, os, sys
import numpy as np
import subprocess
import os

record_file_name = "gpu_usage.txt"

def save_img(out_path, img):
    """
        Save the image
        Arg:    out_path    - The image path that you want to store
                img         - The ndarray image object
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def get_img(src, img_size=False):
    """
        Get the image ndarray from specific path
        Arg:    src         - The path of image you want to get
                img_size    - The size you want to align
        Ret:    The image object
    """
    img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img