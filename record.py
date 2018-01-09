from utils import save_img
import numpy as np
import subprocess
import os

record_file_name = "gpu_usage.txt"

def storeTensor(tensor, store_path = "?"):
    """
        This function assume the size of tensor is [32, x, x, 1]
    """
    result_img = None
    for i in range(8):
        row = None
        for j in range(4):
            if row is None:
                row = tensor[i*8+j]
            else:
                row = np.concatenate((row, tensor[i*8+j]), axis=2)
        if result_img is None:
            result_img = row
        else:
            result_img = np.concatenate((result_img, row), axis=1)
    save_img(store_path, result_img)

def saveGeneratedBatch(tensor, num_per_row, idx, output_dir='output'):
    # Check output dir
    if not os.path.exists(output_dir):
        os.system('mkdir -p ' + output_dir)

    # Ensure the num_per_row is the factor of batch size
    if (np.shape(tensor)[0] // num_per_row) * num_per_row != np.shape(tensor)[0]:
        num_per_row = 1
        for i in range(1, np.shape(tensor)[0]):
            if (np.shape(tensor)[0] // i) * i == np.shape(tensor)[0]:
                num_per_row = i

    # Save grids
    res = None
    for i in range(np.shape(tensor)[0] // num_per_row):
        res_row = None
        for j in range(num_per_row):
            if j == 0:
                res_row = tensor[j]
            else:
                res_row = np.concatenate((res_row, tensor[i*num_per_row+j]), axis=1)
        if i == 0:
            res = res_row
        else:
            res = np.concatenate((res, res_row), axis=0)
    if len(np.shape(res)) == 3:     # Gray image tensor should reduce as rank-2
        res = np.squeeze(res, axis=-1)
    res = (res * 255).astype(np.uint8)
    save_img(output_dir + '/' +str(idx) + '.png', res)


def recordGPUUsage():
    """
        Dump the nvidia-smi information into file
        * Notice: the imformation will be flashed while you call this function again
    """
    global record_file_name

    # Write information into temp file
    temp_file_name = './.temp.txt'
    nvidia_args = ['nvidia-smi', '-f', temp_file_name]
    cmd = " ".join(nvidia_args)
    subprocess.call(cmd, shell=True)   

    # Concatenate information into specific file
    with open(record_file_name, 'a') as f:
        new_symbol = '\n' + '-' * 100 + '\n\n'
        info = open(temp_file_name, 'r').readlines()
        f.write(new_symbol)
        f.writelines(info)

    # Kill dummy file
    os.remove(temp_file_name)