"""
Modified from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
"""


from six.moves import urllib
from tqdm import tqdm
import subprocess
import requests
import shutil
import json
import gzip
import sys
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={ 'id': id }, stream=True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32*1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
            unit='B', unit_scale=True, desc=destination):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_celeb_a(dirpath):
    data_dir = 'celebA'
    filename, drive_id  = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    save_path = os.path.join(dirpath, filename)
    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

    zip_dir = ''
    os.system('unzip ' + dirpath + '/*.zip')
    os.remove(save_path)
    os.rename(filename[:-4], os.path.join(dirpath, data_dir))

def _download_lsun(out_dir, category, set_name, tag):
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
        '&category={category}&set={set_name}'.format(**locals())
    print(url)
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = os.path.join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)

def download_lsun(dirpath):
    data_dir = os.path.join(dirpath, 'lsun')
    if os.path.exists(data_dir):
        print('Found LSUN - skip')
        return
    else:
        os.mkdir(data_dir)

    tag = 'latest'
    categories = ['church_outdoor']
    for category in categories:
        _download_lsun(data_dir, category, 'train', tag)
        _download_lsun(data_dir, category, 'val', tag)

def download_mnist(dirpath):
    data_dir = os.path.join(dirpath, 'mnist')
    if os.path.exists(data_dir):
        print('Found MNIST - skip')
        return
    else:
        os.mkdir(data_dir)
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz',
                  'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (url_base+file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir,file_name)
        cmd = ['curl', url, '-o', out_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', out_path]
        print('Decompressing ', file_name)
        subprocess.call(cmd)