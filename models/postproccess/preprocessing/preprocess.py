#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description: Pretreatment integration process , Note that there are two modes to choose .
"""
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import os
import cv2
import numpy as np
import glob
from preprocessing.add_noise import add_img_noise
from preprocessing.nor import normalize
from preprocessing.gabor import gabor_enhance
from preprocessing.thining import make_thin


def preprocessing_imnoise(img_dir, out_dir='./train_set/', noise_type=(('small', 1), ('large', 1))):
    """
    Image preprocess: normalization, imnoise, Gabor enhancement, thinning
    :param img_dir:
    :param out_dir: output images will be divided into different subdirectories according to their types
    :param noise_type: each tuple defines a noise given (size, num)
    :return: 
    """
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    img_all = glob.glob(img_dir + '*_O*')
    subtype = img_dir.split('/')[-2]
    img_out_dir = out_dir + subtype + '/'

    # Add border to original image, name output image as "xxx_v1.png"
    broaden_list = []
    for img_path in img_all:
        prefix = os.path.splitext(os.path.basename(img_path))[0]
        file_broad = '{dir}{fname}_v1.png'.format(dir=img_out_dir, fname=prefix)
        broaden_list.append(file_broad)
        img = cv2.imread(img_path, 0)
        border = np.full((10, img.shape[1]), 200)
        img_broad = np.concatenate((border, img, border))
        cv2.imwrite(file_broad, img_broad)

    # Normalization
    img_normal = []
    for img_path in broaden_list:
        img_normal.append(normalize(img_path, img_out_dir))
    print('Normalization --- Done!')

    # Gabor enhancement & thinning for initial normalized images (without noise)
    for img in img_normal:
        img_enhance = gabor_enhance(img, img_out_dir)  # Gabor enhancement
        make_thin(img_enhance)  # Extract fingerprint skeleton

    # Imnoise
    img_noise = []
    for img_path in img_normal:
        for idx in noise_type:
            img_noise.append(add_img_noise(img_path, img_out_dir, mode=idx[0], num=idx[1]))
    print('Imnoise --- Done!')

    # Gabor enhancement & thinning for imnoised images
    for img in img_noise:
        img_enhance = gabor_enhance(img, img_out_dir)  # Gabor enhancement
        make_thin(img_enhance)  # Extract fingerprint skeleton

    print('Done!')


def preprocess_general(img_path):
    """
    General preprocessing workflow
    :param img_path:
    :param out_dir:
    :return:
    """
    img = cv2.imread(img_path, 0)
    border = np.full((10, img.shape[1]), 200)
    img_broad = np.concatenate((border, img, border))
    img_broad = img_broad.astype("uint8")
    img_normal = normalize(img_broad)
    img_enhance = gabor_enhance(img_normal)
    thin_img = make_thin(img_enhance)
    return img_broad, img_normal, img_enhance, thin_img


if __name__ == '__main__':
    preprocess_general('/home/naserwin/hamze/dataset_tmp/MOLF/fingerprints_eval/sample_test_bg_2/1.png')