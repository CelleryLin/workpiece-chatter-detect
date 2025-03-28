import numpy as np
import cv2
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit

@jit(nopython=True)
def adaptive_binarize(img, ksize=(100, 100), th_shift=0):
    H, W = img.shape

    # split image into 100x100 blocks
    image_new = np.zeros((H, W))
    for i in range(0, H, ksize[0]):
        for j in range(0, W, ksize[1]):
            th = get_best_threshold(img[i:i+ksize[0], j:j+ksize[1]])
            th += th_shift
            image_new[i:i+ksize[0], j:j+ksize[1]] = binarize(img[i:i+ksize[0], j:j+ksize[1]], th)

    return image_new

@jit(nopython=True)
def binarize(img, threshold):
    img[img < threshold] = 0
    img[img >= threshold] = 1
    return img

@jit(nopython=True)
def get_best_threshold(img):
    resolution = 100
    img_px = img.reshape(-1)
    hist = np.zeros(resolution)
    hist_tmp = 0
    for i in range(resolution):
        # hist[i] is the number of pixels whose value is in i/100 and (i+1)/100
        if i == 99:
            val = np.where((img_px >= i/resolution) & (img_px <= (i+1)/resolution))[0]
        else:
            val = np.where((img_px >= i/resolution) & (img_px < (i+1)/resolution))[0]
        
        hist[i] = np.sum(img_px[val])

    # plt.plot(hist)
    # plt.show()

    _, hist_mass_center = get_mass_center(hist, 0, resolution)
    last_th = hist_mass_center

    while True:
        _, hist_mass_center_left = get_mass_center(hist, 0, int(last_th))
        _, hist_mass_center_right = get_mass_center(hist, int(last_th), resolution)
        hist_mass_center = (hist_mass_center_left + hist_mass_center_right)/2
        if np.abs(hist_mass_center - last_th) < 0.01:
            return hist_mass_center/resolution
        else:
            last_th = hist_mass_center

@jit(nopython=True)
def get_mass_center(f, a, b):
    # return integral(i*f,a,b)/integral(f,a,b)
    area = 0
    area_tmp = 0
    for i in range(a, b):
        area += f[i]
        area_tmp += i*f[i]
    
    return area, area_tmp/(area+1e-8)