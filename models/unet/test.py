
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/aditomer/baseline-wp')
from data import transforms
import cv2
from scipy.fftpack import fft2, ifft2, ifftshift, fftshift

resolution=320
# oimage = cv2.imread('DIPSourceHW1.jpg',0)
center_fraction=0.08
acceleration=4
num_cols=320
rng = np.random.RandomState()
for i in range(5):
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        mask=mask*0.5+0.5*rng.uniform(size=num_cols)
        sorted_parameters = np.sort(mask)
        threshold = sorted_parameters[320 - int(320 / acceleration)-1]
        bimask = mask > threshold
        plt.imshow(np.ones((320,320))*bimask)
        plt.show()
        print(i)