"""
**Wavelet Feature Extraction**

Performs a Wavelet Decomposition of the RGB and HSV channels of a given image,
and generates features given each coefficient of the decomposition.
"""


import cv2
import pywt
import numpy as np


def getWaveletFeatures(img):

    # Create an HSV representation of the image.
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get a histogram for each channel of the image.
    img_hists = [
        # RGB Channels
        cv2.calcHist([img], [2], None, [256], [0,256]),
        cv2.calcHist([img], [1], None, [256], [0,256]),
        cv2.calcHist([img], [0], None, [256], [0,256]),

        # HSV Channels
        cv2.calcHist([img_hsv], [0], None, [180], [0,180]),
        cv2.calcHist([img_hsv], [1], None, [256], [0,256]),
        cv2.calcHist([img_hsv], [2], None, [256], [0,256]),
    ]

    img_features = []

    # Get features for each histogram.
    for hist in img_hists:

        # Get the first (most relevant) wavelet coefficient.
        wav_coefficient = pywt.wavedec(hist, "db20")[0]

        # Get features for the coefficient.
        img_features += getCoefficientFeatures(wav_coefficient)

    return img_features


def getCoefficientFeatures(wav_coefficient):

    n5 = np.nanpercentile(wav_coefficient, 5)
    n25 = np.nanpercentile(wav_coefficient, 25)
    n75 = np.nanpercentile(wav_coefficient, 75)
    n95 = np.nanpercentile(wav_coefficient, 95)

    median = np.nanpercentile(wav_coefficient, 50)
    mean = np.nanmean(wav_coefficient)
    std = np.nanstd(wav_coefficient)
    var = np.nanvar(wav_coefficient)
    rms = np.nanmean(np.sqrt(wav_coefficient**2))

    return [n5, n25, n75, n95, median, mean, std, var, rms]
