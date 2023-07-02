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
        coeff = pywt.wavedec(hist, "db20")[0]

        # Get features for the coewfficient.
        img_features += getCoefficientFeatures(coeff)

    return img_features

def getCoefficientFeatures(coeff):

    n5 = np.nanpercentile(coeff, 5)
    n25 = np.nanpercentile(coeff, 25)
    n75 = np.nanpercentile(coeff, 75)
    n95 = np.nanpercentile(coeff, 95)

    median = np.nanpercentile(coeff, 50)
    mean = np.nanmean(coeff)
    std = np.nanstd(coeff)
    var = np.nanvar(coeff)
    rms = np.nanmean(np.sqrt(coeff**2))

    return [n5, n25, n75, n95, median, mean, std, var, rms]

