import cv2
import numpy as np

def getPCAFeatures (image):

    # Get the channels of the image.
    b, g, r = cv2.split(image)

    # Create an HSV representation of the image.
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get the HSV channels of the image.
    h, s, v = cv2.split(img_hsv)

    # Normalize each channel.
    r_norm = r / 255
    g_norm = g / 255
    b_norm = b / 255
    h_norm = h / 179
    s_norm = s / 255
    v_norm = v / 255

    # "Flatten" the channels.
    r_norm = r_norm.reshape([-1])
    g_norm = g_norm.reshape([-1])
    b_norm = b_norm.reshape([-1])
    h_norm = h_norm.reshape([-1])
    s_norm = s_norm.reshape([-1])
    v_norm = v_norm.reshape([-1])

    # Stack the individual arrays to a single array.
    flat_ch = np.vstack([r_norm, g_norm, b_norm, h_norm, s_norm, v_norm])

    # Center the data.
    x_center = flat_ch - np.mean(flat_ch, axis = 1, keepdims = True)

    # Calculate covariance matrix.
    cov = np.cov(x_center)

    # Perform eigendecomposition of the covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort the eigenvalues and eigenvectors.
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    return eigenvectors.reshape([-1])
