import cv2
import numpy as np

def getImageFeatures (image):

    # Get the channels of the image.
    b, g, r = cv2.split(image)

    # Normalize each channel.
    r_norm = r / 255
    g_norm = g / 255
    b_norm = b / 255

    # "Flatten" the channels.
    r_norm = r_norm.reshape([-1])
    g_norm = g_norm.reshape([-1])
    b_norm = b_norm.reshape([-1])

    # Stack the individual arrays to a single array.
    flat_rgb = np.vstack([r_norm, g_norm, b_norm])

    # Center the data.
    x_center = flat_rgb - np.mean(flat_rgb, axis = 1, keepdims = True)

    # Calculate covariance matrix.
    cov = np.cov(x_center)

    # Perform eigendecomposition of the covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort the eigenvalues and eigenvectors.
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    return eigenvectors.reshape([-1])
