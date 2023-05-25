import cv2
import numpy as np
from matplotlib import pyplot as plt
# import pickle
# import os

import pandas as pd
from FeatureExtraction import getImageFeatures

#structure of the folder:
#projects
#|fire-detection
#||fire_dataset
#|||fire_images
#|||non_fire_images
#||preprocessing
#||data
#|||dataset.pk

#do this only the first time, then comment from here:
#we remove image 370 from fire_images because it gives some errors
######################################################
#os.remove("projects/fire-detection/fire_dataset/fire_images/fire.370.png")
#renaming to keep consistency
#for i in range(371, 756):
#    os.rename(f"projects/fire-detection/fire_dataset/fire_images/fire.{i}.png",
#              f"projects/fire-detection/fire_dataset/fire_images/fire.{i-1}.png")
#to here:
######################################################"""

#1) coment this out to load the images the first time
fire_images = [cv2.imread(f"data/fire_dataset/fire_images/fire.{i}.png") for i in range(1,756)]
non_fire_images = [cv2.imread(f"data/fire_dataset/non_fire_images/non_fire.{i}.png") for i in range(1,245)]

# #2) comment this out to save lists in a file
# with open("projects/fire-detection/data/dataset.pk", "wb") as f:
#     pickle.dump((fire_images, non_fire_images), f)

# #3) comment this out to load the lists from the file
# """with open("projects/fire-detection/data/dataset.pk", "rb") as f:
#     fire_images, non_fire_images = pickle.load(f)"""

# ******************************************************************************

# Store extracted features from PCA.
img_features = []
i = 1

# Extract features from "fire" images.
for image in fire_images:
    print(f"[ > ] Extracting features from fire image {i}.")
    img_features.append(getImageFeatures(image))
    i += 1

i = 1

# Extract features from "non-fire" images.
for image in non_fire_images:
    print(f"[ > ] Extracting features from non-fire image {i}.")
    img_features.append(getImageFeatures(image))
    i += 1

# Generate the labels for the data.
fire_labels = np.ones(len(fire_images))
non_fire_labels = np.zeros(len(non_fire_images))
label_set = np.hstack([fire_labels, non_fire_labels])

# Create a feature DataFrame.
dataset = pd.DataFrame(data = img_features)
dataset = pd.concat([dataset, pd.DataFrame(label_set, columns = ["Label"])],
                    axis = 1)

# Randomize the order of the samples.
dataset = dataset.sample(frac = 1)
dataset.reset_index(drop = True, inplace = True)

# Save generated features to a .csv file.
dataset.to_csv("data/PCA_Features.csv", index = False)
