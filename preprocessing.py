import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os

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
for i in range(371, 756):
    os.rename(f"projects/fire-detection/fire_dataset/fire_images/fire.{i}.png", 
              f"projects/fire-detection/fire_dataset/fire_images/fire.{i-1}.png")
#to here:
######################################################"""

#1) coment this out to load the images the first time
fire_images = [cv2.imread(f"projects/fire-detection/fire_dataset/fire_images/fire.{i}.png") for i in range(1,755)]
non_fire_images = [cv2.imread(f"projects/fire_dataset/non_fire_images/non_fire.{i}.png") for i in range(1,243)]

#2) comment this out to save lists in a file
with open("projects/fire-detection/data/dataset.pk", "wb") as f:
    pickle.dump((fire_images, non_fire_images), f)
    
#3) comment this out to load the lists from the file
"""with open("projects/fire-detection/data/dataset.pk", "rb") as f:
    fire_images, non_fire_images = pickle.load(f)"""


    

    



