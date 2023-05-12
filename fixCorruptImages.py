import os
# fire.479 is corrupted, with this script it will delete it
# and rename all the next images to keep consistenci with the numbers

#480 ---> 479 
#481 ---> 480
#............
#and so on
os.remove("../fire_dataset/fire_images/fire.479.png")

for i in range(480,750+1):
    os.rename(f"../fire_dataset/fire_images/fire.{i}.png", f"../fire_dataset/fire_images/fire.{i-1}.png")
print("done")