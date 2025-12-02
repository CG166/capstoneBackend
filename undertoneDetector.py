#Importing libraries
import cv2
import numpy as np

undertoneimages = ["test.jpg", "test1.png", "test2.png", "test3.png", "test4.png", "test5.png", "test6.png", "test7.png", "test8.png", "test9.jpg"]


##Images##
#Default BGR image for OpenCV
image = cv2.imread(undertoneimages[0])
#cLab image for undertone detection
cLab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

#Origninal image height and width (use for maximum accuracy)
height, width = image.shape[:2]

##################Image Display Scaling for debugging only######################
screen_width = 800
screen_height = 600
scale = min(screen_width / width, screen_height / height)

if scale < 1:
    adjusted_width = int(width * scale)
    adjusted_height = int(height * scale)
    image = cv2.resize(image, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
    cLab_image = cv2.resize(cLab_image, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
else:
    adjusted_width = width
    adjusted_height = height