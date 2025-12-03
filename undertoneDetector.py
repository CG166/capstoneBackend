#Importing libraries
import cv2
import numpy as np

undertoneimages = ["warmRed.png", "neutralRed.png", "coolRed.png", "warmTan.png", "test4.png", "neutralBeige.png", "coolBrown.png"]


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
##################################################################################
#Average the color of the image


