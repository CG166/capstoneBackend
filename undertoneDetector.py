#Importing libraries
import cv2
import numpy as np
import math

#undertoneimages = ["warmRed.png", "neutralRed.png", "coolRed.png", "warmTan.png", "neutralBeige.png", "coolBrown.png"]
undertoneimages = ['warmRed.png', 'warmGreen.png', 'warmBlue.png', 'warmGray.png', 'warmPink.png', "warmTan.png",
 'neutralRed.png', 'neutralGreen.png', 'neutralBlue.png', 'neutralGray.png', 'neutralPink.png', "neutralBeige.png",
 'coolRed.png', 'coolGreen.png', 'coolBlue.png', 'coolGray.png', 'coolPink.png', "coolBrown.png"]


for index in range(len(undertoneimages)):
    ##Images##
    #Default BGR image for OpenCV
    image = cv2.imread(undertoneimages[index])
    #cLab image for undertone detection
    cLab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    #Average the color of the image
    avgLab = cLab_image.mean(axis=(0,1))
    #Normalize CLAB values
    L = avgLab[0] * 100 / 255
    a = avgLab[1] - 128
    b = avgLab[2] - 128
    a2 = math.pow(a,2)
    b2 = math.pow(b,2)
    chroma = math.sqrt(a2+b2)
    hue = (math.atan2(b, a)) * (180/math.pi)
    if hue < 0:
        hue = hue + 360

    print(f"\nPrintout for {undertoneimages[index]}: \n")
    print(f" Lightness: {round(L)}\n Green-Red axis: {round(a)}\n Blue-Yellow axis: {round(b)}")
    #print(f"\n Hue: {round(hue)}\n")
    print(f"\n Chroma: {round(chroma)} \n Hue: {round(hue)}")



