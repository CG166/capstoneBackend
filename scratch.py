import cv2
import mediapipe as mp
import numpy as np

#########Functions##########
def drawFeauture(image, points):
    #Drawing feauture outline
    dPoints = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [dPoints], isClosed=True, color=(0, 0, 255), thickness=2)
    return dPoints

def getColor(image, points):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #Grabbing average color
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    mean_color = cv2.mean(hsvImage, mask=mask)[:3]
    return mean_color

def getFaceColor(image, face, eyezone, nose, lips):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #Grabbing average color
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [face], 255)
    cv2.fillPoly(mask, [eyezone, nose, lips], 0)

    #Show
    #cv2.imshow("Mask", mask)
    #cv2.waitKey(1)
    mean_color = cv2.mean(hsvImage, mask=mask)[:3]
    return mean_color

def colorConversions(hsv):
    #bgr = np.array(bgr, dtype=np.uint8).reshape((1, 1, 3))
    #rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #print("Converting BGR:", bgr[0, 0], "to RGB:", rgb[0, 0])
    print("\n\nSkin HSV color: ", hsv, "\n\n")
    mean_hsv_array = np.uint8([[hsv]])  # shape (1,1,3)
    mean_bgr = cv2.cvtColor(mean_hsv_array, cv2.COLOR_HSV2BGR)[0, 0]
    print("Mean color converted back to BGR:", mean_bgr)

####################

#Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8
)

testimages = ["test.jpg", "test1.png", "test2.png", "test3.png", "test4.png", "test5.png", "test6.png", "test7.png", "test8.png", "test9.jpg"]

image = cv2.imread(testimages[0])
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]

#Facial Landmarks
result = face_mesh.process(image)

#Image and Image proccessing
screen_width = 800
screen_height = 600
scale = min(screen_width / width, screen_height / height)

if scale < 1:
    adjusted_width = int(width * scale)
    adjusted_height = int(height * scale)
    image = cv2.resize(image, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)
else:
    adjusted_width = width
    adjusted_height = height

############Coordinates###############
lipPoints = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
facePoints = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 215, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
lEyePoints = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
rEyePoints = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
############CutOuts###################
noseCutout = [64, 48, 115, 220, 45, 4, 275, 440, 344, 278, 294, 460, 326, 2, 97, 98]
eyeZoneCutout = [34, 139, 71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301, 368, 264, 346, 347, 348, 343, 351, 168, 122, 114, 119, 118, 117]
######################################

for facial_landmarks in result.multi_face_landmarks:
    for i in range(0, 468):
        pt = facial_landmarks.landmark[i]
        x = int(pt.x * adjusted_width)
        y = int(pt.y * adjusted_height)
        
        #Snatching lip cordinates
        for j in range(len(lipPoints)):
            if(i == lipPoints[j]):
                point = (x, y)
                lipPoints[j] = point

        #Snatching face cordinates
        for k in range(len(facePoints)):
            if(i == facePoints[k]):
                point = (x, y)
                facePoints[k] = point

        #Snatching left eye cordinates
        for l in range(len(lEyePoints)):
            if(i == lEyePoints[l]):
                point = (x, y)
                lEyePoints[l] = point

        #Snatching right eye cordinates
        for m in range(len(rEyePoints)):
            if(i == rEyePoints[m]):
                point = (x, y)
                rEyePoints[m] = point
        
        #Snatching right eye cordinates
        for n in range(len(noseCutout)):
            if(i == noseCutout[n]):
                point = (x, y)
                noseCutout[n] = point

        #Snatching right eye cordinates
        for o in range(len(eyeZoneCutout)):
            if(i == eyeZoneCutout[o]):
                point = (x, y)
                eyeZoneCutout[o] = point

        cv2.circle(image, (x, y), 3, (100, 0, 0), -1)


#Image display
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
