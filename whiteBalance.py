#Importing libraries
import cv2
import mediapipe as mp
import numpy as np

#References
#https://www.youtube.com/watch?v=Z0-iM37wseI #White-balancing mechanisms


#Initializing face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8
)

##############Facial landmark indices#################

#Testing images for Debugging
testimages = ["test.jpg", "test1.png", "test2.png", "test3.png", "test4.png", "test5.png", "test7.png", "test8.png", "test9.jpg"]

##Images##
#Default BGR image for OpenCV
image = cv2.imread(testimages[8])
#RGB image for mediapipe
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

############Fallback Gray-World white-balancing algorithm########################
def grayWorldAlgo(CLABimage):
    lVal = CLABimage[:,:,0]
    aVal = CLABimage[:,:,1]
    bVal = CLABimage[:,:,2]
    avgAChannel = np.mean(aVal)
    avgBChannel = np.mean(aVal)
    aVal = aVal - ((avgAChannel-128)*(lVal/255.0))
    bVal = bVal - ((avgBChannel-128)*(lVal/255.0))

    #Setting white-balanced values back to CIELAB image
    CLABimage[:,:,1] = aVal
    CLABimage[:,:,2] = bVal

    GWWhiteBalancedImage = cv2.cvtColor(CLABimage, cv2.COLOR_LAB2BGR)
    return GWWhiteBalancedImage
##################################################################################

#Getting facial Landmarks
result = face_mesh.process(image)

####################Landmark indices arrays#######################
lEyePoints = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
rEyePoints = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
##################################################################
#Test
testZone = [111,117,118,101,36,205,207,187,123]

#########White Patch#########
lAddX = 0
lAddY = 0

rAddX = 0
rAddY = 0
#############################

#Getting facial landmark coordinates
for facial_landmarks in result.multi_face_landmarks:
    for i in range(0, 468):
        pt = facial_landmarks.landmark[i]
        x = int(pt.x * adjusted_width)
        y = int(pt.y * adjusted_height)

        for z in range(len(testZone)):
            if(i == testZone[z]):
                point = (x, y)
                testZone[z] = point
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)

        #Drawing left eye cordinates
        for l in range(len(lEyePoints)):
            if(i == lEyePoints[l]):
                if(i == 161 or i == 144):
                    #print(f"X L-value: {x}")
                    lAddX = lAddX + x
                    #print(f"Y L-value: {y}")
                    lAddY = lAddY + y
                    cv2.circle(image, (x, y), 3, (100, 0, 0), -1)
                    #cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)


        #Drawing right eye cordinates
        for m in range(len(rEyePoints)):
            if(i == rEyePoints[m]):
                if(i == 388 or i == 373):
                    #print(f"X R-value: {x}")
                    rAddX = rAddX + x
                    #print(f"Y R-value: {y}")
                    rAddY = rAddY + y
                    cv2.circle(image, (x, y), 3, (0, 0, 100), -1)
                    #cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

########Getting eye-white midpoints#########
lMidpointX = int(lAddX/2)
lMidpointY = int(lAddY/2)
#cv2.circle(image, (lMidpointX, lMidpointY), 3, (0, 100, 0), -1)

rMidpointX = int(rAddX/2)
rMidpointY = int(rAddY/2)
#cv2.circle(image, (rMidpointX, rMidpointY), 3, (0, 100, 0), -1)
############################################

#############Getting Small rectangle of white###############
lhStart, lwStart, lhWidth, lwWidth = lMidpointY, lMidpointX, 3, 3
rhStart, rwStart, rhWidth, rwWidth = rMidpointY-3, rMidpointX-3, 3, 3

clone = image.copy()
img = clone
whitePatchL = img[lhStart:lhStart+lhWidth, lwStart:lwStart+lwWidth]
whitePatchR = img[rhStart:rhStart+rhWidth, rwStart:rwStart+rwWidth]

############################################################

############White patch white-balancing algorithm########################
def whitePatchAlgo(image, patch):
    #Image in BRG/default
    normalizedImage = image/patch.max(axis=(0,1))
    WPWhiteBalancedImage = normalizedImage.clip(0,1)
    return WPWhiteBalancedImage
##################################################################################

def drawFeauture(image, points):
    #Drawing feauture outline
    dPoints = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [dPoints], isClosed=True, color=(0, 0, 255), thickness=2)
    return dPoints

dummy = drawFeauture(image, testZone)

############Testing White-balanced images##################
cLab_image = grayWorldAlgo(cLab_image)
#cLab_image = whitePatchAlgo(image, whitePatchL)

#Displaying image
#PatchRectangle
#cv2.rectangle(clone, (lwStart,lhStart), (lwStart+lwWidth, lhStart+lhWidth), (0, 0, 0), 1)
#cv2.rectangle(clone, (rwStart,rhStart), (rwStart+rwWidth, rhStart+rhWidth), (0, 0, 0), 1)
#cv2.imshow("Rect", img)
#
cv2.imshow("Image before white-balancing", image)
cv2.imshow("Image after white-balancing", cLab_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
###########################################################


