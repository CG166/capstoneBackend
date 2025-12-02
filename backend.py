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
testimages = ["test.jpg", "test1.png", "test2.png", "test3.png", "test4.png", "test5.png", "test6.png", "test7.png", "test8.png", "test9.jpg"]

##Images##
#Default BGR image for OpenCV
image = cv2.imread(testimages[0])
#RGB image for mediapipe
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
else:
    adjusted_width = width
    adjusted_height = height
#################################################################################

###############Before image display for comparison (debugging)###################
cv2.imshow("Image before white-balancing", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#################################################################################

############Fallback Gray-World white-balancing algorithm########################




##################################################################################

#Getting facial Landmarks
result = face_mesh.process(image)

#Getting facial landmark coordinates
for facial_landmarks in result.multi_face_landmarks:
    for i in range(0, 468):
        pt = facial_landmarks.landmark[i]
        x = int(pt.x * adjusted_width)
        y = int(pt.y * adjusted_height)


