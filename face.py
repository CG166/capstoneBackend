import cv2
import mediapipe as mp
import numpy as np
import math
from pygeom.geom3d import Vector

##Variables##
#Initializing face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8
)

###Functions###
def getlength(image, p1, p2):
    #Draw line
    cv2.line(image, p1, p2, (0,0,255), 2, cv2.LINE_AA)

    #Get Vector
    x1,y1 = p1
    x2, y2 = p2
    v = ((x2-x1), (y2-y1))

    #Get vector length
    x,y = v
    vx = math.pow(x,2)
    vy = math.pow(y,2)
    vLength = math.sqrt(vx+vy)
    vlength = abs(vLength)

    return vLength

def approximatelyEqual(a, b):
    difference = abs(a-b)
    #print(f"Difference is {difference}")
    PercentDiff = ((a+b)/2) * 0.1
    #(f"4% difference is {PercentDiff}")
    if difference < PercentDiff:
        #print(f"{a} and {b} are approximatly the same")
        return True
    else:
        #print(f"{a} and {b} are not approximatly the same")
        return False
    
def isWidest(subject, a, b):
    aBool = approximatelyEqual(subject, a)
    bBool = approximatelyEqual(subject, b)

    if aBool == False and bBool == False:
        if subject > a and subject > b:
            return True    
    return False

def drawLine(image, points):
    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i+1], (100, 0, 0), 2)





def faceAnalysis(filename):
    #####Variables######
    faceTop = ()
    faceBottom = ()
    lEyeOuterCorner = ()
    rEyeOuterCorner = ()
    lEyeInnerCorner = ()
    rEyeInnerCorner = ()
    jawLine1 = ()
    jawline2 = ()
    chin1 = ()
    chin2 = ()
    cheekbone1 = ()
    cheekbone2 = ()
    cheekbone = [143, 111, 117, 118, 119, 120]
    face = [54,162, 234,172,176, 152, 400, 397, 454, 389, 284]
    forehead1 = ()
    forehead2 = ()
    ####################
    #Get image
    image = cv2.imread(filename) #Default BGR image for OpenCV
    #Make RGB and CIELAB copy
    RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #For mediapipe

    #Image height and width
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
    ####################################################################################


    #Get facial landmarks
    result = face_mesh.process(RGBimage)

    #Getting relevant facial landmark coordinates
    for facial_landmarks in result.multi_face_landmarks:
        for i in range(0, 468):
            pt = facial_landmarks.landmark[i]
            x = int(pt.x * adjusted_width)
            y = int(pt.y * adjusted_height)

            if i == 133:
                lEyeInnerCorner = (x, y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)
            elif i == 362:
                rEyeInnerCorner = (x, y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)

            if i == 130:
                lEyeOuterCorner = (x, y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)
            elif i == 359:
                rEyeOuterCorner = (x, y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)

            if i == 10:
                faceTop = (x,y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)
            elif i == 152:
                faceBottom = (x, y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)

            if i == 172:
                jawLine1 = (x,y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)
            elif i == 397:
                jawline2 = (x, y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)
            
            if i == 176:
                chin1 = (x,y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)
            elif i == 400:
                chin2 = (x, y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)

            if i == 234:
                cheekbone1 = (x,y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)
            elif i == 454:
                cheekbone2 = (x, y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)

            if i == 21:
                forehead1 = (x,y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)
            elif i == 251:
                forehead2 = (x, y)
                cv2.circle(image, (x, y), 3, (100, 0, 0), -1)

            #Getting face cordinates
            for k in range(len(face)):
                if(i == face[k]):
                    point = (x, y)
                    face[k] = point

            

            

    
    #Draw line down face
    cv2.line(image, faceTop, faceBottom, (100,0,0), 2, cv2.LINE_AA)

    #Cheekbone
    #drawLine(image, cheekbone)

    #face
    drawLine(image, face)

    #Get width of chin
    chinWidth = getlength(image, chin1, chin2)
    #print(f"The width of the chin is: {chinWidth}\n")

    #Get width of jaw
    jawWidth = getlength(image, jawLine1, jawline2)
    #print(f"The width of the jaw is: {jawWidth}\n")

    #Get width of cheekbones
    faceWidth = getlength(image, cheekbone1, cheekbone2)
    print(f"The width of the face is: {faceWidth}\n")

    #Forhead Width
    foreheadWidth = getlength(image, forehead1, forehead2)
    #(f"The width of the forehead is: {foreheadWidth}\n")

    #Get face lengths
    faceLength = getlength(image, faceBottom, faceTop)
    faceLength = faceLength + (faceLength/6)
    
    print(f"The length of the face is: {faceLength}\n")

    #Get liplength



    #approximatelyEqual(foreheadWidth, faceWidth)
    #approximatelyEqual(jawWidth, faceWidth)

    faceshape = ""
    focalpoints=[]
    #Faceshape determination

    if approximatelyEqual(faceLength, faceWidth) == True and isWidest(faceWidth) == True:
        faceshape = "round"
    elif (faceLength/faceWidth) >= (1.46) and approximatelyEqual(foreheadWidth, jawWidth) == True:
        faceshape = "rectangle"
    elif (jawWidth/faceWidth) >= .80:
        faceshape = "square"
    elif isWidest(jawWidth, faceWidth, foreheadWidth) == True:
        faceshape = "triangle"
    elif (faceLength/faceWidth) >= (1.46):
        faceshape = "oblong"
    elif isWidest(foreheadWidth, faceWidth, jawWidth) == True:
        faceshape = "heart"
    elif isWidest(faceWidth, foreheadWidth, jawWidth) == True:
        faceshape = "diamond"
    else:
        faceshape = "oval"

    print(f"\nFor {filename}\n")
    print(f"Faceshape is: {faceshape}\n")
    

    #Image display
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

























##RUN##
testimages = ["test.jpg", "test1.png", "test2.png", "test3.png", "test4.png", "test5.png", "test7.png", "test8.png", "test9.jpg"]

for i in range(len(testimages)):
    faceAnalysis(testimages[i])

faceAnalysis("square.png")