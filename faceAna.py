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
def getlength(p1, p2):
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
    PercentDiff = ((a+b)/2) * 0.1
    if difference < PercentDiff:
        return True
    else:
        return False
    
def isWidest(subject, a, b):
    aBool = approximatelyEqual(subject, a)
    bBool = approximatelyEqual(subject, b)

    if aBool == False and bBool == False:
        if subject > a and subject > b:
            return True    
    return False

def eyeProportion(sBE, lEL, rEL):
    avgEL = (lEL+rEL)/2

    if approximatelyEqual(avgEL, sBE):
        return 0
    elif avgEL > sBE:
        return 1
    elif avgEL < sBE:
        return -1
    
def noseProportion(nW, sBE):

    if approximatelyEqual(nW, sBE):
        return 0
    elif nW > sBE:
        return 1
    elif nW < sBE:
        return -1
    
def lipProportion(nw, lipLen):

    if (nw/lipLen) < 0.8 and (nw/lipLen) > 0.6:
        return 0
    elif (nw/lipLen) < 0.6:
        return 1
    elif approximatelyEqual(nw, lipLen) or lipLen < nw or (nw/lipLen) > 0.8:
        return -1


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
    forehead1 = ()
    forehead2 = ()
    nose1 = ()
    nose2 = ()
    lip1 = ()
    lip2 = ()
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
            elif i == 362:
                rEyeInnerCorner = (x, y)

            if i == 130:
                lEyeOuterCorner = (x, y)
            elif i == 359:
                rEyeOuterCorner = (x, y)

            if i == 10:
                faceTop = (x,y)
            elif i == 152:
                faceBottom = (x, y)

            if i == 172:
                jawLine1 = (x,y)
            elif i == 397:
                jawline2 = (x, y)
            
            if i == 176:
                chin1 = (x,y)
            elif i == 400:
                chin2 = (x, y)

            if i == 234:
                cheekbone1 = (x,y)
            elif i == 454:
                cheekbone2 = (x, y)

            if i == 21:
                forehead1 = (x,y)
            elif i == 251:
                forehead2 = (x, y)

            if i == 129:
                nose1 = (x,y)
            elif i == 358:
                nose2 = (x, y)

            if i == 61:
                lip1 = (x,y)
            elif i == 291:
                lip2 = (x, y)
    proportions = []

    #Get width of chin
    chinWidth = getlength(chin1, chin2)

    #Get width of jaw
    jawWidth = getlength(jawLine1, jawline2)

    #Get width of cheekbones
    faceWidth = getlength(cheekbone1, cheekbone2)

    #Forhead Width
    foreheadWidth = getlength(forehead1, forehead2)

    #Get face lengths
    faceLength = getlength(faceBottom, faceTop)
    faceLength = faceLength + (faceLength/6)

    #Get length between eyes
    lenBetweenEyes = getlength(lEyeInnerCorner, rEyeInnerCorner)
    
    #Get r eye length
    rEyeLength = getlength(rEyeOuterCorner, rEyeInnerCorner)

    #Get left eye length
    lEyeLength = getlength(lEyeInnerCorner, lEyeOuterCorner)

    #Get nose width
    nWidth = getlength(nose1, nose2)

    #Get mouth length
    mWidth = getlength(lip1, lip2)

    #Exam Eye Proportion
    eyePorp = eyeProportion(lenBetweenEyes, lEyeLength, rEyeLength)
    proportions.append(eyePorp)

    #Exam nose width
    nosePorp = noseProportion(nWidth, lenBetweenEyes)
    proportions.append(nosePorp)

    #Mouth length
    mouthPorp = lipProportion(nWidth, mWidth)
    proportions.append(mouthPorp)

    faceshape = ""
    focalpoints=[]

    #Find focal points
    eyes = proportions[0]
    nose = proportions[1]
    mouth = proportions[2]

    if nose > eyes and nose > mouth and mouth == eyes:
        focalpoints = ["nose"]
    elif nose > eyes and nose > mouth and mouth != eyes:
        if mouth > eyes:
            focalpoints = ["nose", "mouth"]
        else:
            focalpoints = ["eyes", "nose"]
    elif mouth > nose and mouth > eyes and nose == eyes:
        focalpoints = ["mouth"]
    elif mouth > nose and mouth > eyes and nose != eyes:
        if nose > eyes:
            focalpoints = ["nose", "mouth"]
        else:
            focalpoints = ["eyes", "mouth"]
    elif eyes > mouth and eyes > nose and mouth == nose:
        focalpoints = ["eyes"]
    elif eyes > mouth and eyes > nose and mouth != nose:
        if mouth > nose:
            focalpoints = ["eyes", "mouth"]
        else:
            focalpoints = ["eyes", "nose"]
    elif mouth == eyes and eyes == nose:
            focalpoints = []
    elif eyes < mouth and eyes < nose and nose == mouth:
        focalpoints = ["nose", "mouth"]
    elif nose < mouth and nose < eyes and eyes == mouth:
        focalpoints = ["eyes", "mouth"]
    elif mouth < eyes and mouth < nose and nose == eyes:
        focalpoints = ["eyes", "nose"]


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
    print(f"Focal Points are {focalpoints}")


##RUN##
testimages = ["test.jpg", "test1.png", "test2.png", "test3.png", "test4.png", "test5.png", "test7.png", "test8.png", "test9.jpg", "square.png"]

for i in range(len(testimages)):
    faceAnalysis(testimages[i])