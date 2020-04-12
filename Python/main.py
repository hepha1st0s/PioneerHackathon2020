import glob, os
import cv2
import numpy as np

sourceImageDirectory = 'jaffedbase'
imageExtension = "*.tiff"
MOUTH_OPEN_THRESHOLD = 200

mouthCascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)

# detects all mouth patterns using haar cascade and takes lowest
def getMouth(img):
    mouths = mouthCascade.detectMultiScale(img, 1.1, 150, flags=cv2.CASCADE_SCALE_IMAGE)
    if len(mouths) > 0:
        mouthsSorted = sorted(mouths, key=lambda x: x[1])
        return True, mouthsSorted[len(mouthsSorted)-1]
    return False, (None, None, None, None)

def getFace(img):
    faces = faceCascade.detectMultiScale(img, 1.1, 5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        return True, faces[0]
    return False, (None, None, None, None)

def extractMouthROI(img, x, y, w, h):
    return img[y:y+h, x:x+w]

# extends the source image with detected data highlight info
def highlightInImage(img, x, y, w, h):
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

def embedImage(base, embedded):
    base[0:embedded.shape[0], 0:embedded.shape[1]] = embedded
    return base

# displays the result
def displayImage(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

os.chdir(sourceImageDirectory)
#for file in glob.glob(imageExtension):
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   img = cv2.imread(file,0)
    img = gray
    ret, (x,y,w,h) = getMouth(img)
    retFace, (xf,yf,wf,hf) = getFace(img)
    if not ret or not retFace:
        print("nope") 
        continue # no mouth found

    xRatio = float(w) / float(wf)
    yRatio = float(h) / float(hf)
    print('x ration = ' + str(xRatio) + ' / yRatio = ' + str(yRatio))
    print(str(xRatio * yRatio))

    mouthImg = extractMouthROI(img, x, y, w, h)
    edges = cv2.Canny(mouthImg,200,300,apertureSize = 3)

    nonZeroPixelsInMouth = np.count_nonzero(edges)
    print( "Non-zero pixels = " + str(nonZeroPixelsInMouth))

    baseImg = highlightInImage(img, x, y, w, h)
    baseImg = highlightInImage(baseImg, xf, yf, wf, hf)
    baseImg = cv2.putText(baseImg, 'Mouth open' if nonZeroPixelsInMouth > MOUTH_OPEN_THRESHOLD else 'Mouth shut', (80, 50), cv2.FONT_HERSHEY_SIMPLEX , .5, (255, 0, 0) , 1, cv2.LINE_AA)

    #displayImage(embedImage(baseImg, edges))
    cv2.imshow('frame',embedImage(baseImg, edges))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
