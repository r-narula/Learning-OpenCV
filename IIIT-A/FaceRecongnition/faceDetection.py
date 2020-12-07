import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
# from cv2.data.haarcascades import haarcascade_frontalface_alt2

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_alt2.xml')

# Loading the image to be tested
test_image = cv2.imread("/home/mononoke/ChessDetection/IIIT-A/FaceRecongnition/elonMusk.jpg")

# Convert to Gray Scale Image
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

haar_cascade_face = cv2.CascadeClassifier(haar_model)

def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()
    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors = 5) # returns the points of the image
    # print("This is the Faces_REC",faces_rect)
    '''
    At this point we have detected the face in the image..
    '''
    # print("number of faces found is : ",len(faces_rect))
    print(faces_rect)    
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print("W is :",w)
        print("Area is : ",w*h)
        # 1 pixel (X) = 0.0264583333
        width = w # in cm
        height = h # in cm
        print(width,height)
        area = width*height
        minor = int(np.sqrt(area*5/16))
        major = int(1.5*minor)
        print(f'{minor} length {major}')
        tup = (major,minor)
        cv2.ellipse(image_copy,(x+w//2,y+h//2),tup,0,0,360,(0,255,0),2)        
    return image_copy

# list = ['elon.webp','group.webp','elonMusk.jpg','elonMuskTest.jpg','bill.jpg']
list = os.listdir('/home/mononoke/ChessDetection/IIIT-A/FaceRecongnition/detectionImages/')
for i in list:
    test_image2 = cv2.imread(f"/home/mononoke/ChessDetection/IIIT-A/FaceRecongnition/detectionImages/{i}")
    test_image2 = cv2.rotate(test_image2,cv2.ROTATE_180)
    print("This is the test image size",test_image2.shape)
    test_image2 = cv2.resize(test_image2,(1024,768))
    #call the function to detect faces
    faces = detect_faces(haar_cascade_face, test_image2)
    cv2.imshow('img',faces)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
