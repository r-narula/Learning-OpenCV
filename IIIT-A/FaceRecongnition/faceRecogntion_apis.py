import cv2
import numpy as np
import face_recognition
import os

'''
Step1 -> Load the images 
'''
path = '/home/mononoke/ChessDetection/IIIT-A/FaceRecongnition/detectionImages'
imgElon = face_recognition.load_image_file(os.path.join(path,'elonMusk.jpg'))
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElonTest = face_recognition.load_image_file("/home/mononoke/ChessDetection/IIIT-A/Lenna.png")
imgElonTest = cv2.cvtColor(imgElonTest,cv2.COLOR_BGR2RGB)

'''
Step2 -> starting with the main work.
'''
faceLoc = face_recognition.face_locations(imgElon)[0] # single iamge getting the first element of it
print("Printing the values of the face(T R B L) : ",faceLoc)
encodeElon = face_recognition.face_encodings(imgElon)[0] # encodes the image and makes them to 128 values.
img = cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,0,0),2)

# Test Image
faceLocTest = face_recognition.face_locations(imgElonTest)[0] # single iamge getting the first element of it
print("Printing the values of the face(T R B L) : ",faceLocTest)
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0] # encodes the image and makes them to 128 values 
img = cv2.rectangle(imgElonTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(0,0,0),2)
'''
Step3 -> Comparing results
'''

results = face_recognition.compare_faces([encodeElon],encodeElonTest)
print(results)


cv2.imshow('img',imgElon)
if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

cv2.imshow('img',imgElonTest)
if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

