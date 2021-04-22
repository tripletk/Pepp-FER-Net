import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

do_once = True

model = load_model('pepp-FKD.h5')
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        cropped_face = frame[y:y+h, x:x+w]
        gray_cropped = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_cropped, (96, 96), interpolation = cv2.INTER_AREA)
        # timag = []
        # timag.append(resized_face)
        timage_list = np.array(resized_face,dtype = 'float')
        X_test = timage_list.reshape(-1,96,96,1)
        pred = model.predict(X_test)
        # cv2.circle(frame, (int(pred[0][0]) + x, int(pred[0][1]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][2]) + x, int(pred[0][3]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][4]) + x, int(pred[0][5]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][6]) + x, int(pred[0][7]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][8]) + x, int(pred[0][9]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][10]) + x, int(pred[0][11]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][12]) + x, int(pred[0][13]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][14]) + x, int(pred[0][15]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][16]) + x, int(pred[0][17]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][18]) + x, int(pred[0][19]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][20]) + x, int(pred[0][21]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][22]) + x, int(pred[0][23]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][24]) + x, int(pred[0][25]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][26]) + x, int(pred[0][27]) + y), 0, (0, 255, 0), -1)
        # cv2.circle(frame, (int(pred[0][28]) + x, int(pred[0][29]) + y), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][0]), int(pred[0][1])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][2]), int(pred[0][3])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][4]), int(pred[0][5])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][6]), int(pred[0][7])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][8]), int(pred[0][9])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][10]), int(pred[0][11])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][12]), int(pred[0][13])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][14]), int(pred[0][15])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][16]), int(pred[0][17])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][18]), int(pred[0][19])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][20]), int(pred[0][21])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][22]), int(pred[0][23])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][24]), int(pred[0][25])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][26]), int(pred[0][27])), 0, (0, 255, 0), -1)
        cv2.circle(resized_face, (int(pred[0][28]), int(pred[0][29])), 0, (0, 255, 0), -1)
        cv2.imshow('Video', resized_face)
    
    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    #cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    #cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()