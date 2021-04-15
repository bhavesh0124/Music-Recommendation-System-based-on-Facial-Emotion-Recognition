import os
import pymongo
import cv2
import numpy as np
import pandas as pd
import warnings
import tkinter as tk
from keras.models import load_model

warnings.filterwarnings("ignore")
from jeanCV import skinDetector


ret=True

cap=cv2.VideoCapture(0)


caffeModel="caffe/Detection_model.caffemodel"
prototextPath="caffe/deploy.prototxt.txt"

net = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)

while ret:
	
	ret,frame=cap.read()
	frame=cv2.flip(frame,1)
	(h,w) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	image=frame

	for i in range(0, detections.shape[2]):
	 
	    confidence = detections[0, 0, i, 2]

	    if confidence > 0.6:
	      
	        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	        (startX, startY, endX, endY) = box.astype("int")
	        
	        text = "{:.2f}%".format(confidence * 100)
	        y = startY - 10 if startY - 10 > 10 else startY + 10

	        cropped=image[startY:endY,startX:endX]
	        detector = skinDetector(frame)
	        output=detector.find_skin()


	        model=load_model('model.hdf5')
	        model.predict(cropped)[0]

	        cv2.imshow('output',output)

	        cv2.rectangle(image, (startX, startY), (endX, endY),(255, 0, 0), 2)
	        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	        cv2.imshow('detections',cropped)


	cv2.imshow('frame',frame)
	cv2.waitKey(100)
	key = cv2.waitKey(1) & 0xFF

	if(key == ord("q")):
		break


cap.release()
cv2.destroyAllWindows()


