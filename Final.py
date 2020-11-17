import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
eyeCascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
profileCascade = cv2.CascadeClassifier('./haarcascade_profileface.xml')
fullbodyCascade = cv2.CascadeClassifier('./haarcascade_fullbody.xml')
cap=cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()
    frame = cv2.flip(frame, 1)
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    profile_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    eyes = eyeCascade.detectMultiScale(roi_gray,scaleFactor= 1.1,minNeighbors=6)
    profile=profileCascade.detectMultiScale(profile_gray, scaleFactor=1.1,minNeighbors=3)
    fullbody=fullbodyCascade.detectMultiScale(profile_gray, scaleFactor=1.1,minNeighbors=3)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
    for x, y, w, h in profile:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    for x, y, w, h in fullbody:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    
        
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('video', frame)
cap.release()
cv2.destroyAllWindows()