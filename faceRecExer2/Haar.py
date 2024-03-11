import cv2
import numpy as np

cap = cv2.VideoCapture(0)
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
count = 0
ID = input("Enter Id:")
while True:
    ret, frame = cap.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDet.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if count <= 500:
            cv2.imwrite(f'Dataset/DataCollect/{count}.{ID}.jpg', frameGray[y:y + h, x:x + w])
            count += 1
    cv2.imshow('img', frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
