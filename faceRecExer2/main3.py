from deepface import DeepFace
import pandas as pd
import cv2
import os
from datetime import datetime
import csv
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
model = YOLO("l_version_1_300.pt")
classNames = ["fake", "real"]

listStud=os.listdir('data')
counter = 0
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    if counter % 200 == 0:
        result = DeepFace.find(img_path=frame, db_path="data", model_name="VGG-Face", detector_backend="opencv",enforce_detection = False, silent=True)

        try:
            obj=os.path.split(os.path.splitext(result[0]["identity"][0])[0])[1]
            results = model(frame, stream=True,verbose=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    cv2.putText(frame, classNames[cls], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)


            cv2.putText(frame, obj, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d %H:%M:%S")
            print(current_date)
            with open('attendance' + '.csv', 'a+', newline='') as f:
                 lnwriter = csv.writer(f)
                 lnwriter.writerow([obj, current_date])
        except:
            cv2.putText(frame, "no match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        cv2.imshow('img', frame)
# cv2.imshow('img', frame)
# cv2.waitKey(0)
    counter += 1

    if cv2.waitKey(1) == 27:
        break
#
cap.release()
cv2.destroyAllWindows()