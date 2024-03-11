from deepface import DeepFace
import pandas as pd
import cv2
import os
from datetime import datetime
import csv
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
model = 'liveness.model'
model = tf.keras.models.load_model(model)

#
models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]

# result = DeepFace.analyze(img_path = "frame.jpg",
#         actions = ['age', 'gender', 'race', 'emotion'],

# )
listStud=os.listdir('data')
counter = 0
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    if counter % 200 == 0:
        result = DeepFace.find(img_path=frame, db_path="data", model_name=models[0], detector_backend="opencv",enforce_detection = False, silent=True)

        try:
            obj=os.path.split(os.path.splitext(result[0]["identity"][0])[0])[1]
            xmin = int(result[0]['source_x'][0])
            ymin = int(result[0]['source_y'][0])
            w = result[0]['source_w'][0]
            h = result[0]['source_h'][0]
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            img = frame[ymin:ymax, xmin:xmax]
            img = cv2.resize(img, (32, 32))
            img = img.astype('float') / 255.0
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            liveness = model.predict(img)

            liveness = liveness[0].argmax()
            print(liveness)

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
