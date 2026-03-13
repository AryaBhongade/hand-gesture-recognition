import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

# LOAD MODEL
model = load_model("model.h5")

# CLASS NAMES (VERY IMPORTANT: same order as folders)
class_names = ["fist", "palm", "thumbs_up"]

IMG_SIZE = 64

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # DRAW BOX
    x1, y1 = 100, 100
    x2, y2 = 400, 400
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    roi = np.reshape(roi, (1, IMG_SIZE, IMG_SIZE, 3))

    # PREDICT
    prediction = model.predict(roi, verbose=0)
    class_id = np.argmax(prediction)
    label = class_names[class_id]

    # SHOW TEXT
    cv2.putText(frame, label, (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



