import cv2
import os

gesture_name = "thumbs_up"   # CHANGE THIS EVERY TIME
save_dir = f"dataset/{gesture_name}"

os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press C to capture image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
    roi = frame[100:400, 100:400]

    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(1)

    if key == ord('c'):
        cv2.imwrite(f"{save_dir}/{count}.jpg", roi)
        print("thumb image:", count)
        count += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

