import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    frame = results[0].plot()

    cv2.imshow("Pose Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break