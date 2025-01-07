import cv2
from ultralytics import YOLO

model = YOLO("./runs/detect/train4/weights/best.pt")

num = 0
camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break
    results = model(frame)
    cv2.imshow("frame", frame)

    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()