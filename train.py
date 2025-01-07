import cv2
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("./runs/detect/train4/weights/best.pt")

# Train the model on the COCO8 example dataset for 100 epochs
#results = model.train(data="./www.pcmns.v1i.yolov11/data.yaml", epochs=30, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
img = "./6/img_18.jpg"

img1 = cv2.imread(img, flags=1)
while True:
    cv2.imshow("test", img1)
    key = cv2.waitKey(1)
    results = model(img1)
    print(results[0].boxes.xyxy)
    if key in [27, ord('q')]:


        break


