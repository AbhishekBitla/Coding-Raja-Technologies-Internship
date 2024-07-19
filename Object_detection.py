import cv2
import numpy as np
import tensorflow as tf

# Load YOLOv4 model
yolo = tf.saved_model.load("path/to/yolov4")

def load_yolo():
    return yolo
def detect_objects(frame, yolo_model):
    input_size = 416
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(img, (input_size, input_size))
    img = img / 255.0
    img = img[np.newaxis, ...].astype(np.float32)
    
    infer = yolo_model.signatures['serving_default']
    batch_data = tf.constant(img)
    pred_bbox = infer(batch_data)
    
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    
    return boxes, pred_conf
from centroid_tracker import CentroidTracker  # You need to implement or import this

tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)
def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or specify a video file
    yolo_model = load_yolo()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        boxes, confidences = detect_objects(frame, yolo_model)
        
        rects = []
        for i in range(len(boxes)):
            if confidences[0][i][0] > 0.5:
                box = boxes[0][i]
                (startX, startY, endX, endY) = box
                rects.append((startX, startY, endX, endY))
        
        objects = tracker.update(rects)
        
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
