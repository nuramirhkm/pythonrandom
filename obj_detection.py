import cv2
import numpy as np
import pyrealsense2 as rs
import tensorflow as tf

coco_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_objects_with_realsense(model_path):
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)  # Default resolution
    pipeline.start(config)

    # Load the pre-trained object detection model
    model = tf.saved_model.load(model_path)

    # Get the model's function for inference
    infer = model.signatures['serving_default']

    while True:
        # Get frames from RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert the RealSense frame to a format compatible with TensorFlow
        color_image = np.asanyarray(color_frame.get_data())

        # Perform object detection on the color image
        input_tensor = tf.convert_to_tensor(color_image)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Perform inference using the model function
        detections = infer(input_tensor)

        # Process the detection output to get bounding box coordinates, class labels, and scores
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)
        scores = detections['detection_scores'][0].numpy()

        # Draw bounding boxes on the frame
        for i in range(len(boxes)):
            if scores[i] > 0.5:  # Adjust the confidence threshold as needed
                h, w, _ = color_image.shape
                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                ymin = int(ymin * h)
                ymax = int(ymax * h)
                
                # Draw bounding box
                cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Draw class label and score
                label = f"Class: {classes[i]}, Score: {scores[i]:.2f}"
                cv2.putText(color_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
            
        # Display the frame with bounding boxes
        cv2.imshow('RealSense Object Detection', color_image)

        # Break the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the RealSense pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()

def detect_objects_with_webcam(model_path):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam. Change it if you have multiple cameras.

    # Load the pre-trained object detection model
    model = tf.saved_model.load(model_path)
    infer = model.signatures['serving_default']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        color_image = frame.copy()

        input_tensor = tf.convert_to_tensor(color_image)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = infer(input_tensor)

        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)
        scores = detections['detection_scores'][0].numpy()

        for i in range(len(boxes)):
            if scores[i] > 0.5:
                h, w, _ = color_image.shape
                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                ymin = int(ymin * h)
                ymax = int(ymax * h)

                class_id = classes[i]
                class_label = coco_labels[class_id - 1]  # COCO IDs start from 1

                cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"{class_label}: {scores[i]:.2f}"
                cv2.putText(color_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Webcam Object Detection', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Usage example for using webcam:
model_path = r'C:\Users\aamih\Downloads\BoeingUAV1\BoeingUAVPhase2\pretrained_models\ssd_mobilenet_v2_coco\ssd_mobilenet_v2_coco_2018_03_29\saved_model'
detect_objects_with_webcam(model_path)

# Usage example:
#model_path = r'C:\Users\aamih\Downloads\BoeingUAV1\BoeingUAVPhase2\pretrained_models\ssd_mobilenet_v2_coco\ssd_mobilenet_v2_coco_2018_03_29\saved_model'
#detect_objects_with_realsense(model_path)
