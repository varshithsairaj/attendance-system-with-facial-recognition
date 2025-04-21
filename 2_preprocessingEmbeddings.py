from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# Define dataset and model paths
dataset = r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\dataset"
embeddingFile = r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\output\embeddings.pickle"
embeddingModel = r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\model\nn4.small2.v1.t7"

# YOLO model paths
cfg_path = r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\model\yolov3.cfg"
weights_path = r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\model\yolov3.weights"

# Check if model files exist
if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
    print("Error: YOLO configuration or weights file not found!")
    exit()

# Load YOLO model for face detection
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()

# Get output layers safely (compatible with OpenCV versions)
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

print("YOLO model loaded successfully!")

# Load embedding model
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Get image paths
imagePaths = list(paths.list_images(dataset))

# Initialization
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.5

# Process images one by one
for (i, imagePath) in enumerate(imagePaths):
    print(f"Processing image {i + 1}/{len(imagePaths)}")
    name = os.path.basename(os.path.dirname(imagePath))

    image = cv2.imread(imagePath)
    if image is None:
        print(f"Error: Unable to load image {imagePath}. Skipping...")
        continue

    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Detecting faces using YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Bounding box parameters
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf and class_id == 0:  # Class 0 = person in COCO dataset
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                w_box = int(detection[2] * w)
                h_box = int(detection[3] * h)
                x = int(center_x - w_box / 2)
                y = int(center_y - h_box / 2)

                boxes.append([x, y, w_box, h_box])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w_box, h_box = boxes[i]

            # Validate bounding box
            if x < 0 or y < 0 or w_box <= 0 or h_box <= 0 or x + w_box > w or y + h_box > h:
                print("Invalid bounding box. Skipping...")
                continue

            # Extract face region
            face = image[y:y + h_box, x:x + w_box]
            (fH, fW) = face.shape[:2]

            # Ensure valid face region
            if fW < 20 or fH < 20:
                print("Face too small. Skipping...")
                continue
            if face.size == 0:
                print("Empty face region. Skipping...")
                continue

            # Get face embeddings
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Store embeddings and names
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

# Save embeddings
print(f"Total embeddings: {total}")
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(embeddingFile, "wb") as f:
    pickle.dump(data, f)

print("Process Completed ✅")
