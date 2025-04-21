
import sys
import cv2
import numpy as np
import os
import time
import datetime
import mysql.connector
import imutils
import pickle
import socketio

# Connect to the Flask-SocketIO server (adjust host/port if needed)
sio = socketio.Client()
try:
    sio.connect('http://localhost:5000')
except socketio.exceptions.ConnectionError as e:
    print("⚠️ Could not connect to Flask server:", e)
    sio = None


print("✅ Script started with args:", sys.argv)


# Establish a connection to the MySQL database
conn = mysql.connector.connect(
    host="127.0.0.1",
    user='root',
    password='password@123',
    database='attendance'
)

cursor = conn.cursor()

def detect_face(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    faces = []
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            # Ensure the bounding box is within the frame dimensions
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                print("Invalid bounding box. Skipping...")
                continue

            # Ensure the face is not too small
            if w < 20 or h < 20:
                print("Face too small. Skipping...")
                continue

            # Extract face region
            face = frame[y:y+h, x:x+w]

            # Ensure face is not empty
            if face.size == 0:
                print("Empty face region. Skipping...")
                continue

            faces.append(face)

    return faces

def save_face_data(name, face_data):
    dataset = 'dataset'
    path = os.path.join(dataset, name)
    if not os.path.isdir(path):
        os.makedirs(path)
    
    for idx, face in enumerate(face_data):
        filename = os.path.join(path, f"{str(idx).zfill(5)}.png")
        cv2.imwrite(filename, face)

def register_new_student(name, roll_number):
    dataset = 'dataset'
    path = os.path.join(dataset, name)
    if not os.path.isdir(path):
        os.makedirs(path)

    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Error: Could not open camera. Please check the camera connection.")
        return
    
    time.sleep(2.0)
    
    total = 0
    print("Capturing face data. Please look at the camera...")

    while total < 81:
        success, frame = cam.read()

        if not success or frame is None:
            print("Error: Failed to grab frame from camera. Retrying...")
            time.sleep(0.1)
            continue
        
        faces = detect_face(frame)
        
        for face in faces:
            if total >= 81:
                break
            
            filename = os.path.join(path, f"{str(total).zfill(5)}.png")
            cv2.imwrite(filename, face)
            total += 1
            print(f"Captured {total}/81")

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Capture stopped by user.")
            break

    cam.release()
    cv2.destroyAllWindows()

    if total >= 81:
        cursor.execute("INSERT INTO students (name, roll_number) VALUES (%s, %s)", (name, roll_number))
        conn.commit()
        print(f"Student '{name}' registered successfully with {total} face images.")
        retrain_model()
    else:
        print("Face capture incomplete. Registration failed.")


def retrain_model():
    import subprocess
    print("Retraining model...")
    subprocess.run(['python', '2_preprocessingEmbeddings.py'])
    subprocess.run(['python', '3_trainingFaceML.py'])
    print("Model retrained successfully.")

def is_registration_time():
    now = datetime.datetime.now().time()
    start_time = datetime.time(8, 0)
    end_time = datetime.time(23, 0)
    return start_time <= now <= end_time

dataset_path = "dataset"
known_face_encodings = []
known_face_names = []

def update_attendance_summary(student_id):
    try:
        # Get total days this student has attended at least once
        cursor.execute("SELECT COUNT(DISTINCT DATE(timestamp)) FROM attendance WHERE student_id = %s", (student_id,))
        total_days = cursor.fetchone()[0] or 0

        # Get total days present for this student
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE student_id = %s", (student_id,))
        total_present = cursor.fetchone()[0] or 0

        # Calculate attendance percentage
        attendance_percentage = 0 if total_days == 0 else (total_present / total_days) * 100


        # Insert or update attendance_summary
        cursor.execute("""
            INSERT INTO attendance_summary (student_id, total_present, total_days, attendance_percentage, last_updated)
            VALUES (%s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE 
                total_present = VALUES(total_present),
                total_days = VALUES(total_days),
                attendance_percentage = VALUES(attendance_percentage),
                last_updated = VALUES(last_updated)
        """, (student_id, total_present, total_days, attendance_percentage))

        conn.commit()
        print(f"📊 Attendance updated: {total_present}/{total_days} days ({attendance_percentage:.2f}%)")

    except mysql.connector.Error as err:
        print(f"❌ Database Error: {err}")
        conn.rollback()

        ############

embedder = cv2.dnn.readNetFromTorch(r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\model\nn4.small2.v1.t7")



# Base directory of your project
BASE_DIR = r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master"

# Absolute paths to model files
recognizer_path = os.path.join(BASE_DIR, "output", "recognizer.pickle")
label_encoder_path = os.path.join(BASE_DIR, "output", "le.pickle")

# Load the trained recognizer (SVM Model) and Label Encoder
recognizer = pickle.load(open(recognizer_path, "rb"))
le = pickle.load(open(label_encoder_path, "rb"))

# Load YOLO model for face detection
cfg_path = os.path.join(BASE_DIR, "model", "yolov3.cfg")
weights_path = os.path.join(BASE_DIR, "model", "yolov3.weights")
net = cv2.dnn.readNet(weights_path, cfg_path)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize globals
known_face_encodings = []
known_face_names = []

def load_known_faces():
    print("📂 Loading known faces...")

    global known_face_encodings, known_face_names

    try:
        # Use absolute paths here too
        recognizer = pickle.load(open(recognizer_path, "rb"))
        le = pickle.load(open(label_encoder_path, "rb"))

        # Get label names
        known_face_names = list(le.classes_)
        known_face_encodings = recognizer

        print(f"✅ Loaded {len(known_face_names)} known faces.")

    except Exception as e:
        print(f"❌ Error loading known faces: {e}")
        known_face_encodings = []
        known_face_names = []

def recognize_and_mark_attendance():
    print("Starting face recognition for attendance marking...")
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = cam.read()
        if not success:
            print("Error: Failed to capture frame from webcam!")
            continue

        frame = imutils.resize(frame, width=600)
        height, width, _ = frame.shape

        # Detect faces using YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_id == 0:  # Class 0 is "person"
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])

        if not boxes:
            print("No faces detected. Trying again...")
            continue

        for x, y, w, h in boxes:
            face = frame[y:y+h, x:x+w]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                print("Detected face is too small. Skipping...")
                continue

            # Extract face embedding using OpenFace
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Perform face recognition
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            if proba < 0.6:  # Confidence threshold for recognition
                print("Face not recognized. Please try again.")
                continue

            # Fetch student ID from the database
            cursor.execute("SELECT id FROM students WHERE name = %s", (name,))
            result = cursor.fetchone()

            if not result:
                print(f"Student {name} not found in database.")
                continue

            student_id = result[0]
            now = datetime.datetime.now()
            date_today = now.strftime('%Y-%m-%d')
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

            # Check if attendance is already marked today
            cursor.execute("SELECT COUNT(*) FROM attendance WHERE student_id = %s AND DATE(timestamp) = %s", 
                           (student_id, date_today))
            attendance_count = cursor.fetchone()[0]

            if attendance_count > 0:
                print(f"Attendance already marked for {name} today.")
            else:
                try:
                    cursor.execute("INSERT INTO attendance (student_id, timestamp) VALUES (%s, %s)", 
                                   (student_id, timestamp))
                    conn.commit()
                    print(f"✅ Welcome, {name}! Attendance marked at {timestamp}")
                    update_attendance_summary(student_id)  
                    
                except mysql.connector.Error as err:
                    print(f"❌ Database Error: {err}")
                    conn.rollback()

            # Display the recognized name on the frame
            text = f"{name} : {proba * 100:.2f}%"
            y_text = y - 10 if y - 10 > 10 else y + 10
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cam.release()
            cv2.destroyAllWindows()
            return  # Exit function after recognizing a student

        # Show camera feed
        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Face recognition stopped.")


def main():
    load_known_faces()
    mode = input("Enter 'login' to mark attendance or 'register' to add a new student: ").strip().lower()
    
    if mode == 'login':
        recognize_and_mark_attendance()
    
    elif mode == 'register':
        if not is_registration_time():
            print("Registration is only allowed between 8 AM and 5 PM.")
            return
        
        name = input("Enter the new student's name: ").strip()
        roll_number = input("Enter the new student's roll number: ").strip()

        register_new_student(name, roll_number)
    
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":

    main()






























# import cv2
# import numpy as np
# import os
# import time
# import datetime
# import csv

# # Load YOLO model for face detection
# net = cv2.dnn.readNet(
#     r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\model\yolov3.weights",
#     r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\model\yolov3.cfg"
# )

# layer_names = net.getLayerNames()

# # Fix for OpenCV versions where getUnconnectedOutLayers() returns a list of integers
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Function to check if current time is within registration hours
# # def is_registration_time():
# #     now = datetime.datetime.now().time()  # Get the current time
# #     start_time = datetime.time(9, 0)      # Registration starts at 9 AM
# #     end_time = datetime.time(17, 0)       # Registration ends at 5 PM
# #     return start_time <= now <= end_time  # Returns True if current time is within this range


# Name = str(input("Enter your Name: "))
# Roll_Number = int(input("Enter your Roll_Number: "))

# # if not is_registration_time():
# #     print("Registration is only allowed between 9 AM and 5 PM.")
# #     exit()
# # else:
# #     print("Registration is allowed. Proceeding...")

# dataset = r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\dataset"

# sub_data = Name
# path = os.path.join(dataset, sub_data)

# if not os.path.isdir(path):
#     os.mkdir(path)
#     print(sub_data)

# info = [str(Name), str(Roll_Number)]
# with open('student.csv', 'a') as csvFile:
#     write = csv.writer(csvFile)
#     write.writerow(info)
# csvFile.close()

# print("Starting video stream...")
# cam = cv2.VideoCapture(0)
# time.sleep(2.0)
# total = 0

# while total < 81:
#     print(total)
#     _, frame = cam.read()
#     height, width, channels = frame.shape

#     # Detecting objects using YOLO
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     # Information to draw bounding boxes
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5 and class_id == 0:  # Class 0 is person in COCO dataset
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]

#             # Ensure the bounding box is within the frame dimensions
#             if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
#                 print("Invalid bounding box. Skipping...")
#                 continue

#             # Ensure the detected face is not too small
#             if w < 20 or h < 20:
#                 print("Face too small. Skipping...")
#                 continue

#             # Extract the face region
#             face = frame[y:y+h, x:x+w]

#             # Ensure the face region is not empty
#             if face.size == 0:
#                 print("Empty face region. Skipping...")
#                 continue

#             # Save the face image
#             p = os.path.sep.join([path, "{}.png".format(str(total).zfill(5))])
#             cv2.imwrite(p, face)
#             total += 1

#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

# cam.release()
# cv2.destroyAllWindows()

