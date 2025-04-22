## 🎓 Smart Attendance System using Real-Time Face Recognition - “Unlocking Identity with Just a Glance.”
 
I’m thrilled to share the development of an AI-powered Smart Attendance System designed for educational institutions and security environments.

## 🔧 Technologies Used:
- Python for scripting & backend logic
- OpenCV + YOLOv3 (yolov3.weights, yolov3.cfg, coco.names) for object and face detection
- OpenFace (nn4.small2.v1.t7) for generating deep facial embeddings
- SVM (Support Vector Machine) for accurate face recognition
- Flask-SocketIO for real-time event communication
- MySQL for storing student details and attendance logs
- Pickle (embeddings.pickle, le.pickle, recognizer.pickle) for model serialization
- Imutils for efficient frame processing


 ## 🧠 Core Features

- ✅ **Face Registration**: Capture 81 high-quality face images per student through webcam  
- ✅ **Real-Time Recognition**: Detects faces using YOLOv3 and recognizes them with OpenFace + SVM  
- ✅ **Automated Attendance**: Marks attendance in the DB if recognition confidence > 60%  
- ✅ **Live Updates**: Uses Flask-SocketIO to send logs, alerts, and attendance events in real time  
- ✅ **Smart Summary**: Maintains daily presence stats, percentages, and history  
- ✅ **Model Retraining**: Automatically retrains SVM when new faces are registered  


## 📂 Database Integration (MySql):
Student info: Name, Roll No, Registration
Attendance: Timestamped records
Summary Table: Days present, percentage stats


## 💡 Why It Matters:
 Manual attendance is time-consuming and error-prone. This system combines Computer Vision, ML, and Real-Time Communication to create a secure, efficient, and scalable attendance and surveillance solution — ideal for educational institutes and smart campuses.

 ![Screenshot 2025-04-20 225335](https://github.com/user-attachments/assets/78e6b9cf-5f0a-40f9-8395-c35027eb5754) ![Screenshot 2025-04-20 205649](https://github.com/user-attachments/assets/e12381f9-a3c5-415e-b45f-dc7d910dccf9)

