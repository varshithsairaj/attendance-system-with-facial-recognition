# attendance-system-with-facial-recognition
🎓 Smart Attendance System using Real-Time Face Recognition
“Unlocking Identity with Just a Glance.”

I’m thrilled to share the development of an AI-powered Smart Attendance System designed specifically for educational institutions and secure environments.

🔧 Technologies Used
Python – Scripting & backend logic

OpenCV + YOLOv3 – Real-time object and face detection

yolov3.weights, yolov3.cfg, coco.names

OpenFace – Deep facial embeddings generation

nn4.small2.v1.t7

SVM (Support Vector Machine) – Face classification and recognition

Flask-SocketIO – Real-time event communication

MySQL – Persistent storage for student data and attendance logs

Pickle – Model serialization

embeddings.pickle, le.pickle, recognizer.pickle

Imutils – Efficient video frame preprocessing

🧠 Core Features
✅ Face Registration – Capture 81 high-quality face images per student via webcam
✅ Real-Time Recognition – Detects faces using YOLOv3 and recognizes them using OpenFace + SVM
✅ Automated Attendance – Marks attendance in the database when confidence score exceeds 60%
✅ Live Updates – Real-time logs, alerts, and events through Flask-SocketIO
✅ Smart Summary – Maintains stats like total days present, attendance percentage, and logs
✅ Model Retraining – Automatically retrains SVM model on every new face registration

📂 Database Integration (MySQL)
Student Table: Name, Roll Number, Registration info

Attendance Table: Timestamped records per session

Summary Table: Daily presence logs, calculated percentages

💡 Why It Matters
Manual attendance systems are time-consuming and error-prone. This AI-powered system blends Computer Vision, Machine Learning, and Real-Time Communication to deliver a scalable, secure, and automated solution — ideal for modern educational campuses and institutions looking to embrace smart technologies.


