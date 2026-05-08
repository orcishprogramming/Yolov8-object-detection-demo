\# YOLOv8 Phone Camera Object Detector



A beginner-friendly computer vision project that uses an iPhone camera stream with OpenCV and YOLOv8 for real-time object detection.



\## Features



\- Real-time object detection using YOLOv8

\- iPhone camera used as an IP camera through DroidCam

\- OpenCV video stream processing

\- Bounding boxes and class labels

\- FPS counter

\- Object counter

\- Screenshot saving



\## Technologies Used



\- Python

\- OpenCV

\- YOLOv8

\- Ultralytics

\- DroidCam



\## How It Works



The iPhone streams video over Wi-Fi using DroidCam.  

OpenCV reads the stream using `cv2.VideoCapture()`.  

YOLOv8 performs object detection on each frame.  

The detected objects are displayed with bounding boxes and labels.




