This project uses YOLOv4 and MediaPipe for object and face detection, and MySQL for event logging. Follow the steps below to get started.

Install the necessary Python packages:
pip install opencv-python numpy flask mediapipe 

You need the following files to run YOLOv4:
1. YOLOv4 weights - https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
2. YOLOv4 config file - [Download yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg)

Move both the files(YOLOv4 weights and YOLOv4 config file into a "models" directory
