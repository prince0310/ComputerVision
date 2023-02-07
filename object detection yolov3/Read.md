# object-detection
Object detection using YoloV3
 OpenCV dnn module supports running inference on pre-trained deep learning models from popular frameworks like Caffe, Torch and TensorFlow.
### When it comes to object detection, popular detection frameworks are
- YOLO
- SSD
- Faster R-CNN
### Support for running YOLO/DarkNet has been added to OpenCV dnn module recently.

# Dependencies
- opencv
- numpy

pip install numpy opencv-python

# YOLO (You Only Look Once)
## Download the pre-trained YOLO v3 weights file from this link and place it in the current directory or you can directly download to the current directory in terminal using
$ wget https://pjreddie.com/media/files/yolov3.weights

Provided all the files are in the current directory, below command will apply object detection on the input image

$ python yolo_opencv.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
## Sample Input
![car-names-for-dogs](https://user-images.githubusercontent.com/85225054/169530150-d68ee7e1-bd71-4f70-be87-12fa8ea186c2.jpg)
## Sample output
![object detection_screenshot_20 05 2022](https://user-images.githubusercontent.com/85225054/169530358-ef258664-86ac-416b-a819-b4f8c51b3233.png)

# Refrences
- arunponnusamy
- https://www.visiongeek.io/2018/07/yolo-object-detection-opencv-python.html
