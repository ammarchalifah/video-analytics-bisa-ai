# faster-rcnn-implementation
Object detection inference using Faster RCNN model. Functions modified from Tensorflow Object Detection API.

# Object Detection with Tensorflow API
The code and models used in this repository is from Google's Tensorflow Object Detection API, click <a href = https://github.com/tensorflow/models/tree/master/research/object_detection> the link for reference </a>.

## Requirements
  - matplotlib
  - numpy
  - opencv 3.4.2
  - tensorflow 1.14
  - PIL
  - imutils
 
## How to use
After activate the virtual environment with all the required packages installed, run the program by type this line of code in your terminal.
```
python detection_photo.py -m faster_rcnn_resnet50_coco_2018_01_28 -i cat.jpeg
```
faster_rcnn_resnet50_coco_2018_01_28 is the name of the model. You can download the model from Tensorflow 1 Model Zoo by clicking this <a href = 'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md'> link. </a>
Download the relevant model (in this case, faster_rcnn_resnet50_coco) by clicking the link. Then, you can unzip the file and copy the directory to `/models`.

<br>cat.jpeg is the image filename of the input.
