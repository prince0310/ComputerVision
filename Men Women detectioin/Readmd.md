### Custom yolo v8 for men and women detction

##### Use a trained YOLOv8n model to run predictions on images.

###### Down load the weights and reqired file 

``` git clone ```


###### Install ultralytics
``` pip install ultralytics```

###### Python 
```
from ultralytics import YOLO

# Load a model
model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("path of image or video", save = True)  # predict on an image
```
###### For cli 
``` yolo detect predict model=path/to/best.pt source="path of image or video"  # predict with custom model ```
