# Head Detection

## How to use:

1. download the weights and cfg files from following website: https://pjreddie.com/darknet/yolo/ and copy them into the head_detection/Training/src/keras_yolo3/ folder
2. run the convert.py script in head_detection/Training/src/keras_yolo3/ using 

```
python convert.py yolov3.cfg yolov3.weights yolo.h5
```

3. run head_detection/Data/prepare.ipynb to prepare the data

+ loads the images from openimages
+ transforms the annotation file
+ splits the the data into train/test 

4. configure the run

+ Configure following in the Train_YOLO.py, Detector.py and Evaluate.ipynb: 

  + set if you want to use histogram equalization (useHistogramEqualisation = True/False and use the second definition for file suffix)
  + set if you want to use median Filter (useMedianFilter = True/False and use the second definition for file suffix)

5. run head_detection/Training/Train_YOLO.py to train the model

+ runs both train stages on the data
+ saves the trained weights and the train history

6. run head_detection/Testing/Detector.py to test the model 

+ applies the model onto the test data
+ creates result file
+ saves images with bounding boxes

7. run head_detection/Evaluate/Evaluate.ipynb to evaluate the model 

+ calculates IUO, nZIR and recall for the test data
+ prints the diagram of the history


