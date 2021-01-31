# Head Detection

How to use:
1. download the weights and cfg files from following website: https://pjreddie.com/darknet/yolo/ and copy them into the head_detection/Training/src/keras_yolo3/ folder
2. run the convert.py script in head_detection/Training/src/keras_yolo3/ using 

```
python convert.py yolov3.cfg yolov3.weights yolo.h5
```

3. run head_detection/Data/prepere.ipynb to prepare the data
4. run head_detection/Training/Train_YOLO.py to train the model
5. run head_detection/Testing/Detector.py to test the model 
6. run head_detection/Evaluate/Evaluate.py to evaluate the model 



