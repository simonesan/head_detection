"""
MODIFIED FROM https://github.com/AntonMu/TrainYourOwnYOLO
"""

import os
import sys


def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)

import argparse
from keras_yolo3.yolo import YOLO
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object, GetFileList
import utils
import pandas as pd
import numpy as np
import random
from keras_yolo3.train import get_anchors

# --------------------------------------------------------------------------------------

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
useMedianFilter = True
useHistogramEqualisation = False

#fileSuffix = '_noAug'
fileSuffix = '_Median_'+str(useMedianFilter) + 'HistogramEq_'+str(useHistogramEqualisation)

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results"+ fileSuffix + ".csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "trained_weights_final"+ fileSuffix + ".h5")
model_classes = os.path.join(model_folder, "data_classes.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

# --------------------------------------------------------------------------------------

save_img = True

input_paths = GetFileList(image_test_folder)


# Split images and videos
img_endings = (".jpg", ".jpeg", ".png")

input_image_paths = []
for item in input_paths:
    if item.endswith(img_endings):
        input_image_paths.append(item)

output_path = detection_results_folder
if not os.path.exists(output_path):
    os.makedirs(output_path)


anchors = get_anchors(anchors_path)
# define YOLO detector
yolo = YOLO(
    **{
        "model_path": model_weights,
        "anchors_path": anchors_path,
        "classes_path": model_classes,
        "score": 0.25,
        "gpu_num": 1,
        "model_image_size": (416, 416),
    }
)

# Make a dataframe for the prediction outputs
out_df = pd.DataFrame(
    columns=[
        "image",
        "image_path",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "label",
        "confidence",
        "x_size",
        "y_size",
    ]
)

# labels to draw on images
class_file = open(model_classes, "r")
input_labels = [line.rstrip("\n") for line in class_file.readlines()]
print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

if input_image_paths:
    print(
        "Found {} input images: {} ...".format(
            len(input_image_paths),
            [os.path.basename(f) for f in input_image_paths[:5]],
        )
    )
    start = timer()
    text_out = ""

    # This is for images
    for i, img_path in enumerate(input_image_paths):
        print(img_path)
        prediction, image = detect_object(
            yolo,
            img_path,
            save_img=save_img,
            save_img_path=detection_results_folder,
            postfix="_head",
            useMedianFilter = useMedianFilter, 
            useHistogramEqualisation = useHistogramEqualisation,
            flip = False
        )
        predictionFlip, imageFlip = detect_object(
            yolo,
            img_path,
            save_img=save_img,
            save_img_path=detection_results_folder,
            postfix="_head",
            useMedianFilter = useMedianFilter, 
            useHistogramEqualisation = useHistogramEqualisation,
            flip = True
        )
        y_size, x_size, _ = np.array(image).shape
        for single_prediction in prediction:
            out_df = out_df.append(
                pd.DataFrame(
                    [
                        [
                            os.path.basename(img_path.rstrip("\n")),
                            img_path.rstrip("\n"),
                        ]
                        + single_prediction
                        + [x_size, y_size]
                    ],
                    columns=[
                        "image",
                        "image_path",
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "label",
                        "confidence",
                        "x_size",
                        "y_size",
                    ],
                )
            )
    end = timer()
    print(
        "Processed {} images in {:.1f}sec - {:.1f}FPS".format(
            len(input_image_paths),
            end - start,
            len(input_image_paths) / (end - start),
        )
    )
    out_df.to_csv(detection_results_file, index=False)

# Close the current yolo session
yolo.close_session()
