# Team: OGP_team - Project 2 - Road Segmentation

Project2 - CS-433 - Machine Learning, EPFL (Fall 2021)

Team membbers of "OGP_team":

- Ottander Gustav Olle Christofilus ([gustav.ottander@epfl.ch](mailto:gustav.ottander@epfl.ch))
- Karlbom Karl Johan Gustav ([karl.karlbom@epfl.ch](mailto:karl.karlbom@epfl.ch))
- Celada Paolo ([paolo.celada@epfl.ch](mailto:paolo.celada@epfl.ch))

## Abstract
Image semantic segmentation is the process of assigning a label to each pixel in an image. It has applications in various different fields, from _medical images_ to _object detection_ and _task recognition_. Using a set of aerial images from urban areas, the aim of this project is to build a classifier able to perform such semantic segmentation, by assigning a label (road=1, background=0) to each pixel. After initial research, where multiple models were considered, a CNN network architecture called U-net was chosen, since it was the best suitable choice for the task. The U-net model is supported by preprocessing, specifically image augmentation, normalization and post-processing. The training phases considered multiple parameters and implementation choices. Once a valid model is setup, predictions on a test set are calculated and submitted to the online platform \textit{AIcrowd}~\cite{AIcrowd-RoadSegmentation}, which calculates both F1 score and accuracy.

## Setup and run
To get a hold of the dataset download the files test_set_images.zip and training.zip [HERE](https://drive.google.com/drive/folders/18yEsYBXJWcZhCC3L9hM5ipafgEFkkDxM). Unzip the folders, and place them in a folder 'data'.

Before running the file, make sure to install the packages below. 

import os
import matplotlib.image as mpimg
import cv2
import os 
import re
import skimage.io as io
from tqdm import tqdm
import torch
import torchvision
import tensorflow as tf
import keras
import keras.backend as K
import sys

We will add a requirements.txt that will be runable, later to the repo, if possible . 

To obtain the best result at AIcrowd, run, run.py

### File structure:
The main folder should have the following structure:

```
-- main_folder/
	- data/
		- test_set_images/
		- training/
    
  - weights/
    - model-unet-msl-black-8features.h5
     
  - output
    - submission_msl_black_8features.csv

  - run.py 
  - main_notebook.ipynb
  - data_postprocessing.py
  - save_output.py
  - image_augmentation.py
  - model.py
  - loss_functions.py
  - data_loading.py
  - metrics.py
  - data_prep.py
  - data_handling.py
  - requirements.txt

