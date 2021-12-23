import numpy
import keras
import keras.backend as K

# these losses have been chosen to deal with the unbalanced dataset problem
# (https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56)

"""Jaccard/Intersection over Union (IoU) Loss"""
def IoULoss(inputs, targets, smooth=1):
  inputs = K.cast(inputs, 'float32')

  intersection = K.sum(K.abs(targets * inputs), axis=-1)
  union = K.sum(targets,-1) + K.sum(inputs,-1) - intersection
  iou = (intersection + smooth) / ( union + smooth)
  return iou

"""Focal Loss"""
ALPHA = 0.8
GAMMA = 2

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    
  targets = K.cast(targets, 'float32')

  inputs = K.clip(inputs, 1e-3, .999)
  targets = K.clip(targets, 1e-3, .999)
  
  BCE = K.binary_crossentropy(targets, inputs)
  BCE_EXP = K.exp(-BCE)
  focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
  
  return focal_loss