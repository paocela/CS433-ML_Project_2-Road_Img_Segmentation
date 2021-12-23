from keras import backend as K
import numpy as np

# note: use of K.epsilon to prevent division by zero error (K.epsilon = 1e-07 in TensorFlow Core v2.2.0)
# K.round = element-wise round to closest integer
# K.clip = element-wise clipping operation

# recall metric
# (TP) / (TP + FN)
def recall_metric(y_true, y_pred):
  tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  tp_fn = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = tp / (tp_fn + K.epsilon())
  return recall

# precision metric
# (TP) / (TP + FP)
def precision_metric(y_true, y_pred):
  tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  tp_fp = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = tp / (tp_fp + K.epsilon())
  return precision

# f1 score metric
# (2 * precision * recall) / (precision + recall)
def f1_metric(y_true, y_pred):
  recall = recall_metric(y_true, y_pred)
  precision = precision_metric(y_true, y_pred)

  f1 = (2 * precision * recall) / (precision + recall + K.epsilon())
  return f1
