import os 
import numpy as np 
import re
import tensorflow as tf
import skimage.io as io
from tqdm import tqdm

round_treshold = 0.75

def save_output(prediction, file_path):
  for i, pred in enumerate(prediction):
    pred[pred>0.5] = 1
    pred[pred<=0.5]=0
    #image = tf.dtypes.cast(pred, tf.uint8)
    pred_np = tf.make_ndarray(pred)
    image = (pred_np*255).astype('uint8')
    io.imsave(os.path.join(file_path, "%d_predict.png"%(i+1)), image)

# perform prediction on test data (608x608x3)
# method:
# - receive input image (608x608x3) each splitten in 4 patches of 400x400x3
# - perform prediction for all image patches
# - reconstruct mask (608x608x1) from mask images patches (400x400x1)
# - return final mask
def model_predict(model, data, new_size):
  predictions = []

  # loop for all test data images
  for test_img in tqdm(data):
    test_img = test_img / 255.0
    img = np.zeros((new_size, new_size,1))
    # predict on patch images
    pred = model.predict(test_img)
    
    # round for greyscale image
    pred = (pred >= round_treshold).astype(np.uint8)

    # assign already label to each patches
    patch_size = 16 
    for index, im in enumerate(pred):
        for j in range(0, im.shape[1], patch_size): 
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch)
                im[i:i + patch_size, j:j + patch_size] = label
        
        # apply to original image
        pred[index] = im


    # gap betweem new size and old size
    gap = pred[0].shape[0] - (new_size - pred[0].shape[0]) # should be 192

    # reconstruct image prediction from patches
    # top-left (400x400)
    img[0:pred[0].shape[0], 0:pred[0].shape[0], :] = pred[0]
    # bottom-left (208x400)
    img[pred[0].shape[0]:, 0:pred[0].shape[0], :] = pred[1][gap:, 0:pred[0].shape[0], :]
    # top-right (400x208)
    img[0:pred[0].shape[0], pred[0].shape[0]:new_size, :] = pred[2][0:pred[0].shape[0], gap:, :]
    # bottom-right (208x208)
    img[pred[0].shape[0]:new_size, pred[0].shape[0]:new_size, :] = pred[3][gap:, gap:, :]

    predictions.append(img)

  return predictions

 
foreground_threshold = 0.5 # percentage of pixels > 1 required to assign a foreground label to a patch 
 
# assign a label to a patch 
def patch_to_label(patch): 
    df = np.mean(patch) 
    if df > foreground_threshold: 
        return 1 
    else: 
        return 0 
 
# generate single submission file
def mask_to_submission_strings(image_prediction, img_number): 
    """Reads a single image and outputs the strings that should go into the submission file""" 
    im = image_prediction 
    im = np.moveaxis(im,-1,0)[0] # can skip
    patch_size = 16 
    for j in range(0, im.shape[1], patch_size): 
        for i in range(0, im.shape[0], patch_size): 
            patch = im[i:i + patch_size, j:j + patch_size] 
            label = patch_to_label(patch) 
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label)) 
 
# generate submission file from prediction
def masks_to_submission_outer(submission_filename, predictions): 
    """Converts images into a submission file""" 
    with open(submission_filename, 'w') as f: 
        f.write('id,prediction\n') 
        print(len(predictions))
        for fn in range(len(predictions)): 
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(predictions[fn], fn+1))