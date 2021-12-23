from data_handling import *
import numpy as np
import os
import matplotlib.image as mpimg
import cv2

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]

# load images traversing directory (normalized)
def load_images_traversing_folder(folder, grayscale):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename == ".ipynb_checkpoints":
          continue

        print('Loading ' + folder + filename)
        
        # load differently if grayscale
        if grayscale:
          img = cv2.imread(os.path.join(folder, filename), 0)
        else:
          img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# load images from filename (normalized)
def load_images(filename, num_images, grayscale, augmentation=False):
  imgs = []
  for i in range(1, num_images+1):
      imageid = "satImage_%.3d" % i
      image_filename = filename + imageid + ".png"
      if os.path.isfile(image_filename):
          print('Loading ' + image_filename)
          if augmentation:
            img = mpimg.imread(image_filename)
          elif grayscale:
            img = cv2.imread(image_filename, 0)
          else:
            img = cv2.imread(image_filename)
          imgs.append(img)
      else:
          print('File ' + image_filename + ' does not exist')
  return imgs

# Extract data images
def extract_data(filename, filename_aug, num_images, load_augmented):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    Result shape = (12500, 16, 16, 3)
    """
    # load images (original)
    imgs = load_images(filename, num_images, False)

    if load_augmented:
      # load images (augmented)
      imgs += load_images_traversing_folder(filename_aug, False)

    # num_images = len(imgs)
    # IMG_WIDTH = imgs[0].shape[0]
    # IMG_HEIGHT = imgs[0].shape[1]
    # N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)
# 
    # img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    # data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    #return np.asarray(data)
    # images in shape (num_images, 400, 400, 3)
    return np.asarray(imgs)

# Extract label images
def extract_labels(filename, filename_aug, num_images, load_augmented):
    """Extract the labels into a 1-hot matrix [image index, label index].
    Result shape = (12500, 2)
    """
    # load gt-images (original)
    gt_imgs = load_images(filename, num_images, True)

    if load_augmented:
      # load gt-images (augmented)
      gt_imgs += load_images_traversing_folder(filename_aug, True)

    # num_images = len(gt_imgs)
    # gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    # data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    # labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    # return labels.astype(np.float32)
    # need to have gt-images in shape (num_images, 400, 400, 1)
    return np.expand_dims(np.asarray(gt_imgs), axis=3)



# Extract test data images
def extract_test_data(filename, num_images):
  imgs = load_test_images(filename, num_images, False)
  return np.asarray(imgs)


# load test images from filename
def load_test_images(filename, num_images, grayscale):
  imgs = []
  for i in range(1, num_images+1):
      imageid = "test_%d/test_%d" % (i, i)
      image_filename = filename + imageid + ".png"
      if os.path.isfile(image_filename):
          print('Loading ' + image_filename)
          if grayscale:
            img = cv2.imread(image_filename, 0)
            img_splitted = split_test_image(img)  
          else:
            img = cv2.imread(image_filename)
            img_splitted = split_test_image(img)
          imgs.append(img_splitted)
      else:
          print('File ' + image_filename + ' does not exist')
  return imgs

# splitting 608x608 into 4 400x400 images
def split_test_image(img):
    imgs=[]
    img11=img[0:400, 0:400, :]
    img21=img[208:608, 0:400]
    img12=img[0:400, 208:608]
    img22 = img[208:608, 208:608]
    imgs.append(img11)
    imgs.append(img21)
    imgs.append(img12)
    imgs.append(img22)
    return imgs
