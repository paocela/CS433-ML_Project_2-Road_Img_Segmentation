from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import matplotlib.image as mpl_image
import cv2

from data_loading import load_images

# seed definition (to provide same random augmentation to both image and mask)
seed = 401
rotation_range = 180
brightness_range=[0.2,1.0]
zoom_range=[0.5,1.0]
width_shift_range=0.2
height_shift_range=0.2

# Outer function peforming all image augmentation tasks (each can be disabled)
def image_augmentation(train_datas_filename, train_labels_filename, train_datas_aug_filename, train_labels_aug_filename, num_images, num_aug_flip, num_aug_rotation, num_aug_zoom_brightness, num_aug_shift, horizontal_vertical_flip=True, rotation=True, brightness=True, zoom=True, shift=True):
  # load images
  imgs = load_images(train_datas_filename, num_images, grayscale=False, augmentation=True)
  masks = load_images(train_labels_filename, num_images, grayscale=False, augmentation=True)

  # loop through all images
  for i in range(num_images):
    # convert to numpy array
    imgs_data = img_to_array(imgs[i])
    masks_data = img_to_array(masks[i])

    print("Performing data augmentation on satImage_%.3d..." % (i+1))

    # expand dimension to one sample
    samples_img = expand_dims(imgs_data, 0)
    samples_mask = expand_dims(masks_data, 0)

    # perform different data augmentation techniques to images and masks (same technique to both)

    # random horizontal and vertical flip
    if horizontal_vertical_flip:
      # type augmentation for output filename
      augmentation_type = 0

      # image data augmentation generator
      generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
      
      # prepare iterator
      it = generator.flow(samples_img, batch_size=1, seed=seed)
      it_mask = generator.flow(samples_mask, batch_size=1, seed=seed)
      
      # generate samples and plot
      for j in range(num_aug_flip):
        # generate batch of images
        batch = it.next()
        batch_mask = it_mask.next()
        # convert to unsigned integers for viewing
        image = (batch[0] * 255).astype('uint8')
        mask_image = (batch_mask[0] * 255).astype('uint8')

        # save image pair for future use
        store_image(train_datas_aug_filename, image, i, augmentation_type, j, False)
        store_image(train_labels_aug_filename, mask_image, i, augmentation_type, j, True) 
      
    # random rotation + horizontal/vertical flip
    if rotation:
      # type augmentation for output filename
      augmentation_type = 2

      # image data augmentation generator
      generator = ImageDataGenerator(rotation_range=rotation_range, horizontal_flip=True, vertical_flip=True, fill_mode="constant", cval=0)
      
      # prepare iterator
      it = generator.flow(samples_img, batch_size=1, seed=seed)
      it_mask = generator.flow(samples_mask, batch_size=1, seed=seed)
      
      # generate samples and plot
      for j in range(num_aug_rotation):
        # generate batch of images
        batch = it.next()
        batch_mask = it_mask.next()
        # convert to unsigned integers for viewing
        image = (batch[0] * 255).astype('uint8')
        mask_image = (batch_mask[0] * 255).astype('uint8')

        # save image pair for future use
        store_image(train_datas_aug_filename, image, i, augmentation_type, j, False)
        store_image(train_labels_aug_filename, mask_image, i, augmentation_type, j, True)


    # random change of brightness
    # not needed for our case (doesn't make sense to grayscale the mask)
    if brightness:
      # type augmentation for output filename
      augmentation_type = 3

      # image data augmentation generator
      generator = ImageDataGenerator(brightness_range=brightness_range)
      # prepare iterator
      it = generator.flow(samples_img, batch_size=1, seed=seed)
      it_mask = generator.flow(samples_mask, batch_size=1, seed=seed)
      
      # generate samples and plot
      for j in range(num_aug_zoom_brightness):
        # generate batch of images
        batch = it.next()
        batch_mask = it_mask.next()
        # convert to unsigned integers for viewing
        image = (batch[0] * 255).astype('uint8')
        mask_image = (batch_mask[0] * 255).astype('uint8')

        # save image pair for future use
        store_image(train_datas_aug_filename, image, i, augmentation_type, j, False)
        store_image(train_labels_aug_filename, mask_image, i, augmentation_type, j, True)


    # random change of zoom
    # not used as images are taken always from the same height
    if zoom:
      # type augmentation for output filename
      augmentation_type = 4

      # image data augmentation generator
      generator = ImageDataGenerator(zoom_range=zoom_range)

      # prepare iterator
      it = generator.flow(samples_img, batch_size=1, seed=seed)
      it_mask = generator.flow(samples_mask, batch_size=1, seed=seed)
      
      # generate samples and plot
      for j in range(num_aug_zoom_brightness):
        # generate batch of images
        batch = it.next()
        batch_mask = it_mask.next()
        # convert to unsigned integers for viewing
        image = (batch[0] * 255).astype('uint8')
        mask_image = (batch_mask[0] * 255).astype('uint8')

        # save image pair for future use
        store_image(train_datas_aug_filename, image, i, augmentation_type, j, False)
        store_image(train_labels_aug_filename, mask_image, i, augmentation_type, j, True)

    # random change of shift + horizontal/vertical flip
    if shift:
      # type augmentation for output filename
      augmentation_type = 5

      # image data augmentation generator
      generator = ImageDataGenerator(width_shift_range=width_shift_range, height_shift_range=height_shift_range, horizontal_flip=True, vertical_flip=True, fill_mode="constant", cval=0)

      # prepare iterator
      it = generator.flow(samples_img, batch_size=1, seed=seed)
      it_mask = generator.flow(samples_mask, batch_size=1, seed=seed)
      
      # generate samples and plot
      for j in range(num_aug_shift):
        # generate batch of images
        batch = it.next()
        batch_mask = it_mask.next()
        # convert to unsigned integers for viewing
        image = (batch[0] * 255).astype('uint8')
        mask_image = (batch_mask[0] * 255).astype('uint8')

        # save image pair for future use
        store_image(train_datas_aug_filename, image, i, augmentation_type, j, False)
        store_image(train_labels_aug_filename, mask_image, i, augmentation_type, j, True)


def store_image(filename_base, image, image_idx, augmentation_type, augmentation_index, isMask):
  # construct image name
  # example: satImage_030_1_3.png 
  # (image id 30 - augmentation type 1 (vertical flip) - application number 3(starting from 0))
  imageid = "satImage_%.3d_%d_%d" % (image_idx + 1, augmentation_type, augmentation_index)
  filename = filename_base + imageid + ".png"
  if not isMask:
    mpl_image.imsave(filename, image)
  else:
    cv2.imwrite(filename, image)








