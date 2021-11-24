import tensorflow as tf


def flip(data):

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.5),])   

temp = data
for i in range(temp):
  i = data_augmentation(i)

return temp 


