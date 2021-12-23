import torch
import torchvision
import tensorflow as tf

""" Down Sample Class"""
""" operations performed:
1. 3x3 CONV (ReLU + Batch Normalization) 
2. 3x3 CONV (ReLU + Batch Normalization) 
3. 2x2 Max Pooling 
4. dropout
Feature maps double as we go down the blocks, starting at 64, then 128, 256, and 512.
After 4 iterations, bottleneck (consists of 2 CONV layers with Batch Normalization & relu)
"""

class DownSample:
    def __init__(self, out_channels, conv_kernel=3, maxpool_kernel=2, maxpool_stride=2, dropout_p=0.1, padding="same"):
      super().__init__()

      # first convolution + batch normalization
      self.conv1 = tf.keras.layers.Conv2D(out_channels, conv_kernel, padding=padding)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      # second convolution + batch normalization
      self.conv2 = tf.keras.layers.Conv2D(out_channels, conv_kernel, padding=padding)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()

      # maxpooling + dropout
      self.maxpool = tf.keras.layers.MaxPooling2D((maxpool_kernel, maxpool_kernel), maxpool_stride)
      self.dropout = tf.keras.layers.Dropout(dropout_p)

    # Method for down sample iteration
    # :param x: input tensor contraction phase
    # :param last: flag for bottleneck
    def forward(self, x, last=False):
      
      # normal case
      if not last:
        # conv 3x3, batchnorm, relu * 2
        x = tf.keras.activations.relu((self.batchnorm1(self.conv1(x))))
        x = tf.keras.activations.relu((self.batchnorm2(self.conv2(x))))
        x_conv = x
        # max pool 2x2
        x = self.maxpool(x)
        x = self.dropout(x)

        return x, x_conv
        
      else:
        # bottleneck
        x = tf.keras.activations.relu((self.batchnorm1(self.conv1(x))))
        x = tf.keras.activations.relu((self.batchnorm2(self.conv2(x))))

        return x


""" Up Sample Class """
""" Operations performed:
1. transposed convolution layer  
2. padding
3. concatenation with the feature map from the corresponding contracting path 
4. dropout
5. 3x3 CONV (ReLU + Batch Normalization) 
6. 3x3 CONV (ReLU + Batch Normalization)
After 4 iterations, output convolution
"""
class UpSample:
    def __init__(self, out_channels, conv_kernel=3, conv_transp_kernel=2, conv_output_kernel=1, dropout_p=0.1, conv_stride=(2,2), padding="same"): 
      super().__init__() 
 
      # transpose convolution
      self.up = tf.keras.layers.Conv2DTranspose(out_channels, conv_transp_kernel, strides=conv_stride, padding=padding)
       
      # first convolution + batch normalization
      self.conv1 = tf.keras.layers.Conv2D(out_channels, conv_kernel, padding=padding)
      self.batchnorm1 = tf.keras.layers.BatchNormalization()

      # second convolution + batch normalization
      self.conv2 = tf.keras.layers.Conv2D(out_channels, conv_kernel, padding=padding)
      self.batchnorm2 = tf.keras.layers.BatchNormalization()
 
      # dropout
      self.dropout = tf.keras.layers.Dropout(dropout_p)

      # concate
      self.concat = tf.keras.layers.Concatenate(axis=3)

      # output convolution
      self.out_conv = tf.keras.layers.Conv2D(out_channels, conv_output_kernel, activation='sigmoid')#, padding=padding

    # Method for up sample iteration
    # :param x: input tensor expantion phase
    # :param x_1: corresponding input tensor contraction phase
    # :param last: flag for output convolution
    def forward(self, x, x_1, last=False): 

      # normal case
      if not last:
        x = self.up(x) 
        
        # copy and crop (padding + concatenation)
        # difference_width = x.shape[2] - x_1.shape[2]
        # difference_height = x.shape[3] - x_1.shape[3]

        # left_padding = difference_width // 2
        # right_padding = difference_width - left_padding
        # top_padding = difference_height // 2
        # bottom_padding = difference_height - top_padding

        # x = tf.pad(x, [[top_padding, bottom_padding], [left_padding, right_padding]])
  
        x = self.concat([x, x_1])

        # dropout
        x = self.dropout(x) 

        # conv 3x3, batchnorm, relu * 2
        x = tf.keras.activations.relu((self.batchnorm1(self.conv1(x))))
        x = tf.keras.activations.relu((self.batchnorm2(self.conv2(x))))
       
      else:
        # output convolution case
        x = self.out_conv(x)
      
      return x 
 
# n_filters is a hyperparameter and can be fine tuned
class Unet:
    def __init__(self, n_classes, n_filters): 
      super().__init__() 

      # 4 down sample operations
      #self.downSample1 = DownSample(64)
      #self.downSample2 = DownSample(128)
      #self.downSample3 = DownSample(256)
      #self.downSample4 = DownSample(512)
      self.downSample1 = DownSample(n_filters)
      self.downSample2 = DownSample(n_filters * 2)
      self.downSample3 = DownSample(n_filters * 4)
      self.downSample4 = DownSample(n_filters * 8)

      # 1 bottleneck
      #self.bottleneck = DownSample(1024)
      self.bottleneck = DownSample(n_filters * 16)

      # 4 up sample operations
      #self.upSample1 = UpSample(512)
      #self.upSample2 = UpSample(256)
      #self.upSample3 = UpSample(128)
      #self.upSample4 = UpSample(64)
      self.upSample1 = UpSample(n_filters * 8)
      self.upSample2 = UpSample(n_filters * 4)
      self.upSample3 = UpSample(n_filters * 2)
      self.upSample4 = UpSample(n_filters)

      # 1 output convolution
      self.outConv = UpSample(n_classes)

    def forward(self, x):
      # contraction phase
      x_1, x_1_conv = self.downSample1.forward(x)
      x_2, x_2_conv = self.downSample2.forward(x_1)
      x_3, x_3_conv = self.downSample3.forward(x_2)
      x_4, x_4_conv = self.downSample4.forward(x_3)
      x_bottleneck = self.bottleneck.forward(x_4, True)

      # expansion phase
      x = self.upSample1.forward(x_bottleneck, x_4_conv)
      x = self.upSample2.forward(x, x_3_conv)
      x = self.upSample3.forward(x, x_2_conv)
      x = self.upSample4.forward(x, x_1_conv)
      x_output = self.outConv.forward(x, [], True)

      return x_output





