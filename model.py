import torch
import torchvision
""" Down Sample Class"""
""" operations performed:
1. 3x3 CONV (ReLU + Batch Normalization) 
2. 3x3 CONV (ReLU + Batch Normalization) 
3. 2x2 Max Pooling 
4. dropout
Feature maps double as we go down the blocks, starting at 64, then 128, 256, and 512.
After 4 iterations, bottleneck (consists of 2 CONV layers with Batch Normalization & relu)
"""

class DownSample(torch.nn.Module)
    def __init__(self, in_channels, out_channels, conv_kernel=3, maxpool_kernel=2, maxpool_stride=2, dropout_p=0.1):
      super().__init__()¨

      # first convolution + batch normalization
      self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel)
      self.batchnorm1 = torch.nn.BatchNorm2d(out_channels)

      # second convolution + batch normalization
      self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=conv_kernel)
      self.batchnorm2 = torch.nn.BatchNorm2d(out_channels)

      # maxpooling + dropout
      self.maxpool = torch.nn.MaxPool2d(maxpool_kernel, stride=maxpool_stride)
      self.dropout = torch.nn.Dropout2d(p=dropout_p)

    # Method for down sample iteration
    # :param x: input tensor contraction phase
    # :param last: flag for bottleneck
    def forward(self, x, last=False):
      # relu function
      relu = torch.nn.functional.relu
      
      # normal case
      if !last:
        # conv 3x3, batchnorm, relu * 2
        x = relu(self.batchnorm1(self.conv1(x)))
        x = relu(self.batchnorm2(self.conv2(x)))
        # max pool 2x2
        x = self.maxpool(x)
        x = self.dropout(x)
      else:
        # bottleneck
        x = relu(self.batchnorm1(self.conv1(x)))
        x = relu(self.batchnorm2(self.conv2(x)))

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
class UpSample(torch.nn.Module) 
    def __init__(self, in_channels, out_channels, conv_kernel=3, conv_transp_kernel=2, stride=2, dropout_p=0.1): 
      super().__init__() 
 
      # transpose convolution
      self.up = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=conv_transp_kernel) 
       
      # first convolution + batch normalization
      self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel) 
      self.batchnorm1 = torch.nn.BatchNorm2d(out_channels) 
 
      # second convolution + batch normalization
      self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=conv_kernel) 
      self.batchnorm2 = torch.nn.BatchNorm2d(out_channels) 
 
      # dropout
      self.dropout = torch.nn.Dropout2d(p=dropout_p) 

    # Method for up sample iteration
    # :param x: input tensor expantion phase
    # :param x_1: corresponding input tensor contraction phase
    # :param last: flag for output convolution
    def forward(self, x, x_1, last=False): 
      # relu function
      relu = torch.nn.functional.relu 

      # normal case
      if !last:
        x = self.up(x) 
        
        # copy and crop (padding + concatenation)
        difference_width = x.size()[2] - x_1.size()[2]
        difference_height = x.size()[3] - x_1.size()[3]

        left_padding = difference_width // 2
        right_padding = difference_width - left_padding
        top_padding = difference_height // 2
        bottom_padding = difference_height - top_padding

        x = nn.functional.pad(x, [left_padding, right_padding, top_padding, bottom_padding])
        x = torch.cat([x_1, x], dim=1)

        x = self.dropout(x) 

        # conv 3x3, batchnorm, relu * 2
        x = relu(self.batchnorm1(self.conv1(x))) 
        x = relu(self.batchnorm2(self.conv2(x))) 
       
      else:
        # output convolution case
        x = self.conv1(x)
      
      return x 
 
class Unet(torch.nn.Module) 
    def __init__(self, n_channels, n_classes): 
      super().__init__() 

      # 4 down sample operations
      self.downSample1 = DownSample(n_channels, 64)
      self.downSample2 = DownSample(64, 128)
      self.downSample3 = DownSample(128, 256)
      self.downSample4 = DownSample(256, 512)

      # 1 bottleneck
      self.bottleneck = DownSample(512, 1024)

      # 4 up sample operations
      self.upSample1 = UpSample(1024, 512)
      self.upSample2 = UpSample(512, 256)
      self.upSample3 = UpSample(256, 128)
      self.upSample4 = UpSample(128, 64)

      # 1 output convolution
      self.outConv = UpSample(64, n_classes)

    def forward(self, x):
      # contraction phase
      x_1 = self.downSample1(x)
      x_2 = self.downSample2(x_1)
      x_3 = self.downSample3(x_2)
      x_4 = self.downSample4(x_3)
      x_bottleneck = self.bottleneck(x_4)

      # expansion phase
      x = self.upSample1(x_bottleneck, x_4)
      x = self.upSample2(x, x_3)
      x = self.upSample3(x, x_2)
      x = self.upSample4(x, x_1)
      x_output = self.outConv(x, [], True)

      return x_output





