import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def tensorflow_unet_1024(input_img):
    # encoder
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4)

    # decoder
    up1 = UpSampling2D((2, 2))(conv5)
    conc_up_1 = Concatenate()([up1, conv4])
    conv7 = Conv2D(512, (3, 3), activation="relu", padding="same")(conc_up_1)
    up2 = UpSampling2D((2, 2))(conv7)
    conc_up_2 = Concatenate()([up2, conv3])
    conv8 = Conv2D(256, (3, 3), activation="relu", padding="same")(conc_up_2)
    up3 = UpSampling2D((2, 2))(conv8)
    conc_up_3 = Concatenate()([up3, conv2])
    conv9 = Conv2D(128, (3, 3), activation="relu", padding="same")(conc_up_3)
    up4 = UpSampling2D((2, 2))(conv9)
    conc_up_4 = Concatenate()([up4, conv1])
    conv10 = Conv2D(64, (3, 3), activation="relu", padding="same")(conc_up_4)
    decoded = Conv2D(1, (1, 1), activation=None, padding="same")(conv10)

    return decoded


class UNet_1024(nn.Module):
    def __init__(self, input_channels):
        super(UNet_1024,self).__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=64,kernel_size=3,padding="same")
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding="same")
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding="same")
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding="same")
        self.conv5 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding="same")

        self.conv7 = nn.Conv2d(in_channels=1024+512,out_channels=512,kernel_size=3,padding="same")
        self.conv8 = nn.Conv2d(in_channels=512+256,out_channels=256,kernel_size=3,padding="same")
        self.conv9 = nn.Conv2d(in_channels=256+128,out_channels=128,kernel_size=3,padding="same")
        self.conv10 = nn.Conv2d(in_channels=128+64,out_channels=64,kernel_size=3,padding="same")

        self.decoded = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,padding="same")

    def forward(self,x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3 = self.relu(self.conv3(self.pool(conv2)))
        conv4 = self.relu(self.conv4(self.pool(conv3)))
        conv5 = self.relu(self.conv5(self.pool(conv4)))

        conv7 = self.relu(self.conv7(torch.cat([self.up(conv5), conv4], dim=1)))
        conv8 = self.relu(self.conv8(torch.cat([self.up(conv7), conv3], dim=1)))
        conv9 = self.relu(self.conv9(torch.cat([self.up(conv8), conv2], dim=1)))
        conv10 = self.relu(self.conv10(torch.cat([self.up(conv9), conv1], dim=1)))

        return self.decoded(conv10)
    

input_tensor = tf.ones([1, 256, 256, 60])  # Format: [batch_size, height, width, channels]
# Assuming input_tensor is a TensorFlow tensor
# Convert TensorFlow tensor to NumPy array
input_tensor_np = input_tensor.numpy()

# Convert NumPy array to PyTorch tensor
input_tensor_torch = torch.from_numpy(input_tensor_np)

# Change the data type and shape to match PyTorch expectations
# PyTorch expects the input in the format [batch_size, channels, height, width] and float32 data type
input_tensor_torch = input_tensor_torch.permute(0, 3, 1, 2).float()



# TensorFlow
init = tf.keras.initializers.Ones()
conv1 = tf.keras.layers.Conv2D(64,3,padding='valid',kernel_initializer=init,bias_initializer=init)
conv2 = tf.keras.layers.Conv2D(128,3,padding='valid',kernel_initializer=init,bias_initializer=init)
output = conv2(conv1(input_tensor))
print("Result 1:", np.sum(output.numpy())) # prints 5026357000.0

# PyTorch
init = torch.nn.init.ones_
conv1 = torch.nn.Conv2d(in_channels=60,out_channels=64,kernel_size=3,padding="valid")
init(conv1.weight)
init(conv1.bias)
conv2 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding="valid")
init(conv2.weight)
init(conv2.bias)
output = conv2(conv1(input_tensor_torch))
print("Result 2: ", np.sum(output.detach().numpy())) # prints 5026358300.0