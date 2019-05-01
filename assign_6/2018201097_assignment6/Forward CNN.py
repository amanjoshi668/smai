
# coding: utf-8

# In[9]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import floor as floor


# In[10]:


from pylab import rcParams
rcParams['figure.figsize'] = 10, 15


# In[11]:


class Convolution2D:
    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

    def relu(self, z):
        #         print(z)
        s = np.maximum(0, z)
        return s

    def tanh(self, z):
        return np.tanh(z)

    def __init__(self,
                 input_channels,
                 filter_size,
                 num_filters,
                 stride,
                 activation_function="relu"):
        self.channels = input_channels
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.stride = stride
        if activation_function == "relu":
            self.activation = self.relu
        elif activation_function == "sigmoid":
            self.activation = self.sigmoid
        else:
            self.activation = self.tanh

        self.weights = np.zeros((self.num_filters, self.channels,
                                 self.filter_size, self.filter_size))
        self.bias = np.zeros((self.num_filters, 1))
        for i in range(0, self.num_filters):
            self.weights[i, :, :, :] = np.random.normal(
                loc=0,
                scale=np.sqrt(
                    1. / self.channels * self.filter_size * self.filter_size),
                size=(self.channels, self.filter_size, self.filter_size))

    def forward(self, inputs):
        self.inputs = inputs
        C = inputs.shape[0]
        IS = inputs.shape[1]
        OS = floor((IS - self.filter_size) / self.stride + 1)
        result = np.zeros((self.num_filters, OS, OS))
        for f in range(self.num_filters):
            for w in range(OS):
                for h in range(OS):
                    result[f, w, h] = np.sum(
                        inputs[:, w:w + self.filter_size, h:h +
                               self.filter_size] *
                        self.weights[f, :, :, :]) + self.bias[f]
        return self.activation(result)

    def plot(self, result):
        fig = plt.figure(figsize=(15, 15))
        ax = []
        for i in range(1, self.num_filters + 1):
            img = result[i - 1]
            ax.append(fig.add_subplot(1, self.num_filters, i))
            ax[-1].set_title("filter: " + str(i))
            plt.imshow(img, alpha=0.25)
        plt.show()


# In[12]:


class MaxPooling2D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        C = inputs.shape[0]
        IS = inputs.shape[1]
        OS = floor((IS - self.pool_size) / self.stride + 1)
        result = np.zeros((C, OS, OS))
        for c in range(C):
            for w in range(floor(IS / self.stride)):
                for h in range(floor(IS / self.stride)):
                    result[c, w, h] = np.max(
                        inputs[c, w * self.stride:w * self.stride +
                               self.pool_size, h *
                               self.stride:h * self.stride + self.pool_size])
        return result

    def plot(self, result):
        fig = plt.figure(figsize=(15, 15))
        ax = []
        for i in range(1, len(result) + 1):
            img = result[i - 1]
            ax.append(fig.add_subplot(1, len(result), i))
            ax[-1].set_title("filter: " + str(i))
            plt.imshow(img, alpha=0.25)
        plt.show()


# In[13]:


class FullyConnected:
    def __init__(self, input_num, output_num):
        self.weights = np.random.rand(input_num, output_num) * 0.01
        self.bias = np.zeros((output_num, 1))

    def forward(self, inputs):
        inputs = inputs.flatten()
        return np.dot(inputs, self.weights) + self.bias.T

    def plot(self, *args):
        pass


class ReLu:
    def forward(self, inputs):
        z = inputs.copy()
        z[z < 0] = 0
        return z

    def plot(self, *args):
        pass


# In[14]:


class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for l in range(len(self.layers)):
            inputs = self.layers[l].forward(inputs)
            self.layers[l].plot(inputs)
        return inputs


# In[20]:


model = Model()
model.add_layer(Convolution2D(3, 5, 6, 1))
model.add_layer(MaxPooling2D(2, 2))
model.add_layer(Convolution2D(6, 5, 16, 1))
model.add_layer(MaxPooling2D(2, 2))
model.add_layer(Convolution2D(16, 5, 120, 1))
model.add_layer(FullyConnected(120, 84))
model.add_layer(FullyConnected(84, 10))


# In[21]:


image = cv.imread('9.png')
image = cv.resize(image, (32, 32))
cv.imwrite('image.jpg', image)


# In[22]:


image = mpimg.imread('image.jpg')
imagePlot = plt.imshow(image)
plt.show()
model.forward(image.T)


# In[25]:


model = Model()
model.add_layer(Convolution2D(3, 5, 6, 1,activation_function="sigmoid"))
model.add_layer(MaxPooling2D(2, 2))
model.add_layer(Convolution2D(6, 5, 16, 1,activation_function="sigmoid"))
model.add_layer(MaxPooling2D(2, 2))
model.add_layer(Convolution2D(16, 5, 120, 1,activation_function="sigmoid"))
model.add_layer(FullyConnected(120, 84))
model.add_layer(FullyConnected(84, 10))
image = cv.imread('9.png')
image = cv.resize(image, (32, 32))
cv.imwrite('image.jpg', image)
image = mpimg.imread('image.jpg')
imagePlot = plt.imshow(image)
plt.show()
model.forward(image.T)


# In[26]:


model = Model()
model.add_layer(Convolution2D(3, 5, 6, 1,activation_function="tanh"))
model.add_layer(MaxPooling2D(2, 2))
model.add_layer(Convolution2D(6, 5, 16, 1,activation_function="tanh"))
model.add_layer(MaxPooling2D(2, 2))
model.add_layer(Convolution2D(16, 5, 120, 1,activation_function="tanh"))
model.add_layer(FullyConnected(120, 84))
model.add_layer(FullyConnected(84, 10))
image = cv.imread('9.png')
image = cv.resize(image, (32, 32))
cv.imwrite('image.jpg', image)
image = mpimg.imread('image.jpg')
imagePlot = plt.imshow(image)
plt.show()
model.forward(image.T)


# ## Question 2

# ### 1
# - Number of parameters in 1st convolutional layer = filter_size x filter_size x n_channels x num_filters + num_filters
# - 5x5x3x6 + 6 = 450 + 6 = 456 parameters
# ### 2
# - Number of parameters in pooling operations = 0
# ### 3
# - Fully connected layer (FC)
# ### 4
# - Fully connected layers at the end
# ### 5
# - With changing the activation function the output images are changing
# - Can't tell much as the initialization are random and there is no back propogation
