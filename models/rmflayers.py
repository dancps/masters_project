from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    MaxPooling2D,
    Concatenate,
    Lambda,
    Reshape,
    Activation,
    Multiply,
    Add
)
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Block(Layer):
    def __init__(self, filters, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.filters = filters
        self.conv_layers = [Conv2D(filters, 3, padding='same') for _ in range(4)]
        self.bn_layers = [BatchNormalization() for _ in range(4)]
        self.relu = ReLU()
        self.concat = Concatenate()
        self.conv1x1_1 = Conv2D(2 * filters, 1, padding='same')
        self.conv1x1_2 = Conv2D(2 * filters, 1, padding='same')
        self.bn1x1_1 = BatchNormalization()
        self.bn1x1_2 = BatchNormalization()
        self.add = Add()

    def call(self, inputs):
        a = self.relu(self.bn_layers[0](self.conv_layers[0](inputs)))
        b = self.relu(self.bn_layers[1](self.conv_layers[1](a)))
        c = self.relu(self.bn_layers[2](self.conv_layers[2](b)))
        d = self.relu(self.bn_layers[3](self.conv_layers[3](c)))
        
        mid = self.relu(self.bn1x1_1(self.conv1x1_1(self.concat([a, b, c, d]))))
        x = self.relu(self.bn1x1_2(self.conv1x1_2(inputs)))
        return self.add([mid, x])

class GlobalAttentionBlock(Layer):
    def __init__(self, **kwargs):
        super(GlobalAttentionBlock, self).__init__(**kwargs)
        self.concat = Concatenate()
        self.relu = Activation('relu')
        self.conv1x1 = Conv2D(1, 1, padding='same')
        self.sigmoid = Activation('sigmoid')
        self.multiply = Multiply()

    def call(self, inputs):
        x = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
        y = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
        x = self.relu(self.conv1x1(self.concat([x, y])))
        x = self.sigmoid(x)
        return self.multiply([x, inputs])

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

        self.conv1x1_1 = None#Conv2D(self.shape[3] // 8, 1, padding='same')
        self.conv1x1_2 = None#Conv2D(self.shape[3] // 8, 1, padding='same')
        self.conv1x1_3 = None#Conv2D(self.shape[3] // 8, 1, padding='same')
        self.conv1x1_4 = None#Conv2D(self.shape[3], 1, padding='same')
        self.reshape = Reshape
        self.permute = K.permute_dimensions
        self.batch_dot = K.batch_dot
        self.softmax = Activation('softmax')
        self.relu = Activation('relu')

    def build(self, input_shape):
        # Define the shape of the Conv2D layers based on input_shape
        self.conv1x1_1 = Conv2D(input_shape[3] // 8, 1, padding='same')
        self.conv1x1_2 = Conv2D(input_shape[3] // 8, 1, padding='same')
        self.conv1x1_3 = Conv2D(input_shape[3] // 8, 1, padding='same')
        self.conv1x1_4 = Conv2D(input_shape[3], 1, padding='same')
        super(SelfAttention, self).build(input_shape)  # Be sure to call this at the end


    def call(self, inputs):
        self.shape = inputs.shape
        
        a = self.relu(self.conv1x1_1(inputs))
        b = self.relu(self.conv1x1_2(inputs))
        c = self.relu(self.conv1x1_3(inputs))
        
        a = self.reshape((self.shape[1] * self.shape[2], self.shape[3] // 8))(a)
        b = self.permute(self.reshape((self.shape[1] * self.shape[2], self.shape[3] // 8))(b), (0, 2, 1))
        c = self.reshape((self.shape[1] * self.shape[2], self.shape[3] // 8))(c)
        
        inter = self.softmax(self.batch_dot(a, b))
        out = self.batch_dot(inter, c)
        out = self.relu(self.conv1x1_4(self.reshape((self.shape[1], self.shape[2], self.shape[3] // 8))(out)))
        return out

class ChannelAttention(Layer):
    def __init__(self, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.max_pool = MaxPooling2D
        self.conv1x1_1 = Conv2D
        self.conv1x1_2 = Conv2D
        self.relu = Activation('relu')
        self.sigmoid = Activation('sigmoid')
        self.multiply = Multiply()

    def build(self, input_shape):
        self.conv1x1_1 = Conv2D(input_shape[3] // 8, 1, padding='same', kernel_initializer='he_normal', use_bias=False)
        self.conv1x1_2 = Conv2D(input_shape[3], 1, padding='same', kernel_initializer='he_normal', use_bias=False)

    def call(self, inputs):
        shape = K.int_shape(inputs)
        x = self.max_pool(pool_size=(shape[1], shape[2]))(inputs)
        x = self.relu(self.conv1x1_1(x))
        x = self.sigmoid(self.conv1x1_2(x))
        return self.multiply([x, inputs])