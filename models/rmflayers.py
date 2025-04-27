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
        self.conv1x1_1 = Conv2D(2 * filters, 1)#, padding='same')
        self.conv1x1_2 = Conv2D(2 * filters, 1)#, padding='same')
        self.conv1x1_3 = Conv2D(2 * filters, 1)#, padding='same')
        self.bn1x1_1 = BatchNormalization()
        self.bn1x1_2 = BatchNormalization()
        self.bn1x1_3 = BatchNormalization()
        self.add = Add()

    def call(self, inputs):
        a = self.relu(self.bn_layers[0](self.conv_layers[0](inputs)))
        b = self.relu(self.bn_layers[1](self.conv_layers[1](a)))
        c = self.relu(self.bn_layers[2](self.conv_layers[2](b)))
        d = self.relu(self.bn_layers[3](self.conv_layers[3](c)))
        
        concat = self.concat([a, b, c, d])

        mid = self.relu(self.bn1x1_1(self.conv1x1_1(concat)))
        x = self.relu(self.bn1x1_2(self.conv1x1_2(inputs)))

        add = self.add([mid, x])

        y = self.relu(self.bn1x1_3(self.conv1x1_3(add)))

        return y
    
    def build(self, input_shape):
        print("input_shape", input_shape)
        # Initialize the convolutional layers with the appropriate number of filters
        temp_input_shape = input_shape
        for i in range(4):
            self.conv_layers[i].build(temp_input_shape) # Same padding type keeps the output shape the same
            temp_input_shape = self.conv_layers[i].compute_output_shape(temp_input_shape)
            self.bn_layers[i].build(temp_input_shape)
            temp_input_shape = self.bn_layers[i].compute_output_shape(temp_input_shape)
            # Update input_shape for the next layer
            print(f"conv_layers[{i}] params:", self.conv_layers[i].count_params())
            print(f"bn_layers[{i}] params:", self.bn_layers[i].count_params())
            # it is needed to calc on each iteration because from the first layer the output shape is different
        
        # Initialize the 1x1 convolutional layers
        # Since we are concatenating 4 layers, the output shape is different. 
        # It will be 4 times the number of filters.
        concat_shape = list(temp_input_shape)
        concat_shape[-1] = 4 * self.filters
        self.conv1x1_1.build(concat_shape)
        print("conv1x1_1 params:", self.conv1x1_1.count_params())
        self.bn1x1_1.build(self.conv1x1_1.compute_output_shape(concat_shape))
        print("bn1x1_1 params:", self.bn1x1_1.count_params())
        
        # The second 1x1 conv takes the original input shape
        self.conv1x1_2.build(input_shape)
        print("conv1x1_2 params:", self.conv1x1_2.count_params())
        self.bn1x1_2.build(self.conv1x1_2.compute_output_shape(input_shape))
        print("bn1x1_2 params:", self.bn1x1_2.count_params())
        
        add_shape = self.conv1x1_2.compute_output_shape(input_shape)
        
        self.conv1x1_3.build(add_shape)
        print("conv1x1_3 params:", self.conv1x1_3.count_params())
        self.bn1x1_3.build(self.conv1x1_3.compute_output_shape(add_shape))
        print("bn1x1_3 params:", self.bn1x1_3.count_params())
        print()
        
        # Call the parent build method
        super(Block, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        # The output shape of the Block is the same as the input shape
        # but with the number of filters doubled
        output_shape = list(input_shape)
        output_shape[-1] = 2 * self.filters
        return tuple(output_shape)

class GlobalAttentionBlock(Layer):
    def __init__(self, **kwargs):
        super(GlobalAttentionBlock, self).__init__(**kwargs)
        self.concat = Concatenate()
        self.relu = Activation('relu')
        self.conv1x1 = Conv2D(1, 1, padding='same')
        self.sigmoid = Activation('sigmoid')
        self.multiply = Multiply()

    def build(self, input_shape):
        # Initialize the 1x1 convolutional layer
        # Input shape will be (batch, height, width, 2) after concatenation
        concat_shape = list(input_shape)
        concat_shape[-1] = 2  # Mean and max pooling outputs concatenated
        self.conv1x1.build(concat_shape)
        
        # Call the parent build method
        super(GlobalAttentionBlock, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        # The output shape of GlobalAttentionBlock is the same as the input shape
        # because the attention map is multiplied with the original input
        return input_shape

    def call(self, inputs):
        x = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
        y = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
        
        concat = self.concat([x, y])
        
        attention = self.sigmoid(self.conv1x1(self.relu(concat)))
        
        return self.multiply([attention, inputs])

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
        # a,b,c
        self.conv1x1_1 = Conv2D(input_shape[-1] // 8, 1, padding='same')
        self.conv1x1_2 = Conv2D(input_shape[-1] // 8, 1, padding='same')
        self.conv1x1_3 = Conv2D(input_shape[-1] // 8, 1, padding='same')

        # out
        self.conv1x1_4 = Conv2D(input_shape[-1], 1, padding='same')

        self.conv1x1_1.build(input_shape)
        self.conv1x1_2.build(input_shape)
        self.conv1x1_3.build(input_shape)

        # Build output conv
        output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] // 8)
        self.conv1x1_4.build(output_shape)


        super(SelfAttention, self).build(input_shape)  # Be sure to call this at the end


    def call(self, inputs):
        shape = inputs.shape
        # self.shape = inputs.shape
        
        a = self.relu(self.conv1x1_1(inputs))
        b = self.relu(self.conv1x1_2(inputs))
        c = self.relu(self.conv1x1_3(inputs))
        
        a = self.reshape((shape[1] * shape[2], shape[3] // 8))(a)
        b = self.permute(self.reshape((shape[1] * shape[2], shape[3] // 8))(b), (0, 2, 1))
        c = self.reshape((shape[1] * shape[2], shape[3] // 8))(c)
        
        inter = self.softmax(self.batch_dot(a, b))
        out = self.batch_dot(inter, c)
        out = self.relu(self.conv1x1_4(self.reshape((shape[1], shape[2], shape[3] // 8))(out)))
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
        # input_shape is (batch, height, width, channels)
        # at least 8 channels
        self.conv1x1_1 = Conv2D(input_shape[-1] // 8, 1, padding='same', kernel_initializer='he_normal', use_bias=False)
        self.conv1x1_2 = Conv2D(input_shape[-1], 1, padding='same', kernel_initializer='he_normal', use_bias=False)
        self.conv1x1_1.build(input_shape)
        conv1x1_output_shape = self.conv1x1_1.compute_output_shape(input_shape)
        self.conv1x1_2.build(conv1x1_output_shape)
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        shape = inputs.shape
        x = self.max_pool(pool_size=(shape[1], shape[2]))(inputs)
        x = self.relu(self.conv1x1_1(x))
        x = self.sigmoid(self.conv1x1_2(x))
        return self.multiply([x, inputs])
    
    def compute_output_shape(self, input_shape):
        return input_shape