import tensorflow as tf
from masters.models.rmflayers import *
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalMaxPooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    Multiply,
    ReLU,
    Reshape,
)

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
# https://stackoverflow.com/questions/55827660/batchnormalization-implementation-in-keras-tf-backend-before-or-after-activa


class RMFNet(Model):
    def __init__(self):
        super(RMFNet, self).__init__()
        self.conv1 = Conv2D(16, 3, padding='same', name="conv1")
        self.bn1 = BatchNormalization(name="bn1")
        self.relu1 = ReLU(name="relu1")
        self.pool1 = MaxPooling2D(name="pool1")

        self.conv2 = Conv2D(16, 3, padding='same', name="conv2")
        self.bn2 = BatchNormalization(name="bn2")
        self.relu2 = ReLU(name="relu2")
        self.pool2 = MaxPooling2D(name="pool2")

        self.block1 = Block(32, name="block1")
        self.b_pool1 = MaxPooling2D(name="block_pool1")
        
        self.block2 = Block(64, name="block2")
        self.b_pool2 = MaxPooling2D(name="block_pool2")

        self.block3 = Block(128, name="block3")

        self.self_attention = SelfAttention()
        self.global_attention = GlobalAttentionBlock()
        self.channel_attention = ChannelAttention()

        self.global_pool = GlobalMaxPooling2D()
        self.dropout = Dropout(0.5)
        self.dense = Dense(4, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        a1 = self.block1(x)
        x = self.b_pool1(a1)

        a2 = self.block2(x)
        x = self.b_pool2(a2)

        a3 = self.block3(x)
        a31 = self.self_attention(a3)
        a32 = self.global_attention(a3)
        a3 = Add(name="add")([a31, a32])
        x = self.channel_attention(a3)

        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.dense(x)

        return x

    def build(self, input_shape):
        # Build the initial convolution layers
        self.conv1.build(input_shape)
        bn_shape = self.conv1.compute_output_shape(input_shape)
        self.bn1.build(bn_shape)
        pool_shape = self.pool1.compute_output_shape(bn_shape)

        # Build second conv block
        self.conv2.build(pool_shape)
        bn_shape2 = self.conv2.compute_output_shape(pool_shape)
        self.bn2.build(bn_shape2)
        pool_shape2 = self.pool2.compute_output_shape(bn_shape2)

        # Build residual blocks
        self.block1.build(pool_shape2)
        block1_shape = self.block1.compute_output_shape(pool_shape2)
        pool_shape3 = self.b_pool1.compute_output_shape(block1_shape)

        self.block2.build(pool_shape3)
        block2_shape = self.block2.compute_output_shape(pool_shape3)
        pool_shape4 = self.b_pool2.compute_output_shape(block2_shape)

        self.block3.build(pool_shape4)
        block3_shape = self.block3.compute_output_shape(pool_shape4)

        # Build attention layers
        self.self_attention.build(block3_shape)
        self.global_attention.build(block3_shape)
        self.channel_attention.build(block3_shape)

        # Build final layers
        pool_shape5 = self.global_pool.compute_output_shape(block3_shape)
        self.dense.build(pool_shape5)

        super(RMFNet, self).build(input_shape)
if __name__=="__main__":
    print("------------------------------------")
    model = RMFNet()
    # model = MyModel()
    model.build((1, 28, 28, 1))
    model.summary()
    print("done")