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
        self.conv1 = Conv2D(16, 3, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.pool1 = MaxPooling2D()

        self.conv2 = Conv2D(16, 3, padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.pool2 = MaxPooling2D()

        self.block1 = Block(32)
        self.b_pool1 = MaxPooling2D()
        
        self.block2 = Block(64)
        self.b_pool2 = MaxPooling2D()

        self.block3 = Block(128)

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
        x = self.b_pool1(a2)

        a3 = self.block3(x)
        a31 = self.self_attention(a3)
        a32 = self.global_attention(a3)
        a3 = Add()([a31, a32])
        x = self.channel_attention(a3)

        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.dense(x)

        return x

if __name__=="__main__":
    print("------------------------------------")
    model = RNe()
    # model = MyModel()
    model.build((1, 28, 28, 1))
    model.summary()
    print("done")