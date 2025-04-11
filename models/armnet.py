import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Concatenate, Add, Activation
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.random import normal
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    MaxPooling2D,
    Concatenate,
    GlobalMaxPooling2D,
    Dropout,
    Dense,
    Lambda,
    Reshape,
    Activation,
    Multiply,
    Add
)

from masters.models.rmflayers import *

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
# https://stackoverflow.com/questions/55827660/batchnormalization-implementation-in-keras-tf-backend-before-or-after-activa

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)
  
class ARMNet(Model):
    def __init__(self):
        super(ARMNet, self).__init__()
        # blockA
        self.layerA = self.blockA
        self.layerB = self.blockB

    def blockA(self,x):
        x = Conv2D(1, (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def blockB(self,x):
        x = Conv2D(1, (1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def call(self, input):
        x1 = self.layerA(input)
        x2 = self.layerA(x1)
        x3 = self.layerA(x2)
        x4 = self.layerA(x3)
        
        conc = Concatenate()([x1, x2, x3, x4])
        
        x5 = self.layerB(conc)

        p1 = self.layerB(input)

        add = Add([x5, p1])

        x6 = self.layerB(add)
        out = conc
        return out

class RNe(Model):
    def __init__(self):
        super(RNe, self).__init__()
        # blockA
        #H1
        self.layerA1 = Conv2D(1, (3, 3), padding='same')
        self.layerA2 = BatchNormalization()
        self.layerA3 = Activation("relu")

        #H2
        self.layerB1 = Conv2D(1, (3, 3), padding='same')
        self.layerB2 = BatchNormalization()
        self.layerB3 = Activation("relu")

        #H3
        self.layerC1 = Conv2D(1, (3, 3), padding='same')
        self.layerC2 = BatchNormalization()
        self.layerC3 = Activation("relu")

        #H4
        self.layerD1 = Conv2D(1, (3, 3), padding='same')
        self.layerD2 = BatchNormalization()
        self.layerD3 = Activation("relu")

        self.layerE1 = Conv2D(1, (1, 1), padding='same')
        self.layerE2 = BatchNormalization()
        self.layerE3 = Activation("relu")


        self.layerP1 = Conv2D(1, (1, 1))
        self.layerP2 = BatchNormalization()
        self.layerP3 = Activation("relu")

        self.layerF1 = Conv2D(1, (1, 1))
        self.layerF2 = BatchNormalization()
        self.layerF3 = Activation("relu")



    def call(self, input):
        x1 = self.layerA1(input)
        x2 = self.layerA2(x1)
        x3 = self.layerA3(x2)

        x4 = self.layerB1(input)#x3
        x5 = self.layerB2(x4)
        x6 = self.layerB3(x5)

        x7 = self.layerC1(input)#x6
        x8 = self.layerC2(x7)
        x9 = self.layerC3(x8)

        x10 = self.layerD1(input)#x9
        x11 = self.layerD2(x10)
        x12 = self.layerD3(x11)
        
        conc = Concatenate()([x3, x6, x9, x12])

        x13 = self.layerE1(conc)
        x14 = self.layerE2(x13)
        x15 = self.layerE3(x14)

        p1 = self.layerP1(input)
        p2 = self.layerP2(p1)
        p3 = self.layerP3(p2)

        add = Add()([x15, p3])
        
        x16 = self.layerF1(add)
        x17 = self.layerF2(x16)
        x18 = self.layerF3(x17)
        out = x18
        return out

    def build(self, input_shape):
        # import tensorflow.keras.random.normal
        # call using random input 
        self.call(normal(input_shape))
        # self.call(RandomNormal()(input_shape))
        self.built = True

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