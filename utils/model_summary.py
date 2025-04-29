from masters.models.armnet import *
from masters.models.rmflayers import *

from tensorflow.keras.applications import ResNet50
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
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from termcolor import colored

def block(inputs, filters, name):
    a = Conv2D(filters, 3, padding='same',name=name+"_conv2d_1")(inputs)
    a = BatchNormalization(name=name+"_batch_normalization_1")(a)
    a = ReLU(name=name+"_relu_1")(a)
    
    b = Conv2D(filters, 3, padding='same',name=name+"_conv2d_2")(a)
    b = BatchNormalization(name=name+"_batch_normalization_2")(b)
    b = ReLU(name=name+"_relu_2")(b)
    
    c = Conv2D(filters, 3, padding='same',name=name+"_conv2d_3")(b)
    c = BatchNormalization(name=name+"_batch_normalization_3")(c)
    c = ReLU(name=name+"_relu_3")(c)
    
    d = Conv2D(filters, 3, padding='same',name=name+"_conv2d_4")(c)
    d = BatchNormalization(name=name+"_batch_normalization_4")(d)
    d = ReLU(name=name+"_relu_4")(d)
    
    mid = Concatenate(name=name+"_concatenate")([a, b, c, d])
    mid = Conv2D(2 * filters, 1, padding='same',name=name+"_conv1x1_1")(mid)
    mid = BatchNormalization(name=name+"_batch_normalization_5")(mid)
    mid = ReLU(name=name+"_relu_5")(mid)
    
    x = Conv2D(filters * 2, 1,name=name+"_conv1x1_2")(inputs)
    x = BatchNormalization(name=name+"_batch_normalization_6")(x)
    x = ReLU(name=name+"_relu_6")(x)
    
    x = Add(name=name+"_add")([mid, x])
    
    y = Conv2D(filters * 2, 1,name=name+"_conv1x1_3")(x)
    y = BatchNormalization(name=name+"_batch_normalization_7")(y)
    y = ReLU(name=name+"_relu_7")(y)
    
    return y

def Global_attention_block(C_A):
    x = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True),name="global_att_lambda_1")(C_A)
    y = Lambda(lambda x: K.max(x, axis=-1, keepdims=True),name="global_att_lambda_2")(C_A)
    
    x = Concatenate(name="global_att_concatenate")([x, y])
    x = Activation('relu',name="global_att_relu")(x)
    x = Conv2D(1, 1, padding='same',name="global_att_conv2d_1")(x)
    x = Activation('sigmoid',name="global_att_sigmoid")(x)
    S_A = Multiply(name="global_att_multiply")([x, C_A])
    
    return S_A

def self_attention(inp):
    shp = inp.shape
    a = Conv2D(shp[3] // 8, 1, padding='same',name="self_att_conv2d_1")(inp)
    a = Activation('relu',name="self_att_relu_1")(a)
    
    b = Conv2D(shp[3] // 8, 1, padding='same',name="self_att_conv2d_2")(inp)
    b = Activation('relu',name="self_att_relu_2")(b)
    
    c = Conv2D(shp[3] // 8, 1, padding='same',name="self_att_conv2d_3")(inp)
    c = Activation('relu',name="self_att_relu_3")(c)
    
    a = Reshape((shp[1] * shp[2], shp[3] // 8),name="self_att_reshape_1")(a)
    b = Reshape((shp[1] * shp[2], shp[3] // 8),name="self_att_reshape_2")(b)
    b = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)),name="self_att_permute")(b)
    c = Reshape((shp[1] * shp[2], shp[3] // 8),name="self_att_reshape_3")(c)
    
    inter = Lambda(lambda x: K.batch_dot(x[0], x[1]),name="self_att_batch_dot_1")([a, b])
    inter = Activation('softmax',name="self_att_softmax")(inter)
    out = Lambda(lambda x: K.batch_dot(x[0], x[1]),name="self_att_batch_dot_2")([inter, c])
    
    out = Reshape((shp[1], shp[2], shp[3] // 8),name="self_att_reshape_4")(out)
    out = Conv2D(shp[3], 1, padding='same',name="self_att_conv2d_4")(out)
    out = Activation('relu',name="self_att_relu_4")(out)
    
    return out

def channel_attention(inputs):
    shape = K.int_shape(inputs)
    x = MaxPooling2D(pool_size=(shape[1], shape[2]),name="ch_att_max_pooling2d")(inputs)
    x = Conv2D(shape[3] // 8, 1, padding='same', kernel_initializer='he_normal', use_bias=False,name="ch_att_conv2d_1")(x)
    x = Activation('relu',name="ch_att_relu_1")(x)
    x = Conv2D(shape[3], 1, padding='same', kernel_initializer='he_normal', use_bias=False,name="ch_att_conv2d_2")(x)
    x = Activation('sigmoid',name="ch_att_sigmoid")(x)
    x = Multiply(name="ch_att_multiply")([x, inputs])
    
    return x

def load_model():
    K.clear_session()
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(16, 3, padding='same', name="conv1")(inputs)
    x = BatchNormalization(name="bn1")(x)
    x = ReLU(name="relu1")(x)
    x = MaxPooling2D(name="pool1")(x)
    
    x = Conv2D(16, 3, padding='same', name="conv2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = ReLU(name="relu2")(x)
    x = MaxPooling2D(name="pool2")(x)
    
    a1 = block(x, 32, "block1")
    x = MaxPooling2D(name="block_pool1")(a1)
    
    a2 = block(x, 64, "block2")
    x = MaxPooling2D(name="block_pool2")(a2)
    
    a3 = block(x, 128, "block3")
    a31 = self_attention(a3)
    a32 = Global_attention_block(a3)
    a3 = Add(name="add")([a31, a32])
    x = channel_attention(a3)
    
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    
    return model


def get_test_model():
    model = tf.keras.Sequential([
        # Block(32) # OK, validated
        GlobalAttentionBlock(),# OK, validated
        # SelfAttention(),# OK, validated
        # ChannelAttention() # OK, validated
    ])
    return model
def test_model(layer, input_shape, expand_nested=False):
    model = tf.keras.Sequential([
        layer,
    ])
    model.build(input_shape)
    model.summary(expand_nested=expand_nested)


if __name__ == "__main__":
    # ref_model = load_model()
    # # ref_model.build((None, 224, 224, 3))
    # ref_model.summary()

    # # Total params: 1,133,685 (4.32 MB)
    # # Trainable params: 1,129,141 (4.31 MB)
    # # Non-trainable params: 4,544 (17.75 KB)
    # print()

    # print("****", colored("GlobalAttentionBlock", "blue"), colored("OK", "green"))
    # test_model(GlobalAttentionBlock(), (None, 14, 14, 256))
    # print()


    # print("****", colored("SelfAttention", "blue"), colored("OK", "green"))
    # test_model(SelfAttention(), (None, 14, 14, 256))
    # print()


    # print("****", colored("ChannelAttention", "blue"), colored("OK", "red"))
    # test_model(ChannelAttention(), (None, 14, 14, 256), True)
    # print()

    print("------------------------------------")
    print("------------------------------------")
    print("------------------------------------")
    model = RMFNet()
    model.build((None, 224, 224, 3))
    model.summary(expand_nested=True)

    print("------------------------------------")
    print("------------------------------------")
    print("------------------------------------")
    model = ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=4,
        classifier_activation="softmax",
    )
    model.build((None, 224, 224, 3))
    model.summary(expand_nested=True)