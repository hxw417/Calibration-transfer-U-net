from keras.models import Model
from keras.layers import  Activation, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, UpSampling1D,Input, BatchNormalization, Lambda, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.optimizers import *
from keras import regularizers
from keras.regularizers import l2
from keras.layers import LeakyReLU


def creat_1D_unet5():
    inputs = Input(shape=(5000, 1))

    conv1 = Conv1D(8, 3, padding='same')(inputs)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling1D(2)(conv1)

    conv2 = Conv1D(16, 3, padding='same')(pool1)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling1D(2)(conv2)

    conv3 = Conv1D(32, 3, padding='same')(pool2)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling1D(2)(conv3)

    conv4 = Conv1D(64, 3, padding='same')(pool3)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling1D(5)(conv4)

    conv5 = Conv1D(128, 3, padding='same')(pool4)
    conv5 = Activation('relu')(conv5)
    pool5 = MaxPooling1D(5)(conv5)

    conv6 = Conv1D(128, 3, padding='same')(pool5)
    conv6 = Activation('relu')(conv6)


    up1 = UpSampling1D(5)(conv6)
    # conv7 = Conv1D( 128, 3, padding='same')(up1)
    # conv7 = Activation('relu')(conv7)
    merged1 = concatenate([conv5, up1])
    # merged1 = concatenate([conv5, conv7])
    conv7 = Conv1D(64, 3, padding='same')(merged1)
    conv7 = Activation('relu')(conv7)
    #
    up2 = UpSampling1D(5)(conv7)
    merged2 = concatenate([conv4, up2])
    # conv7 = Conv1D(64, 3, padding='same')(up2)
    # conv7 = Activation('relu')(conv7)
    # merged2 = concatenate([conv3, conv7])
    conv7 = Conv1D(32, 3, padding='same')(merged2)
    conv7 = Activation('relu')(conv7)
    #
    up3 = UpSampling1D(2)(conv7)
    # conv8 = Conv1D(32, 3, padding='same')(up3)
    # conv8 = Activation('relu')(conv8)
    merged3 = concatenate([conv3, up3])
    conv8 = Conv1D(16, 3, padding='same')(merged3)
    conv8 = Activation('relu')(conv8)
    #
    up4 = UpSampling1D(2)(conv8)
    # conv9 = Conv1D(16, 3, padding='same')(up4)
    # conv9 = Activation('relu')(conv9)
    merged4 = concatenate([conv2, up4])
    conv9 = Conv1D(8, 3, padding='same')(merged4)
    conv9 = Activation('relu')(conv9)
    #
    #
    up5 = UpSampling1D(2)(conv9)
    merged5 = concatenate([conv1, up5])
    conv10 = Conv1D(1, 3, padding='same')(merged5)
    # conv10 = Activation('relu')(conv10)
    #
    # # conv10 = Conv1D(8, 3, padding='same')(conv9)
    # # conv10 = Activation('relu')(conv10)
    # conv11 = Conv1D(1, 1, padding='same')(conv10)
    conv11 = Activation('sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=conv10)
    print(model.summary())
    return model
