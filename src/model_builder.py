import tensorflow as tf
import numpy as np
import librosa # NOTE: librosa is built on top of matplotlib
import librosa.display
import random
import math
import subprocess

from config import INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS

"""
@author: Sreenivas Bhattiprolu, Python for Microscopists
"""

class ModelBuilder:
    """
    relu ensures that values are nonnegative
    kernel_initializer initializes weights(?) based on some probability distribution / function?
    padding denotes that the dimensions should be the same

    skip connections allow the model to "remember" what it's trying to recreate instead of just working with its details (i.e. features) https://ai.stackexchange.com/questions/37321/what-is-the-role-of-skip-connections-in-u-net
    the dropout drops out nodes in the input and hidden layer with probability p to ensure that the model is not overfitting


    C1:
    16 filters that are 3 by 3, filters define the features
    Drop 10%(?)
    Another 16 filters that are 3 by 3
    Max pooling layer 2 by 2 with stride 2 compresses the image to half its size

    C2: 
    32 filters that are 3 by 3
    Drop 10%(?)
    Another 32 filters
    Max pooling layer halves the size again

    C3:
    ...

    C4:
    ...

    C5:
    ...
    """

    def __init__(self, height: int=INPUT_HEIGHT, width: int=INPUT_WIDTH, channels: int=INPUT_CHANNELS):
        self.height = height
        self.width = width
        self.channels = channels

    def buildModel(self) -> tf.keras.Model:
        # Input
        inputs = tf.keras.layers.Input((self.width, self.height, self.channels))
        # s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

        # Contraction path -> Extract details
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        c1 = tf.keras.layers.BatchNormalization()(c1)

        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        c2 = tf.keras.layers.BatchNormalization()(c2)

        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.BatchNormalization()(c3)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        c3 = tf.keras.layers.BatchNormalization()(c3)

        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.BatchNormalization()(c4)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        c4 = tf.keras.layers.BatchNormalization()(c4)

        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
        
        c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.BatchNormalization()(c5)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)

        # Expansive path -> Upsample, reconstruct image
        u6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])

        c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.BatchNormalization()(c6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = tf.keras.layers.BatchNormalization()(c6)

        u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])

        c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.BatchNormalization()(c7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = tf.keras.layers.BatchNormalization()(c7)

        u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])

        c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.BatchNormalization()(c8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = tf.keras.layers.BatchNormalization()(c8)

        u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)

        c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.BatchNormalization()(c9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = tf.keras.layers.BatchNormalization()(c9)
        
        outputs = tf.keras.layers.Conv2D(2, (1, 1), activation='relu')(c9)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        return model
    
    def buildDataFromGenerator(self, data_generator):
        train_dataset = tf.data.Dataset.from_generator(data_generator,
                                                       output_signature=(tf.TensorSpec(shape=(None, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(None, INPUT_WIDTH, INPUT_HEIGHT, OUTPUT_CHANNELS), dtype=tf.float32)))

        return train_dataset