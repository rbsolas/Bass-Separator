import tensorflow as tf
import numpy as np
import librosa # NOTE: librosa is built on top of matplotlib
import librosa.display
import random
import math
import subprocess
import glob
import os
import datetime

from config import INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS, BATCH_SIZE, EPOCHS, PATH_TO_MODEL, PATH_TO_LOGS, PATIENCE, DEFAULT_MODEL
from wav_handler import WavHandler
from scipy.io.wavfile import write


"""
@author: Sreenivas Bhattiprolu, Python for Microscopists
"""

class ModelHandler:
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

    def __init__(self, height: int=INPUT_HEIGHT, width: int=INPUT_WIDTH, channels: int=INPUT_CHANNELS, batch_size: int=BATCH_SIZE, epochs: int=EPOCHS):
        self.height = height
        self.width = width
        self.channels = channels
        self.batch_size = batch_size
        self.epochs = epochs 

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
    
    def dataGenerator(self):
        """
        https://stackoverflow.com/questions/74404355/how-do-i-use-a-custom-generator-function-to-feed-keras-model-fit-samples-one-by
        https://stackoverflow.com/questions/56079223/custom-keras-data-generator-with-yield
        https://stackoverflow.com/questions/65928983/how-to-input-large-amount-of-data-for-training-keras-model-to-prevent-ram-crashi
        """
        # while True:
        start = 0
        end = self.batch_size

        num_samples = len(glob("./dataset/npy/train/input/*.npy"))

        while start < num_samples:
            train_inputs = []
            train_outputs = []

            if end > num_samples:
                end = num_samples
            
            for i in range(start, end - 1): 
                spec_in = np.load(f"./dataset/npy/train/input/padded_{i}.npy")
                spec_out = np.load(f"./dataset/npy/train/output/output_padded_{i}.npy")

                train_inputs.append(spec_in)
                train_outputs.append(spec_out)

            train_inputs = np.array(train_inputs)
            train_outputs = np.array(train_outputs)

            yield train_inputs, train_outputs

            start += self.batch_size
            end += self.batch_size
    
    def buildDataFromGenerator(self, dtype: tf.DType=tf.float64):
        train_dataset = tf.data.Dataset.from_generator(self.dataGenerator,
                                                       output_signature=(tf.TensorSpec(shape=(None, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS), dtype=dtype),
                                                                   tf.TensorSpec(shape=(None, INPUT_WIDTH, INPUT_HEIGHT, OUTPUT_CHANNELS), dtype=dtype)))

        return train_dataset
    
    def createCallbacks(self, checkpoint_dir: str=PATH_TO_MODEL, log_dir: str=PATH_TO_LOGS, early_stop_patience: int=PATIENCE) -> list:
        if not os.path.exists("./model/"):
            os.mkdir("./model")

        model_name = f'/bass_separator_{self.epochs}e_{self.batch_size}b.h5'

        checkpoint_dir = checkpoint_dir + model_name

        log_dir = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=early_stop_patience, monitor='loss'), 
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, verbose=1, save_best_only=False, save_freq="epoch")]

        return callbacks
    
    def modelFit(self, model: tf.keras.Model, callbacks, generated_data) -> tf.keras.callbacks.History:
        
        return model.fit(generated_data, batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks, verbose=1)
    
    def predictSong(self, path_to_song: str, name_pretrained: str=DEFAULT_MODEL, verbose: bool=True) -> tuple[np.array, np.array]:

        path_to_pretrained = PATH_TO_MODEL + "/" + name_pretrained

        pretrained = tf.keras.models.load_model(path_to_pretrained)
        
        song = WavHandler(wav=path_to_song)
        segmented = song.segmentWav()
        spectrograms = song.computeSTFT(segmented)
        padded = song.zeroPadSTFT(spectrograms)

        # istft = song.computeInverseSTFT(padded)

        if verbose:
            print("Sampling rate: ", song.sr)
            print("Number of segments: ", len(segmented))
            print("Spectrograms shape: ", spectrograms.shape)
            print("Padded spectrograms shape: ", padded.shape)
            print()

        output = pretrained.predict(padded, batch_size=self.batch_size, verbose=1)

        # Changes the shape
        swap_axes = np.swapaxes(output, 0, -1)
        swap_axes = np.swapaxes(swap_axes, 1, 2)
        swap_axes = np.swapaxes(swap_axes, 1, -1)

        bass = swap_axes[0]
        bassless = swap_axes[1]

        if verbose:
            print("Output shape: ", output.shape) # (x, 1040, 176, 2)
            print("Output shape (swapped axes): ", swap_axes.shape) # (2, x, 1040, 176)
            print("Bass output shape: ", bass.shape)
            print("Bassless output shape: ", bassless.shape)
            print()

        return bass, bassless
    
    def getOutputWavs(self, path_to_song: str, bass: np.array, bassless: np.array) -> tuple[WavHandler, WavHandler]:
        song = WavHandler(wav=path_to_song)

        bass_istft = song.computeInverseSTFT(bass)
        bassless_istft = song.computeInverseSTFT(bassless)

        return bass_istft, bassless_istft
    
    def saveOutputs(self, bass_istft: WavHandler, bassless_istft: WavHandler, path: str) -> None:
    
        bass_wav = bass_istft.wav
        bassless_wav = bassless_istft.wav
        sr = bass_istft.sr # Same with bassless_istft

        write(f"{path}_bass.wav", rate=sr, data=bass_wav)
        write(f"{path}_bassless.wav", rate=sr, data=bassless_wav)

        