import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import IPython.display as ipd
import musdb
import subprocess
import pydub
import tensorflow_addons as tfa
import time
import shutil
import ipdb
import scipy as sp
import soundfile as sf
import museval
import random
import math
from scipy.io.wavfile import write
from glob import glob

from config import PATH_TO_STEMS, PATH_TO_WAV
from wav_handler import WavHandler


class DatasetHandler:
    def __init__(self, path_to_stems: str=PATH_TO_STEMS, path_to_wav: str=PATH_TO_WAV) -> None:
        self.path_stems = path_to_stems
        self.path_wav = path_to_wav

    def stemToWav(self) -> None:
        """
        Create wav files of the stems if they don't exist yet
        """
        if len(glob(self.path_wav)) <= 0:
            subprocess.run(["musdbconvert", self.path_stems, self.path_wav])
        else:
            print("'wav' folder already exists!")
    
    def createBassless(self) -> None:
        """
        Creates another wav file called bassless.wav for each song in the train and test set
        """

        test_dir = glob(self.path_wav + r"\*")[0]
        train_dir = glob(self.path_wav + r"\*")[1]

        test_songs = glob(test_dir + r"\*")
        train_songs = glob(train_dir + r"\*")

        for folder in [test_songs, train_songs]:
            for song in folder:
                if len(glob(song + "\\bassless.wav")) <= 0:
                    drums, sr = librosa.load(song + "\\drums.wav", sr = None)
                    vocals, _ = librosa.load(song + "\\vocals.wav", sr = None)
                    other, _ = librosa.load(song + "\\other.wav", sr = None)

                    write('bassless.wav', sr, (drums + vocals + other)/3)
                else:
                    print("'bassless.wav' already exists!")

        def loadTrainInput(self):
            """
            Converts each input wav (mixture.wav) into WavHandler object and appends to an array to be returned
            """

            train_inputs = glob(self.path_wav + r"\train\mixture.wav")

            inputs = []

            for song in train_inputs:
                wav_handler = WavHandler(song)

                inputs.append(wav_handler)

            return inputs

        def loadTrainOutput(self):
            """
            Converts each expected output wav (bass.wav, bassless.wav) into WavHandler object and appends to an array to be returned
            """
            train_bass = glob(self.path_wav + r"\train\bass.wav")
            train_bassless = glob(self.path_wav + r"\train\bassless.wav")

            outputs = []

            for i in range(len(train_bass)): # train_bass and train_bassless have same length
                bass_wav = WavHandler(train_bass[i])
                bassless_wav = WavHandler(train_bassless[i])

                outputs.append((bass_wav, bassless_wav))

            return outputs

        def loadTest(self):
            """
            Converts each mixture.wav in test folder into WavHandler object and appends to an array to be returned
            """

            test_inputs = glob(self.path_wav + r"\test\mixture.wav")

            inputs = []

            for song in test_inputs:
                wav_handler = WavHandler(song)

                inputs.append(wav_handler)

            return inputs




