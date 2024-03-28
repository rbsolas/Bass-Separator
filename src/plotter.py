import numpy as np
import matplotlib.pyplot as plt
import librosa # NOTE: librosa is built on top of matplotlib
import librosa.display
import random
import math

from scipy.io.wavfile import write
from glob import glob
from config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_HOP

class Plotter:
    sr = DEFAULT_SAMPLE_RATE
    hop = DEFAULT_HOP


    def plotWav(self, wav: np.array) -> None:
        librosa.display.waveshow(wav, sr=self.sr)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    def plotFFT(self, freq: np.array, magnitude: np.array) -> None:
        plt.plot(freq, magnitude)
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.show()

    def plotSpec(self, stft: np.array) -> None:
        librosa.display.specshow(stft, sr=self.sr, hop_length=self.hop)
        plt.gray()
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        # plt.savefig("stft", bbox_inches='tight', pad_inches=0)
    
    