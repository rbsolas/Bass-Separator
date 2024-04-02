import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io.wavfile import write
from glob import glob

PATH_TO_SRC_FOLDER = r".\src"

if PATH_TO_SRC_FOLDER not in sys.path:
    sys.path.append(PATH_TO_SRC_FOLDER)

from wav_handler import WavHandler
from dataset_handler import DatasetHandler
from plotter import Plotter
from model_handler import ModelHandler
from config import DEFAULT_MODEL


if __name__ == "__main__":
    model_handler = ModelHandler()

    song_path = sys.argv[1]
    
    song_path_as_list = song_path.split("\\") # Song path must be separated by forward slashes and ends in *.wav

    if len(song_path_as_list) == 0:
        sys.exit("Improper path to .wav file provided. Please make sure that the path is separated by '\\' and ends in '.wav'")

    song_name = song_path_as_list[-1][:-4]

    print(f"No model name provided. Using {DEFAULT_MODEL} to predict.")
    bass, bassless = model_handler.predictSong(song_path)


    bass_istft, bassless_istft = model_handler.getOutputWavs(song_path, bass, bassless)
    model_handler.saveOutputs(bass_istft, bassless_istft, song_name)