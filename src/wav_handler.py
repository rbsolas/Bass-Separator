import numpy as np
import librosa
import librosa.display
import random
import math
from scipy.io.wavfile import write
from glob import glob

from config import DEFAULT_WINDOW, DEFAULT_HOP, IMG_WIDTH, IMG_HEIGHT

class WavHandler: 
    """
    Object for individual wav files (not just segments)
    """

    def __init__(self, path_to_wav: str) -> None:
        self.path = path_to_wav
        self.wav, self.sr = librosa.load(self.path, sr=None)

    def padLength(self) -> int: 
        """"
        Number of zeros to add as padding
        """
        pad = self.sr*2 - len(self.wav) % (self.sr*2)

        return pad

    def zeroPad(self) -> np.array: 
        """
        Adds padding to make song divisible into 2-second segments
        """

        padLength = self.padLength()

        pad = np.zeros(padLength)
        padded = np.append(self.wav, pad)

        return padded
    
    def numSegments(self) -> int: 
        """
        Number of segments AFTER padding
        """

        padded = self.zeroPad()

        twoSec = self.sr * 2

        return len(padded) / int(twoSec)
    
    def segmentWav(self) -> np.array: 
        """
        Divides song into 2-second segments
        """

        numSegments = self.numSegments()
        padded = self.zeroPad()

        return np.split(padded, numSegments)
    
    def computeFFT(self) -> tuple[np.array, np.array]: 
        """
        Computes FFT of each segment
        """

        segments = self.segmentWav() # Array of arrays

        #array_magnitudes = []
        #array_freqs = []

        #for segment in segments:
        fft = np.fft.fft(segments)
        magnitude = np.abs(fft) # 'Power' of each frequency
        freq = np.linspace(0, self.sr, len(magnitude)) # Sample rate determines the frequencies that can be captured

        left_magnitude = magnitude[:int(len(freq)/2)] 
        left_freq = freq[:int(len(freq)/2)] # Halved because we only need half of the sample rate to recreate the signal (Nyquist Theorem WOWOWOWOW)

        #array_magnitudes.append(left_magnitude)
        #array_freqs.append(left_freq)

        #array_magnitudes = np.array(array_magnitudes)
        #array_freqs = np.array(array_freqs)

        return (left_magnitude, left_freq)
    
    def computeSTFT(self, window: int=DEFAULT_WINDOW, hop: int=DEFAULT_HOP) -> np.array: 
        """
        Computes STFT
        """

        segments = self.segmentWav()

        #array_spec = []

        #for segment in segments:
        stft = librosa.core.stft(segments, n_fft=window, hop_length=hop)
        spectrogram = np.abs(stft) # Linear

        #array_spec.append(spectrogram)
        
        #array_spec = np.array(array_spec)

        return spectrogram # 3D Array where each element is the STFT of each segment
    
    def computeMelSpec(self) -> np.array:
        """
        Turns computed STFT from amplitude (linear) to decibels (logarithmic)
        """

        array_spec = self.computeSTFT()

        array_log_spec = librosa.amplitude_to_db(array_spec) # We make sense of sound logarithmically

        return array_log_spec
    
    def zeroPadSpec(self): 
        """
        Adds padding to each spectrogram to properly fit the model's input (1040, 176). Call this instead of computeSpec for the input data.
        """

        array_spec = self.computeSpec()

        num_specs, num_rows, num_cols = array_spec.shape

        m_pad = IMG_WIDTH - num_rows
        n_pad = IMG_HEIGHT - num_cols

        padded_array_spec = []

        for spec in array_spec:
            padded = np.pad(spec, [(0, m_pad), (0, n_pad)], mode='constant')
                                            # Padding at first axis m (pad at start, at end) 
                                                    # Padding at second axis n (pad at start, at end)
            
            padded_array_spec.append(padded)
        
        return np.array(padded_array_spec)

    def computeInverseSTFT(self, spectrograms: np.array, window: int=DEFAULT_WINDOW, hop: int=DEFAULT_HOP):

        spectrograms = np.hstack(spectrograms) # Audio to reconstruct
        print("Reconstructed spectrogram shape: ", spectrograms.shape)

        # Get original phase value from the song
        segments_spec = self.computeSTFT()
        song_spec = np.hstack(segments_spec) # Original audio
        print("Orig spectrogram shape: ", song_spec.shape)

        # Get phase from the complex STFT, multiplies it element-wisely with the predicted spectrogram and uses ISTFT to get the reconstructed audio
        complex_segment_phases = librosa.magphase(song_spec)[1]
        print("Phases array shape: ", complex_segment_phases.shape)

        spectrograms = spectrograms[:, :complex_segment_phases.shape[1]] # Removes the added silence during original segmentation of data
        complex_valued_spectrogram = np.multiply(complex_segment_phases, spectrograms)

        print("Complex valued spec: ", complex_valued_spectrogram.shape)
        
        reconstructed = librosa.istft(complex_valued_spectrogram, hop_length = hop,  win_length = window) # ISTFT to move the audio from time-frequency domain to time-domain

        return reconstructed

