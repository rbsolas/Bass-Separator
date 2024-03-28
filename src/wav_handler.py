import numpy as np
import librosa # NOTE: librosa is built on top of matplotlib
import librosa.display
import random
import math

from scipy.io.wavfile import write
from glob import glob
from config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_HOP, ORIG_WIDTH, ORIG_HEIGHT, INPUT_WIDTH, INPUT_HEIGHT

class WavHandler: 
    """
    Object for individual wav files (not just segments)
    """

    def __init__(self, wav, sr: str=DEFAULT_SAMPLE_RATE) -> None:
        if type(wav) == str: # Provided path
            self.path = wav
            self.wav, self.sr = librosa.load(self.path, sr=None)
        else:
            self.wav = wav
            self.sr = sr

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

        # print("Padded shape: ", padded.shape)

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
    
    def computeFFT(self, wav: np.array, segmented: bool=True) -> tuple[np.array, np.array]: 
        """
        Computes FFT of each segment -> 'Snapshot' of the whole range and the average magnitudes throughout the signal
        """

        # segments = self.segmentWav() # Array of arrays

        if segmented:
            array_magnitudes = []
            array_freqs = []
            
            for segment in wav:
                fft = np.fft.fft(segment)
                magnitude = np.abs(fft) # 'Power' of each frequency
                freq = np.linspace(0, self.sr, len(magnitude)) # Sample rate determines the frequencies that can be captured

                left_magnitude = magnitude[:int(len(freq)/2)] 
                left_freq = freq[:int(len(freq)/2)] # Halved because we only need half of the sample rate to recreate the signal (Nyquist Theorem WOWOWOWOW)

                array_magnitudes.append(left_magnitude)
                array_freqs.append(left_freq)

            array_magnitudes = np.array(array_magnitudes)
            array_freqs = np.array(array_freqs)

            return (array_magnitudes, array_freqs)
        
        fft = np.fft.fft(wav)
        magnitude = np.abs(fft) # 'Power' of each frequency
        freq = np.linspace(0, self.sr, len(magnitude)) # Sample rate determines the frequencies that can be captured

        left_magnitude = magnitude[:int(len(freq)/2)] 
        left_freq = freq[:int(len(freq)/2)] # Halved because we only need half of the sample rate to recreate the signal (Nyquist Theorem WOWOWOWOW)

        left_magnitude = np.array(left_magnitude)
        left_freq = np.array(left_freq)

        return (left_magnitude, left_freq)
    
    def computeSTFT(self, wav: np.array, window: int=DEFAULT_WINDOW, hop: int=DEFAULT_HOP, segmented: bool=True) -> np.array: 
        """
        Computes STFT -> Computes the transforms of several windows throughout the signal, with the color representing the magnitude (similar to a heat map)
        
        Shape: (1025, 188) for 48000 kHz
        (1025, 173) for 44100 kHz
        """

        # segments = self.segmentWav()

        if segmented:
            array_spec = []

            for segment in wav:
                stft = librosa.core.stft(segment, n_fft=window, hop_length=hop)
                # spectrogram = np.abs(stft) # Linear magnitude representation

                array_spec.append(stft)
            
            array_spec = np.array(array_spec)

            return array_spec # 3D Array where each element is the STFT of each segment
        
        stft = librosa.core.stft(wav, n_fft=window, hop_length=hop)
        # spectrogram = np.abs(stft) # Linear magnitude representation

        return stft
    
    def computeMelSpec(self, wav: np.array, segmented: bool=True) -> np.array:
        """
        Gets amplitude (linear) from computed STFT and turns it to decibels (logarithmic). Note that phase is preserved
        """
        if segmented:
            wav = np.abs(wav)

            array_log_spec = librosa.amplitude_to_db(wav) # We make sense of sound logarithmically

            return array_log_spec
    
    def zeroPadSTFT(self, spectrograms: np.array, segmented: bool=True): 
        """
        Adds padding to each spectrogram to properly fit the model's input (1040, 176).
        """

        if segmented:
            _, num_rows, num_cols = spectrograms.shape

            m_pad = INPUT_WIDTH - num_rows
            n_pad = INPUT_HEIGHT - num_cols

            padded_array_spec = []

            for spec in spectrograms:
                padded = np.pad(spec, [(0, m_pad), (0, n_pad)], mode='constant')
                                                # Padding at first axis m (pad at start, at end) 
                                                        # Padding at second axis n (pad at start, at end)
                
                padded_array_spec.append(padded)
            
            return np.array(padded_array_spec)
        
    def unpadSTFT(self, spectrograms: np.array, width: int=ORIG_WIDTH, height: int=ORIG_HEIGHT, segmented: bool=True):
        """
        Removes padding that was added to fit the model's input layer dimensions to each spectrogram.
        """

        if segmented:
            unpadded = []

            for spec in spectrograms:
                unpad = spec[:width, :height]

                unpadded.append(unpad)

            return np.array(unpadded)

    def computeInverseSTFT(self, spectrograms: np.array, window: int=DEFAULT_WINDOW, hop: int=DEFAULT_HOP):
        """
        Computes the Inverse STFT of a song given its array of segment spectrograms. The magnitude info is extracted from each segment while the phase info is extracted from the original signal.
        """
        
        # Turn segments back to 2D array first after removing padding
        # print("3D Shape: ", spectrograms.shape)
        spectrograms = self.unpadSTFT(spectrograms)
        spectrograms = np.hstack(spectrograms) # Audio to reconstruct
        # print("Reconstructed spectrogram shape: ", spectrograms.shape)

        # Get original phase value from the song
        song_spec = self.computeSTFT(self.wav, segmented = False)
        # print("Orig spectrogram shape: ", song_spec.shape)

        # Get phase from the complex STFT, multiplies it element-wisely with the predicted spectrogram and uses ISTFT to get the reconstructed audio
        complex_segment_phases = librosa.magphase(song_spec)[1]
        # print("Phases array shape: ", complex_segment_phases.shape)

        # Removes the added silence during original segmentation of data
        spectrograms = spectrograms[:, :complex_segment_phases.shape[1]] 

        # Combines magnitude and phase information
        complex_valued_spectrogram = np.multiply(complex_segment_phases, np.abs(spectrograms)) 

        # print("Complex valued spec shape: ", complex_valued_spectrogram.shape)
        
        # ISTFT to move the audio from time-frequency domain to time-domain
        reconstructed = librosa.istft(complex_valued_spectrogram, hop_length = hop,  win_length = window) 
        
        return WavHandler(reconstructed)

