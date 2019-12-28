# !/usr/bin/env python

""" Main program to test the psycho-acoustic encoder/decoder
"""

import os
import winsound

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# from audiocodec import codec
from audiocodec.tf import codec
from audiocodec import codec_utils


def play_sound(audio_filepath):
    print("Playing {0}...".format(audio_filepath), end=' ', flush=True)
    winsound.PlaySound(audio_filepath, winsound.SND_FILENAME)
    print("done")
    return


def read_wav(audio_filepath):
    """Read in wav file

    :param audio_filepath: path and filename of wav file
    :return:               sample_rate and raw wave data (#channels x #audio_samples)
    """
    sample_rate, wave_data = wav.read(audio_filepath)
    channels = len(wave_data.shape)
    if channels == 1:
        wave_data = wave_data[:, np.newaxis]  # add channels dimension 1
    wave_data = np.array(wave_data).T

    return sample_rate, wave_data


def sine_wav(amplitude, frequency):
    """Create wav which contains sine wave
    """
    sample_rate = 44100
    duration_sec = 2.0

    wave_data = amplitude * np.sin(2.0 * np.pi * frequency * np.arange(0, sample_rate*duration_sec) / sample_rate)
    wave_data = wave_data[np.newaxis, :]

    return sample_rate, wave_data


def main():
    # settings of program
    audio_filepath = './data/'
    audio_filename = 'asot_00.wav'   # 'high_clover.wav'

    if audio_filename:
        print("Audio file = ", audio_filepath + audio_filename)
        sample_rate, wave_data = read_wav(audio_filepath + audio_filename)
    else:
        audio_filename = 'sine.wav'
        print("Sine wave is input")
        # 44100 samples/s; 1024 samples/block; 44100/1024 blocks/s = f; 1024/44100 s/block = T;
        sample_rate, wave_data = sine_wav(20000, 10.05 * 44100 / 1024)

        wav.write(audio_filepath + audio_filename, sample_rate, np.int16(np.clip(wave_data.T, -2 ** 15, 2 ** 15 - 1)))

    # play_sound(audio_filepath + audio_filename)
    # limit wav length
    wave_data = wave_data[:, :2**23]  # ~3min

    # modify signal
    print('Modifying signal...')
    wave_data = codec_utils.modify_signal(wave_data, sample_rate, alpha=0.8)

    # (encode and) plot spectrum
    print('Plotting spectrum...')
    fig, ax = plt.subplots()
    ims = codec_utils.plot_spectrum(ax, wave_data, [0], range(10, 20), sample_rate, alpha=0.6)
    _ = animation.ArtistAnimation(fig, ims, interval=500)
    plt.show()

    # encode
    print('Encoding...')
    encoder_init = codec.encoder_setup(sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6)
    mdct_amplitudes_quantized, log_mask_thresholds_bark = codec.encoder(wave_data, encoder_init, quality=100)

    # decode
    print('Decoding...')
    wave_data_reconstructed = codec.decoder(mdct_amplitudes_quantized, log_mask_thresholds_bark, encoder_init)

    # write back to WAV file
    filepath, ext = os.path.splitext(audio_filepath + audio_filename)
    decoded_filepath = filepath + '_reconstructed.wav'
    wave_data = np.clip(wave_data_reconstructed.T, -2 ** 15, 2 ** 15 - 1)  # limit values in the array
    wav.write(decoded_filepath, sample_rate, np.int16(wave_data))

    play_sound(decoded_filepath)


if __name__ == "__main__":
    main()
