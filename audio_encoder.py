# !/usr/bin/env python

""" Main program to test the psycho-acoustic encoder/decoder
"""

import os
import winsound

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tensorflow as tf

# from audiocodec import codec_utils
from audiocodec.tf import codec

# todo: 1. port codec to tf
# todo: 1.1. chunking is audible due to delay in transform. Better to compute some overlap and stitch together
# todo: 1.2. port psychoacoustic to tf
# todo: 2. use codec in Donahue DCGAN
# todo: 2.1. experiment with adding in masking threshold noise and removing it (at both Generator & Discriminator stage)
# todo: 2.2. resolve how to work with psychoacoustic redundancy in audio representation (can we lower dim, or do we
# todo:      filter out redundancy before we pass it to the discriminator?
# todo:      2 options: work with redundant representation & filter psychoacoustic at start of Discriminator
# todo:      OR first train a NN to reduce the dimensionality of mdct amplitudes, then freeze and use in full network
# todo: 3. move to NVidia StyleGAN


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

    # modify signal
    # print('Modifying signal...')
    # wave_data = codec_utils.modify_signal(wave_data, sample_rate)

    # (encode and) plot spectrum
    # print('Plotting spectrum...')
    # channel = 0
    # fig, ax = plt.subplots()
    # ims = [codec_utils.plot_spectrum(ax, wave_data, channel, block, sample_rate) for block in range(10)]
    # _ = animation.ArtistAnimation(fig, ims, interval=500)
    # plt.show()

    # encode
    print('Encoding...')
    filter_bands_n = 1024  # note: the less filters we take, the more blocks we have in the signal
    encoder_init = codec.encoder_setup(sample_rate, 0.6, filter_bands_n, bark_bands_n=64)
    mdct_amplitudes = codec.encoder(wave_data, encoder_init, quality=100)

    # decode
    print('Decoding...')
    wave_data_reconstructed = codec.decoder(mdct_amplitudes, encoder_init)

    # with tf.Session() as sess:
    #     writer = tf.summary.FileWriter("output", sess.graph)
    #     result = sess.run(wave_data_reconstructed)
    #     writer.close()

    # write back to WAV file
    filepath, ext = os.path.splitext(audio_filepath + audio_filename)
    decoded_filepath = filepath + '_reconstructed.wav'
    wave_data = np.clip(wave_data_reconstructed.T, -2 ** 15, 2 ** 15 - 1)  # limit values in the array
    wav.write(decoded_filepath, sample_rate, np.int16(wave_data))

    play_sound(decoded_filepath)


if __name__ == "__main__":
    main()
