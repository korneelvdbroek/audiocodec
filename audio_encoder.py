# !/usr/bin/env python

""" Main program to test the psycho-acoustic encoder/decoder
"""

import os
import winsound

import tensorflow as tf
import numpy as np

import scipy.io.wavfile as wav
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from audiocodec import codec_utils, psychoacoustic, mdct, codec

# todo: 0. mdct: check for precision errors -- why do we get all this noisy stuff in the spectrogram... (not the phase!)
# todo: 1.1 move all psychoacoustic filter to the normalized mdct space (speedup...)
# todo: 1.2 add psychoacoustic filters to arsenal (noise-filter, zero filter, LReLU filter) (move trancegan code)

CPU_ONLY = False
DEBUG = False

# Set CPU as available physical device
if CPU_ONLY:
    print('Running Tensorflow on CPU only')
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=my_devices)

if DEBUG:
    tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
    # To find out which devices your operations and tensors are assigned to
    tf.debugging.set_log_device_placement(True)


def play_wav(wave_data, sample_rate):
    print("Playing wave data...", end=' ', flush=True)
    audio_filepath = "temp.wav"
    save_wav(audio_filepath, wave_data, sample_rate)
    winsound.PlaySound(audio_filepath, winsound.SND_FILENAME)
    os.remove(audio_filepath)
    print("done")
    return


def sine_wav(amplitude, frequency, sample_rate=44100, duration_sec=2.0):
    """Create wav which contains sine wave
    """
    wave_data = amplitude * np.sin(2.0 * np.pi * frequency * tf.range(0, sample_rate * duration_sec, dtype=tf.float32) / sample_rate)
    return tf.expand_dims(wave_data, axis=0), sample_rate


def load_wav(audio_filepath, sample_rate=None):
    """Read in wav file at given sample_rate.

    :param audio_filepath: path and filename of wav file
    :param sample_rate:    sample rate at which audio file needs to be read.
                           If sample_rate is None (default), then original sample rate is preserved
    :return:               raw wave data in range -1..1 (#channels x #audio_samples) and sample rate
    """
    print("Loading audio file {0}...".format(audio_filepath), end=' ', flush=True)
    wave_data, sample_rate = librosa.core.load(audio_filepath, sr=sample_rate, mono=False)

    if wave_data.ndim == 1:
        wave_data = np.reshape(wave_data, [1, wave_data.shape[0]])
    print('done (sample rate = {})'.format(sample_rate))
    return wave_data, sample_rate


def save_wav(audio_filepath, wave_data, sample_rate):
    wave_data = wave_data.numpy().T
    wave_data = np.clip(2**15 * wave_data, -2 ** 15, 2 ** 15 - 1)  # limit values in the array
    wav.write(audio_filepath, sample_rate, np.int16(wave_data))
    return


def clip_wav(start, stop, wave_data, sample_rate):
    minute_start, second_start = start
    minute_stop, second_stop = stop

    return wave_data[:, (minute_start*60+second_start)*sample_rate:(minute_stop*60+second_stop)*sample_rate]


def test_codec():
    # load audio file
    audio_filepath = './data/'
    audio_filename = 'asot_02_cosmos_sr8100_118_128'
    sample_rate = 90 * 90  # None   # 90*90
    wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)

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

    # cut up wav
    # wave_data_reconstructed = wave_data[:, 44100*(11*60 + 37):44100*(16*60+40)]

    # write back to WAV file
    play_wav(wave_data_reconstructed, sample_rate)
    filepath, ext = os.path.splitext(audio_filepath + audio_filename)
    decoded_filepath = filepath + '_reconstructed.wav'
    save_wav(decoded_filepath, wave_data_reconstructed, sample_rate)

    return


def test_psychoacoustic():
    # load audio file
    # audio_filepath = './data/'
    # audio_filename = 'sine'
    # sample_rate = 90 * 90  # None   # 90*90
    # wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)
    wave_data, sample_rate = sine_wav(1.0, 3.95*787.5, 44100, 1.0)

    play_wav(wave_data, sample_rate)

    # setup
    filter_bands_n = 210
    mdct_setup = mdct.setup(filter_bands_n)
    psychoacoustic_init = psychoacoustic.setup(sample_rate, filter_bands_n, bark_bands_n=24, alpha=0.6)

    # manipulate signal
    wave_data = wave_data[:, 0:filter_bands_n * int(wave_data.shape[1] / filter_bands_n)]
    spectrum = mdct.transform(wave_data, mdct_setup)
    spectrum_modified = psychoacoustic.psychoacoustic_filter(spectrum, psychoacoustic_init, drown=0.0)
    wave_reproduced = mdct.inverse_transform(spectrum_modified, mdct_setup)

    # plot both spectrograms
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    codec_utils.plot_spectrogram(ax1, spectrum, sample_rate, filter_bands_n)
    codec_utils.plot_spectrogram(ax2, spectrum_modified, sample_rate, filter_bands_n)
    plt.show()

    # plot time-slice
    if False:
        freq_bin_slice = 5
        plt.plot(spectrum_modified[0, freq_bin_slice, :])
        plt.plot(spectrum_modified[0, freq_bin_slice + 1, :])
        plt.plot(spectrum_modified[0, freq_bin_slice + 2, :])
        plt.show()

    play_wav(wave_reproduced, sample_rate)
    # save_wav(audio_filepath + audio_filename + '_reconstructed.wav', wave_reproduced.numpy(), sample_rate)

    return


def test_mdct():
    # load audio file
    audio_filepath = './data/'
    audio_filename = 'sine'   # 'asot_02_cosmos_sr8100_118_128.wav'
    sample_rate = 90*90  # None   # 90*90
    wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)

    # plot spectrum
    filter_bands_n = 90
    mdct_setup = mdct.setup(filter_bands_n, window_type='sine')

    wave_data = wave_data[:, 0:filter_bands_n * int(wave_data.shape[1] / filter_bands_n)]

    spectrum = mdct.transform(wave_data, mdct_setup)
    wave_reproduced = mdct.inverse_transform(spectrum, mdct_setup)

    fig, ax = plt.subplots(nrows=1)
    codec_utils.plot_spectrogram(ax, spectrum, sample_rate, filter_bands_n)
    plt.show()

    play_wav(wave_reproduced.numpy(), sample_rate)
    save_wav(audio_filepath + audio_filename + '_reconstructed.wav', wave_reproduced.numpy(), sample_rate)

    return


def main():
    # limit wav length to ~3min
    # wave_data = wave_data[:, :2 ** 23]
    #wave_data = clip_wav((1, 18), (1, 28), wave_data, sample_rate)

    test_psychoacoustic()

    exit()


if __name__ == "__main__":
    main()
