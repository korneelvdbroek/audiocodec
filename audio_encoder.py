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


# from audiocodec import codec
from audiocodec.tf import codec, mdct, psychoacoustic
from audiocodec import codec_utils

# todo: 2. move to [batches_n, blocks_n, filters_n, channels_n] format
# todo: 3. move all psychoacoustic filter to the normalized mdct space (speedup...)
# todo: 3. pull psychoacoustic filter in from trancegan code

CPU_ONLY = True

# Set CPU as available physical device
if CPU_ONLY:
    print('Running Tensorflow on CPU only')
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=my_devices)
    # To find out which devices your operations and tensors are assigned to
    # tf.debugging.set_log_device_placement(True)


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
    wave_data = amplitude * np.sin(2.0 * np.pi * frequency * np.arange(0, sample_rate*duration_sec) / sample_rate)
    wave_data = wave_data[np.newaxis, :]

    return wave_data, sample_rate


def load_wav(audio_filepath, sample_rate=None):
    """Read in wav file at given sample_rate. If sample_rate is None, then original sample rate is preserved

    :param audio_filepath: path and filename of wav file
    :param sample_rate:    sample rate at which audio file needs to be read
    :return:               raw wave data (#channels x #audio_samples) and sample rate
    """
    print("Loading audio file {0}...".format(audio_filepath), end=' ', flush=True)
    wave_data, sample_rate = librosa.core.load(audio_filepath, sr=sample_rate, mono=False)
    wave_data = wave_data * 2**15

    if wave_data.ndim == 1:
        wave_data = np.reshape(wave_data, [1, wave_data.shape[0]])
    print('done')
    return wave_data, sample_rate


def save_wav(audio_filepath, wave_data, sample_rate):
    wave_data = np.clip(wave_data.T, -2 ** 15, 2 ** 15 - 1)  # limit values in the array
    wav.write(audio_filepath, sample_rate, np.int16(wave_data))
    return


def clip_wav(start, stop, wave_data, sample_rate):
    minute_start, second_start = start
    minute_stop, second_stop = stop

    return wave_data[:, (minute_start*60+second_start)*sample_rate:(minute_stop*60+second_stop)*sample_rate]



def main():
    # load audio file
    audio_filepath = './data/'
    audio_filename = 'asot_02_cosmos_sr8100_118_128.wav'
    sample_rate = 8100
    wave_data, sample_rate = load_wav(audio_filepath + audio_filename, sample_rate)

    # limit wav length to ~3min
    # wave_data = wave_data[:, :2 ** 23]
    # wave_data = clip_wav((1, 18), (1, 28), wave_data, sample_rate)

    # play_wav(wave_data, sample_rate)

    # plot spectrum
    filter_bands_n = 90
    N, H, H_inv = mdct.setup(filter_bands_n)
    psychoacoustic_init = psychoacoustic.setup(sample_rate, filter_bands_n, bark_bands_n=24, alpha=0.6)

    spectrum = mdct.transform(wave_data, H)
    spectrum_modified = psychoacoustic.psychoacoustic_filter(spectrum, psychoacoustic_init, drown=0.0)

    # plot both spectrograms
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    image1 = codec_utils.plot_spectrogram(ax1, spectrum)
    image2 = codec_utils.plot_spectrogram(ax2, spectrum_modified)
    plt.show()

    # slice freq bin time axis
    if False:
        freq_bin_slice = 5
        plt.plot(spectrum_modified[0, freq_bin_slice, :])
        plt.plot(spectrum_modified[0, freq_bin_slice + 1, :])
        plt.plot(spectrum_modified[0, freq_bin_slice + 2, :])
        plt.show()

    # apply another mdct transform...
    spectrum2 = mdct.transform(spectrum_modified[0, :, :], H).numpy()
    # spectrum2 = np.sign(spectrum2) * (np.log(np.abs(spectrum2)) + 14.)
    vmin = np.amin(spectrum2)
    vmax = np.amax(spectrum2)
    # spectrum2 = [90=freq_bins=channels, 90, blocks]
    fig, axes = plt.subplots(nrows=2, ncols=5)
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(np.flip(spectrum2[:, :, i], axis=0), cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

    # modify spectrum2
    spectrum2_mod = np.where(np.abs(spectrum2) < 1000, 0, spectrum2)
    print(np.shape(spectrum2_mod), np.count_nonzero(spectrum2_mod))
    spectrum2_modified = mdct.inverse_transform(spectrum2_mod, H_inv)[:, 90:]
    spectrum2_modified = np.expand_dims(spectrum2_modified, axis=0)  # add channel dimension again
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    codec_utils.plot_spectrogram(ax1, spectrum)
    codec_utils.plot_spectrogram(ax2, spectrum_modified)
    codec_utils.plot_spectrogram(ax3, spectrum2_modified)
    plt.show()


    # reproduce original waveform
    wave_reproduced = mdct.inverse_transform(spectrum2_modified, H_inv)

    play_wav(wave_reproduced.numpy(), sample_rate)
    save_wav(audio_filepath + 'asot_02_cosmos_sr8100_118_128_reconstructed.wav', wave_reproduced.numpy(), sample_rate)

    exit()

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


if __name__ == "__main__":
    main()
