# !/usr/bin/env python

""" Utility functions

"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from audiocodec import psychoacoustic, mdct


def modify_signal(wave_data_np, sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6):
    wave_data = tf.convert_to_tensor(wave_data_np, dtype=tf.float32)

    filter_bands_n, H, H_inv = mdct.setup(filter_bands_n)
    pa_setup = psychoacoustic.setup(sample_rate, filter_bands_n, bark_bands_n, alpha)

    # Encode to mdct freq domain
    mdct_amplitudes = mdct.transform(wave_data, H)

    mdct_modified = psychoacoustic.psychoacoustic_filter(mdct_amplitudes, pa_setup)

    # Decode to time-domain
    wave_modified = mdct.inverse_transform(mdct_modified, H_inv)

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter("output", sess.graph)
        wave_modified_out = sess.run(wave_modified)
        # writer.close()

    return wave_modified_out


def plot_spectrogram(ax, mdct_amplitudes, channel=0):
    spectrum_norm = mdct.normalize_mdct(mdct_amplitudes)
    image = ax.imshow(np.flip(np.transpose(spectrum_norm[channel, :, :]), axis=0), cmap='gray', vmin=-1., vmax=1., interpolation='none')
    return image


def plot_spectrum(ax, wave_data_np, channels, blocks, sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6):
    """Plots the mdct spectrum with quiet and masking threshold for a given channel and block

    :param ax:              matplotlib axes
    :param wave_data_np:    raw audio signal
    :param channels:        channels to plot
    :param blocks:          block numbers to plot
    :param sample_rate:     sample rate
    :param filter_bands_n:  number of mdct filters
    :param bark_bands_n:    number of bark bands
    :param alpha            exponent for non-linear superposition of spreading functions
    :return:                image frames
    """
    wave_data = tf.convert_to_tensor(wave_data_np, dtype=tf.float32)

    filter_bands_n, H, H_inv = mdct.setup(filter_bands_n)
    sample_rate, W, W_inv, quiet_threshold_in_bark, spreading_matrix, alpha = psychoacoustic.setup(sample_rate,
                                                                                                   filter_bands_n,
                                                                                                   bark_bands_n, alpha)

    # 1. MDCT analysis filter bank
    mdct_amplitudes = mdct.transform(wave_data, H)

    # 2. Masking threshold calculation
    sound_threshold = psychoacoustic._mappingfrombark(
      psychoacoustic._masking_threshold_in_bark(mdct_amplitudes, W, spreading_matrix, alpha, sample_rate), W_inv)
    quiet_threshold = psychoacoustic._mappingfrombark(quiet_threshold_in_bark, W_inv)

    # 3. Compute quantities to be plot
    max_frequency = sample_rate / 2
    filter_bands_n = tf.dtypes.cast(tf.shape(mdct_amplitudes)[1], dtype=tf.float32)
    filter_band_width = max_frequency / filter_bands_n
    filter_bands_mid_freq = filter_band_width * tf.range(filter_bands_n) + filter_band_width / 2

    x = tf.log(filter_bands_mid_freq) / tf.log(10.)
    y_amplitudes = 10. * tf.log(mdct_amplitudes ** 2.0) / tf.log(10.)
    y_masking_threshold = 10. * tf.log(sound_threshold ** 2.0) / tf.log(10.)
    y_quiet_threshold = 10. * tf.log(quiet_threshold ** 2.0) / tf.log(10.)

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter("output", sess.graph)
        [x_out, y_amplitudes_out, y_masking_threshold_out, y_quiet_threshold_out] = sess.run([x, y_amplitudes, y_masking_threshold, y_quiet_threshold])
        # writer.close()

    # make images
    image_frames = []
    for channel in channels:
        for block in blocks:

            image_amplitudes, = ax.plot(x_out, y_amplitudes_out[channel, :, block], color='blue')
            image_masking_threshold, = plt.plot(x_out, y_masking_threshold_out[channel, block, :], color='green')
            image_quiet_threshold,  = plt.plot(x_out, y_quiet_threshold_out[0, 0, :], color='black')

            ax.set_xlabel("log10 of frequency")
            ax.set_ylabel("intensity [dB]")
            ax.set_ylim(ymin=-50)

            image_frames.append([image_amplitudes, image_masking_threshold, image_quiet_threshold])

    return image_frames
