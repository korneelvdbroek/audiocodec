# !/usr/bin/env python

""" Utility functions

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from audiocodec.tf import mdct, psychoacoustic


def modify_signal(wave_data_np, sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6):
    wave_data = tf.convert_to_tensor(wave_data_np, dtype=tf.float32)

    # Encode to mdct freq domain
    H = mdct.polyphase_matrix(filter_bands_n)
    mdct_amplitudes = mdct.transform(wave_data, H)

    # Compute masking thresholds
    W, W_inv = psychoacoustic._bark_freq_mapping(sample_rate, bark_bands_n, filter_bands_n)
    spreading_matrix = psychoacoustic.spreading_matrix_in_bark(sample_rate, bark_bands_n, alpha)
    quiet_threshold_in_bark = psychoacoustic.quiet_threshold_in_bark(sample_rate, bark_bands_n)

    sound_threshold = tf.transpose(psychoacoustic._mappingfrombark(
      psychoacoustic._masking_threshold_in_bark(
            mdct_amplitudes, W, spreading_matrix, alpha, sample_rate), W_inv), perm=[0, 2, 1])
    quiet_threshold = psychoacoustic._mappingfrombark(quiet_threshold_in_bark, W_inv)
    total_threshold = tf.transpose(psychoacoustic._mappingfrombark(
      psychoacoustic.global_masking_threshold_in_bark(
            mdct_amplitudes, W, spreading_matrix, quiet_threshold_in_bark, alpha, sample_rate), W_inv), perm=[0, 2, 1])

    # Update signal
    # 1. remove anything below masking threshold
    mdct_modified = tf.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, mdct_amplitudes, tf.fill(tf.shape(mdct_amplitudes), 1e-6))
    # 2. pass-through
    # mdct_modified = mdct_amplitudes
    # 3. put in masking threshold
    # mdct_modified = np.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, mdct_amplitudes, total_threshold)
    # 4. keep only masking threshold
    # mdct_modified = total_threshold
    # 5. keep only sound below masking threshold
    # mdct_modified = np.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, 1e-6, mdct_amplitudes)
    compressed = tf.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, tf.zeros(tf.shape(mdct_amplitudes)), tf.ones(
        tf.shape(mdct_amplitudes)))
    mdct_modified_out = tf.Print(mdct_modified,
                          [tf.reduce_sum(compressed),
                           tf.reduce_prod(tf.shape(compressed)),
                           100. * tf.reduce_sum(compressed) / tf.reduce_prod(tf.shape(compressed, out_type=tf.float32))])

    # Decode to time-domain
    H_inv = mdct.inverse_polyphase_matrix(filter_bands_n)
    wave_modified = mdct.inverse_transform(mdct_modified_out, H_inv)

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter("output", sess.graph)
        wave_modified_out = sess.run(wave_modified)
        # writer.close()

    return wave_modified_out


def plot_spectrum(ax, wave_data_np, channels, blocks, sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6):
    """Plots the mdct spectrum with quiet and masking threshold for a given channel and block

    :param ax:              matplotlib axes
    :param wave_data:       raw audio signal
    :param channel:         channel to plot
    :param block:           block number to plot
    :param sample_rate:     sample rate
    :param filter_bands_n:  number of mdct filters
    :param bark_bands_n:    number of bark bands
    :param alpha            exponent for non-linear superposition of spreading functions
    :return:                image frames
    """
    wave_data = tf.convert_to_tensor(wave_data_np, dtype=tf.float32)

    # 1. MDCT analysis filter bank
    H = mdct.polyphase_matrix(filter_bands_n)
    mdct_amplitudes = mdct.transform(wave_data, H)

    # 2. Masking threshold calculation
    W, W_inv = psychoacoustic._bark_freq_mapping(sample_rate, bark_bands_n, filter_bands_n)
    spreading_matrix = psychoacoustic.spreading_matrix_in_bark(sample_rate, bark_bands_n, alpha)

    # Compute masking thresholds
    sound_threshold = psychoacoustic._mappingfrombark(
      psychoacoustic._masking_threshold_in_bark(mdct_amplitudes, W, spreading_matrix, alpha, sample_rate), W_inv)
    quiet_threshold = psychoacoustic._mappingfrombark(
      psychoacoustic.quiet_threshold_in_bark(sample_rate, bark_bands_n), W_inv)

    # compute quantities to be plot
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
