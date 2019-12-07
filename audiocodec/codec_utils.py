# !/usr/bin/env python

""" Utility functions

"""

import matplotlib.pyplot as plt
import numpy as np

from audiocodec import mdct, psychoacoustic


def modify_signal(wave_modified, sample_rate, filter_bands_n=1024, bark_bands_n=64):
    # Encode to mdct freq domain
    H = mdct.polyphase_matrix(filter_bands_n)
    mdct_amplitudes = mdct.transform(wave_modified, H)

    # Compute masking thresholds
    alpha = 0.6  # Exponent for non-linear superposition of spreading functions
    W, W_inv = psychoacoustic.bark_freq_mapping(sample_rate, bark_bands_n, filter_bands_n)
    spreading_matrix = psychoacoustic.spreading_matrix_in_bark(sample_rate, bark_bands_n, alpha)

    sound_threshold = np.transpose(psychoacoustic._mappingfrombark(
      psychoacoustic._masking_threshold_in_bark(
            mdct_amplitudes, W, spreading_matrix, alpha, sample_rate), W_inv), axes=(0, 2, 1))
    quiet_threshold_in_bark = psychoacoustic.quiet_threshold_in_bark(sample_rate, bark_bands_n)
    quiet_threshold = psychoacoustic._mappingfrombark(quiet_threshold_in_bark, W_inv)
    total_threshold = np.transpose(psychoacoustic._mappingfrombark(
      psychoacoustic.global_masking_threshold_in_bark(
            mdct_amplitudes, W, spreading_matrix, quiet_threshold_in_bark, alpha, sample_rate), W_inv), axes=(0, 2, 1))

    # Update signal
    # 1. remove anything below masking threshold
    mdct_modified = np.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, mdct_amplitudes, 1e-6)
    # 2. pass-through
    # mdct_modified = mdct_amplitudes
    # 3. put in masking threshold
    # mdct_modified = np.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, mdct_amplitudes, total_threshold)
    # 4. keep only masking threshold
    # mdct_modified = total_threshold
    # 5. keep only sound below masking threshold
    # mdct_modified = np.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, 1e-6, mdct_amplitudes)
    compressed = np.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, 0, 1)
    print("  compression = {0} / {1} = {2:.0f}%".format(compressed.sum(), np.prod(np.asarray(compressed.shape)),
                                                        100 * compressed.sum() / np.prod(np.asarray(compressed.shape))))

    # Decode to time-domain
    H_inv = mdct.inverse_polyphase_matrix(filter_bands_n)
    wave_modified = mdct.inverse_transform(mdct_modified, H_inv)

    return wave_modified


def plot_spectrum(ax, wave_data, channel, block, sample_rate, filter_bands_n=1024, bark_bands_n=64):
    """Plots the mdct spectrum with quiet and masking threshold for a given channel and block

    :param ax:              matplotlib axes
    :param wave_data:       raw audio signal
    :param channel:         channel to plot
    :param block:           block number to plot
    :param sample_rate:     sample rate
    :param filter_bands_n:  number of mdct filters
    :param bark_bands_n:    number of bark bands
    :return:                matplotlib plot
    """
    # 1. MDCT analysis filter bank
    H = mdct.polyphase_matrix(filter_bands_n)
    mdct_amplitudes = mdct.transform(wave_data, H)

    # 2. Masking threshold calculation
    alpha = 0.6  # Exponent for non-linear superposition of spreading functions
    W, W_inv = psychoacoustic.bark_freq_mapping(sample_rate, bark_bands_n, filter_bands_n)
    spreading_matrix = psychoacoustic.spreading_matrix_in_bark(sample_rate, bark_bands_n, alpha)

    # Compute masking thresholds
    sound_threshold = psychoacoustic._mappingfrombark(
      psychoacoustic._masking_threshold_in_bark(mdct_amplitudes, W, spreading_matrix, alpha, sample_rate), W_inv)
    quiet_threshold = psychoacoustic._mappingfrombark(
      psychoacoustic.quiet_threshold_in_bark(sample_rate, bark_bands_n), W_inv)

    # Plot
    max_frequency = sample_rate / 2
    filter_bands_n = mdct_amplitudes.shape[1]
    filter_band_width = max_frequency / filter_bands_n
    filter_bands_mid_freq = np.arange(filter_bands_n) * filter_band_width + filter_band_width / 2

    image_amplitudes, = ax.plot(np.log10(filter_bands_mid_freq),
                                10 * np.log10(mdct_amplitudes[channel, :, block] ** 2.0), color='blue')

    image_masking_threshold, = plt.plot(np.log10(filter_bands_mid_freq),
                                        10 * np.log10(sound_threshold[channel, block, :] ** 2.0), color='green')

    image_quiet_threshold,  = plt.plot(np.log10(filter_bands_mid_freq),
                                       10 * np.log10(quiet_threshold ** 2.0), color='black')

    ax.set_xlabel("log10 of frequency")
    ax.set_ylabel("intensity [dB]")
    ax.set_ylim(ymin=-50)

    return [image_amplitudes, image_masking_threshold, image_quiet_threshold]
