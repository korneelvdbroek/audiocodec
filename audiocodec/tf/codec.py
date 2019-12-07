# !/usr/bin/env python

""" Lossy audio encoder and decoder with psycho-acoustic model

Based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
"""

import numpy as np

import tensorflow as tf

from audiocodec.tf import mdct
from audiocodec import psychoacoustic


def encoder_setup(sample_rate, alpha, filter_bands_n=1024, bark_bands_n=64):
    H = mdct.polyphase_matrix(filter_bands_n)
    H_inv = mdct.inverse_polyphase_matrix(filter_bands_n)

    # W, W_inv = psychoacoustic.bark_freq_mapping(sample_rate, bark_bands_n, filter_bands_n)
    #
    # quiet_threshold = psychoacoustic.quiet_threshold_in_bark(sample_rate, bark_bands_n)
    # spreading_matrix = psychoacoustic.spreading_matrix_in_bark(sample_rate, bark_bands_n, alpha)

    return sample_rate, H, H_inv  # , W, W_inv, quiet_threshold, spreading_matrix, alpha


def encoder(wave_data, encoder_init, quality=1.0):
    """Audio encoder.
    Takes the raw wave data of the audio signal as input.
    Returns the discretized mdct amplitudes and the logarithms of the masking thresholds in the Bark scale. These allow
    to (re)construct the scale-factors applied to discretize the mdct amplitudes for each block

    :param wave_data:       signal wave data: each row is a channel (#channels x #samples)
    :param encoder_init:    initialization data for the encoder
    :param quality:         compression quality higher is better quality (default: 1.0)
    :return:                discretized mdct amplitudes (#channels x filter_bands_n x #blocks) and
                            logarithms of masking thresholds in the bark scale (#channels x #blocks x bark_bands_n)
    """
    with tf.name_scope('encoder'):
        sample_rate, H, H_inv, = encoder_init

        # 1. MDCT analysis filter bank
        mdct_amplitudes = mdct.transform(wave_data, H)

        # # 2. Masking threshold calculation
        # mask_thresholds_bark = psychoacoustic.global_masking_threshold_in_bark(mdct_amplitudes, W, spreading_matrix,
        #                          quiet_threshold, alpha, sample_rate) / quality
        #
        # # 3. Use masking threshold to discretize each mdct amplitude
        # # logarithmic discretization of masking threshold (#channels x #blocks x bark_bands_n)
        # log_mask_thresholds_bark = np.maximum(np.round(np.log2(mask_thresholds_bark) * 4), 0)
        #
        # scale_factors = psychoacoustic.scale_factors(log_mask_thresholds_bark, W_inv)
        # mdct_amplitudes_quantized = np.round(scale_factors * mdct_amplitudes)
        #
        # return mdct_amplitudes_quantized, log_mask_thresholds_bark

    return mdct_amplitudes


# def decoder(mdct_amplitudes_quantized, log_mask_thresholds_bark, encoder_init):
def decoder(mdct_amplitudes, encoder_init):
    """Audio decoder

    :param mdct_amplitudes_quantized: discretized mdct amplitudes (#channels x filter_bands_n x #blocks)
    :param log_mask_thresholds_bark:  logarithms of masking thresholds in Bark scale (#channels x#blocks xbark_bands_n)
    :param encoder_init:              initialization data for the encoder
    :return:                          signal wave data: each row is a channel (#channels x #samples)
    """
    with tf.name_scope('decoder'):
        sample_rate, H, H_inv = encoder_init

        # # 1. Compute and apply scale factor on discretized mdct amplitudes
        # scale_factors = psychoacoustic.scale_factors(log_mask_thresholds_bark, W_inv)
        #
        # mdct_amplitudes = mdct_amplitudes_quantized / scale_factors

        # 2. MDCT synthesis filter bank:
        wave_data = mdct.inverse_transform(mdct_amplitudes, H_inv)

    return wave_data


