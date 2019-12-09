# !/usr/bin/env python

""" Lossy audio encoder and decoder with psycho-acoustic model

Based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
"""

import numpy as np

import tensorflow as tf

from audiocodec.tf import mdct
from audiocodec import psychoacoustic

MAX_CHUNK_SIZE = 8 * 2 ** 20  # 8 MiB; needs to be multiple of filter_band_n


def encoder_setup(sample_rate, alpha, filter_bands_n=1024, bark_bands_n=64):
    H = mdct.polyphase_matrix(filter_bands_n)
    H_inv = mdct.inverse_polyphase_matrix(filter_bands_n)

    # W, W_inv = psychoacoustic.bark_freq_mapping(sample_rate, bark_bands_n, filter_bands_n)
    #
    # quiet_threshold = psychoacoustic.quiet_threshold_in_bark(sample_rate, bark_bands_n)
    # spreading_matrix = psychoacoustic.spreading_matrix_in_bark(sample_rate, bark_bands_n, alpha)

    return sample_rate, filter_bands_n, H, H_inv  # , W, W_inv, quiet_threshold, spreading_matrix, alpha


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
    _, filter_bands_n, _, _ = encoder_init

    # chunk up the wave, since tf graph involves some complex64 inside the dct which can blow up the gpu memory
    channels_n, samples_n = wave_data.shape

    max_samples_in_chunk = int(MAX_CHUNK_SIZE / (4 * channels_n))  # 4 since tf.float32
    chunks_n = int(np.ceil(samples_n / max_samples_in_chunk))

    # chunk wave with overlap of filter_bands_n at end (since mdct introduces delay of filter_bands_n)
    wave_chunks = [
        wave_data[:, n * max_samples_in_chunk:(n + 1) * max_samples_in_chunk + filter_bands_n]
        for n in range(0, chunks_n)]

    # make a dataset from a numpy array
    def wave_chunk_generator():
        return iter(wave_chunks)

    dataset = tf.data.Dataset.from_generator(wave_chunk_generator, tf.float32, tf.TensorShape([None, None]))

    # create the iterator
    wave_iter = dataset.make_one_shot_iterator()
    wave_chunk = wave_iter.get_next()

    # encode
    wave_float = tf.dtypes.cast(wave_chunk, dtype=tf.float32)
    mdct_chunk = _encode_chunk(wave_float, encoder_init, quality)

    mdct_chunks = []
    with tf.Session() as sess:
        while True:
            try:
                # writer = tf.summary.FileWriter("output", sess.graph)
                value = sess.run(mdct_chunk)[:, :, 1:-1]
                # writer.close()
                mdct_chunks.append(value)
            except tf.errors.OutOfRangeError:
                break
    mdct_amplitudes = np.concatenate(mdct_chunks, axis=2)

    return mdct_amplitudes


def _encode_chunk(wave_data, encoder_init, quality):
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
        sample_rate, filter_bands_n, H, H_inv = encoder_init

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
    # chunk up the wave, since tf graph involves some complex64 inside the dct which can blox up the gpu memory
    channels_n, filter_bands_n, blocks_n = mdct_amplitudes.shape

    max_blocks_in_chunk = int(MAX_CHUNK_SIZE / (4 * channels_n * filter_bands_n))  # 4 since tf.float32
    chunks_n = int(np.ceil(blocks_n / max_blocks_in_chunk))

    # pad mdct at beginning, since we chunk with overlap at start and end
    # mdct_padded = np.concatenate([mdct_amplitudes, np.zeros((channels_n, filter_bands_n, 1))], axis=2)
    mdct_padded = mdct_amplitudes

    # print("max_blocks_in_chunk = ", max_blocks_in_chunk)

    # chunk wave with overlap of filter_bands_n at start and end (since mdct introduces delay of filter_bands_n)
    mdct_chunks = [
        mdct_padded[:, :, n * max_blocks_in_chunk:(n + 1) * max_blocks_in_chunk + 1]
        for n in range(0, chunks_n)]

    # make a dataset from a numpy array
    def mdct_chunk_generator():
        return iter(mdct_chunks)

    dataset = tf.data.Dataset.from_generator(mdct_chunk_generator, tf.float32, tf.TensorShape([None, None, None]))

    # create the iterator
    mdct_iter = dataset.make_one_shot_iterator()
    mdct_chunk = mdct_iter.get_next()

    # encode
    wave_chunk = _decode_chunk(mdct_chunk, encoder_init)

    wave_chunks = []
    with tf.Session() as sess:
        while True:
            try:
                # writer = tf.summary.FileWriter("output", sess.graph)
                value = sess.run(wave_chunk)[:, 2 * filter_bands_n:]
                # writer.close()
                wave_chunks.append(value)
            except tf.errors.OutOfRangeError:
                break
    wave_data = np.concatenate(wave_chunks, axis=1)

    return wave_data


# def decoder(mdct_amplitudes_quantized, log_mask_thresholds_bark, encoder_init):
def _decode_chunk(mdct_amplitudes, encoder_init):
    """Audio decoder

    :param mdct_amplitudes_quantized: discretized mdct amplitudes (#channels x filter_bands_n x #blocks)
    :param log_mask_thresholds_bark:  logarithms of masking thresholds in Bark scale (#channels x#blocks xbark_bands_n)
    :param encoder_init:              initialization data for the encoder
    :return:                          signal wave data: each row is a channel (#channels x #samples)
    """
    with tf.name_scope('decoder'):
        sample_rate, filter_bands_n, H, H_inv = encoder_init

        # # 1. Compute and apply scale factor on discretized mdct amplitudes
        # scale_factors = psychoacoustic.scale_factors(log_mask_thresholds_bark, W_inv)
        #
        # mdct_amplitudes = mdct_amplitudes_quantized / scale_factors

        # 2. MDCT synthesis filter bank:
        wave_data = mdct.inverse_transform(mdct_amplitudes, H_inv)

    return wave_data


