# !/usr/bin/env python

""" Lossy audio encoder and decoder with psycho-acoustic model
"""

import numpy as np

import tensorflow as tf

from audiocodec import psychoacoustic, mdct

MAX_CHUNK_SIZE = 8 * 2 ** 20  # 8 MiB; needs to be multiple of filter_band_n


def encoder_setup(sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6):
  """Computes required initialization matrices (stateless, no OOP)

  :param sample_rate:       sample_rate
  :param alpha:             exponent for non-linear superposition (~0.6)
  :param filter_bands_n:    number of filter bands of the filter bank
  :param bark_bands_n:      number of bark bands
  :return:                  tuple with pre-computed required for encoder and decoder
  """
  mdct_setup = mdct.setup(filter_bands_n)

  pa_setup = psychoacoustic.setup(sample_rate, filter_bands_n, bark_bands_n, alpha)

  return mdct_setup, pa_setup


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
  (filter_bands_n, _, _), _ = encoder_init

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

  # create dataset iterator
  wave_iter = dataset.make_one_shot_iterator()
  wave_chunk = wave_iter.get_next()

  # encode
  wave_float = tf.dtypes.cast(wave_chunk, dtype=tf.float32)
  mdct_chunk, mask_chunk = _encode_chunk(wave_float, encoder_init, quality)

  mdct_chunks = []
  mask_chunks = []
  with tf.Session() as sess:
    while True:
      try:
        # writer = tf.summary.FileWriter("output", sess.graph)
        value = sess.run([mdct_chunk, mask_chunk])
        # writer.close()
        mdct_chunks.append(value[0][:, :, 1:-1])
        mask_chunks.append(value[1][:, 1:-1, :])
      except tf.errors.OutOfRangeError:
        break
  mdct_amplitudes_quantized = np.concatenate(mdct_chunks, axis=2)
  log_mask_thresholds_bark = np.concatenate(mask_chunks, axis=1)

  return mdct_amplitudes_quantized, log_mask_thresholds_bark


def decoder(mdct_amplitudes_quantized, log_mask_thresholds_bark, encoder_init):
  """Audio decoder

  :param mdct_amplitudes_quantized: discretized mdct amplitudes (#channels x filter_bands_n x #blocks)
  :param log_mask_thresholds_bark:  logarithms of masking thresholds in Bark scale (#channels x#blocks xbark_bands_n)
  :param encoder_init:              initialization data for the encoder
  :return:                          signal wave data: each row is a channel (#channels x #samples)
  """
  # chunk up the wave, since tf graph involves some complex64 inside the dct which can blox up the gpu memory
  channels_n, filter_bands_n, blocks_n = mdct_amplitudes_quantized.shape

  max_blocks_in_chunk = int(MAX_CHUNK_SIZE / (4 * channels_n * filter_bands_n))  # 4 since tf.float32
  chunks_n = int(np.ceil(blocks_n / max_blocks_in_chunk))

  # chunk wave with overlap of filter_bands_n at end, since mdct introduces delay of 1 block (=filter_bands_n)
  mdct_chunks = [
    mdct_amplitudes_quantized[:, :, n * max_blocks_in_chunk:(n + 1) * max_blocks_in_chunk + 1]
    for n in range(0, chunks_n)]
  mask_chunks = [
    log_mask_thresholds_bark[:, n * max_blocks_in_chunk:(n + 1) * max_blocks_in_chunk + 1, :]
    for n in range(0, chunks_n)]

  # make a dataset from a numpy array
  def chunk_generator():
    for mdct_chunk_gen, mask_chunk_gen in zip(mdct_chunks, mask_chunks):
      yield (mdct_chunk_gen, mask_chunk_gen)

  dataset = tf.data.Dataset.from_generator(chunk_generator, (tf.float32, tf.float32),
                                           (tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None])))

  # create dataset iterator
  iterator = dataset.make_one_shot_iterator()
  mdct_chunk, mask_chunk = iterator.get_next()

  # decode
  wave_chunk = _decode_chunk(mdct_chunk, mask_chunk, encoder_init)

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
    (_, H, _), pa_setup = encoder_init
    _, _, W_inv, _, _, _ = pa_setup

    # 1. MDCT analysis filter bank
    mdct_amplitudes = mdct.transform(wave_data, H)

    # 2. Masking threshold calculation
    mask_thresholds_bark = psychoacoustic._global_masking_threshold_in_bark(mdct_amplitudes, pa_setup) / quality

    # 3. Use masking threshold to discretize each mdct amplitude
    # logarithmic discretization of masking threshold (#channels x #blocks x bark_bands_n)
    log_mask_thresholds_bark = tf.maximum(tf.round(4. * tf.log(mask_thresholds_bark)/tf.log(2.)), 0.)

    scale_factors = psychoacoustic.scale_factors(log_mask_thresholds_bark, W_inv)

    mdct_amplitudes_quantized = tf.round(scale_factors * mdct_amplitudes)

  return mdct_amplitudes_quantized, log_mask_thresholds_bark


def _decode_chunk(mdct_amplitudes_quantized, log_mask_thresholds_bark, encoder_init):
  """Audio decoder

  :param mdct_amplitudes_quantized: discretized mdct amplitudes (#channels x filter_bands_n x #blocks)
  :param log_mask_thresholds_bark:  logarithms of masking thresholds in Bark scale (#channels x#blocks xbark_bands_n)
  :param encoder_init:              initialization data for the encoder
  :return:                          signal wave data: each row is a channel (#channels x #samples)
  """
  with tf.name_scope('decoder'):
    (_, _, H_inv), (_, _, W_inv, _, _, _) = encoder_init

    # 1. Compute and apply scale factor on discretized mdct amplitudes
    scale_factors = psychoacoustic.scale_factors(log_mask_thresholds_bark, W_inv)

    mdct_amplitudes = mdct_amplitudes_quantized / scale_factors

    # 2. MDCT synthesis filter bank:
    wave_data = mdct.inverse_transform(mdct_amplitudes, H_inv)

  return wave_data


