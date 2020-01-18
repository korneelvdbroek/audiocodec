# !/usr/bin/env python

""" Utility functions

"""

import imageio

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

  mdct_modified = psychoacoustic.zero_filter(mdct_amplitudes, pa_setup)

  # Decode to time-domain
  wave_modified = mdct.inverse_transform(mdct_modified, H_inv)

  with tf.Session() as sess:
    # writer = tf.summary.FileWriter("output", sess.graph)
    wave_modified_out = sess.run(wave_modified)
    # writer.close()

  return wave_modified_out


def plot_spectrogram(ax, mdct_amplitudes, sample_rate, filter_bands_n, channel=0):
  mdct_norm = psychoacoustic.ampl_to_norm(mdct_amplitudes)
  image = ax.imshow(np.flip(np.transpose(mdct_norm[channel, :, :]), axis=0), cmap='gray', vmin=-1., vmax=1., interpolation='none')

  # convert labels to Hz on y-axis
  bottom, top = ax.get_ylim()
  ytick_locations = [y for y in [100, 200, 500, 1000, 2000, 5000, 10000, 20000] if y <= top]
  max_frequency = sample_rate / 2  # Nyquist frequency: maximum frequency given a sample rate
  filter_band_width = max_frequency / filter_bands_n
  ax.set_yticks([filter_bands_n - (f/filter_band_width - .5) for f in ytick_locations])
  ax.set_yticklabels(["{0:3.0f}".format(f) for f in ytick_locations])
  return image


def save_spectrogram(mdct_amplitudes, filepath, channel=0):
  mdct_norm = psychoacoustic.ampl_to_norm(mdct_amplitudes)
  spectrum = np.flip(np.transpose(mdct_norm[channel, :, :]), axis=0)

  mdct_uint8 = 128. * (spectrum + 1.)
  mdct_uint8 = np.clip(mdct_uint8, a_min=0, a_max=255.)

  imageio.imwrite(filepath, mdct_uint8.astype(np.uint8))
  return


def read_spectrogram(filepath):
  mdct_uint8 = imageio.imread(filepath, format="PNG-PIL", pilmode='L')
  spectrum = mdct_uint8 / 128. - 1.
  mdct_norm = np.transpose(np.flip(spectrum, axis=0))
  mdct_norm = np.expand_dims(mdct_norm, axis=0)
  mdct_amplitudes = psychoacoustic.norm_to_ampl(mdct_norm)
  mdct_amplitudes = tf.cast(mdct_amplitudes, dtype=tf.float32)
  return mdct_amplitudes


def plot_spectrum(ax, wave_data, channels, blocks, sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6):
  """Plots the mdct spectrum with quiet and masking threshold for a given channel and block

  :param ax:              matplotlib axes
  :param wave_data:       raw audio signal
  :param channels:        channels to plot
  :param blocks:          block numbers to plot
  :param sample_rate:     sample rate
  :param filter_bands_n:  number of mdct filters
  :param bark_bands_n:    number of bark bands
  :param alpha            exponent for non-linear superposition of spreading functions
  :return:                image frames
  """
  mdct_init = mdct.setup(filter_bands_n)
  sample_rate, W, W_inv, quiet_threshold_in_bark, spreading_matrix, alpha = psychoacoustic.setup(sample_rate,
                                                                                                 filter_bands_n,
                                                                                                 bark_bands_n, alpha)

  # 1. MDCT analysis filter bank
  mdct_norm = mdct.transform(wave_data, mdct_init)
  mdct_dB = mdct.norm_to_dB(mdct_norm)
  mdct_amplitudes = 10.**5. * mdct.dB_to_ampl(mdct_dB)

  # 2. Masking threshold calculation
  sound_threshold = psychoacoustic._mappingfrombark(
    psychoacoustic._masking_threshold_in_bark(mdct_amplitudes, W, spreading_matrix, alpha, sample_rate), W_inv)
  quiet_threshold = psychoacoustic._mappingfrombark(quiet_threshold_in_bark, W_inv)

  # 3. Compute quantities to be plot
  max_frequency = sample_rate / 2
  filter_band_width = max_frequency / filter_bands_n
  filter_bands_mid_freq = filter_band_width * tf.range(filter_bands_n, dtype=tf.float32) + filter_band_width / 2

  x = tf.math.log(filter_bands_mid_freq) / tf.math.log(10.)
  y_amplitudes = 10. * tf.math.log(mdct_amplitudes ** 2.0) / tf.math.log(10.)
  y_masking_threshold = 10. * tf.math.log(sound_threshold ** 2.0) / tf.math.log(10.)
  y_quiet_threshold = 10. * tf.math.log(quiet_threshold ** 2.0) / tf.math.log(10.)

  # make images
  image_frames = []
  for channel in channels:
    for block in blocks:
      image_amplitudes, = ax.plot(x, y_amplitudes[channel, block, :].numpy(), color='blue')
      image_masking_threshold, = ax.plot(x, y_masking_threshold[channel, block, :].numpy(), color='green')
      image_quiet_threshold,  = ax.plot(x, y_quiet_threshold[0, 0, :].numpy(), color='black')

      # set xtick labels
      xtick_locations = ax.get_xticks()
      ax.set_xticklabels(["{0:3.0f}".format(tf.pow(10., f)) for f in xtick_locations])

      ax.set_xlabel("log10 of frequency")
      ax.set_ylabel("intensity [dB]")
      ax.set_ylim(ymin=-50)

      image_frames.append([image_amplitudes, image_masking_threshold, image_quiet_threshold])

  return image_frames
