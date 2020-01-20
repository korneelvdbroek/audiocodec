# !/usr/bin/env python

""" Utility functions

"""

import imageio

import tensorflow as tf
import numpy as np

from audiocodec import psychoacoustic as pa


def plot_spectrogram(ax, mdct_norm, sample_rate, filter_bands_n, channel=0):
  image = ax.imshow(np.flip(np.transpose(mdct_norm[channel, :, :]), axis=0),
                    cmap='gray', vmin=-1., vmax=1., interpolation='none')

  # convert labels to Hz on y-axis
  bottom, top = ax.get_ylim()
  ytick_locations = [y for y in [100, 200, 500, 1000, 2000, 5000, 10000, 20000] if y <= top]
  max_frequency = sample_rate / 2  # Nyquist frequency: maximum frequency given a sample rate
  filter_band_width = max_frequency / filter_bands_n
  ax.set_yticks([filter_bands_n - (f/filter_band_width - .5) for f in ytick_locations])
  ax.set_yticklabels(["{0:3.0f}".format(f) for f in ytick_locations])

  # x axis is time
  blocks = tf.shape(mdct_norm)[1].numpy()
  blocks_per_sec = sample_rate / filter_bands_n
  duration = blocks / blocks_per_sec
  ax.set_xticks([t * blocks_per_sec for t in range(int(duration))])
  ax.set_xticklabels(["{0:3.0f}".format(t) for t in range(int(duration))])

  return image


def plot_logspectrogram(ax, mdct_norm, sample_rate, filter_bands_n, channel=0):
  # [channel, block, octave, note]
  #
  tiled_mdct = [entry.numpy() for block in range(tf.shape(mdct_norm)[1]) for entry in (mdct_norm[channel, block, :, :],
                                                                                       -tf.ones([1, tf.shape(mdct_norm)[3]]))]
  tiled_mdct = np.vstack(tiled_mdct)
  image = ax.imshow(np.flip(np.transpose(tiled_mdct), axis=0),
                    cmap='gray', vmin=-1., vmax=1., interpolation='none')

  # x axis is time
  blocks = tf.shape(mdct_norm)[1].numpy()
  blocks_per_sec = sample_rate / filter_bands_n
  duration = blocks / blocks_per_sec
  octaves = tf.shape(mdct_norm)[2].numpy()
  ax.set_xticks([t * (octaves+1) * blocks_per_sec for t in range(int(duration))])
  ax.set_xticklabels(["{0:3.0f}".format(t) for t in range(int(duration))])
  return image


def plot_logspectrum(ax, mdct_norm, sample_rate, filter_bands_n, channel=0):
  image_frames = []
  blocks_per_sec = sample_rate / filter_bands_n
  for block in range(tf.shape(mdct_norm)[1]):
    image = ax.imshow(np.flip(np.transpose(mdct_norm[channel, block, :, :]), axis=0),
                      cmap='gray', vmin=-1., vmax=1., interpolation='none')
    txt = ax.text(0.05, 0.90, "{0:3.1f}".format(block / blocks_per_sec))
    image_frames.append([image, txt])
  return image_frames


def save_spectrogram(mdct_amplitudes, filepath, channel=0):
  mdct_norm = pa.ampl_to_norm(mdct_amplitudes)
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
  mdct_amplitudes = pa.norm_to_ampl(mdct_norm)
  mdct_amplitudes = tf.cast(mdct_amplitudes, dtype=tf.float32)
  return mdct_amplitudes


def plot_spectrum(mdct, psychoacoustic, ax, wave_data, channels, blocks, sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6):
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

  # 1. MDCT analysis filter bank
  mdct_amplitudes = mdct.transform(wave_data)

  # 2. Masking threshold calculation
  sound_threshold = psychoacoustic._mappingfrombark(
    psychoacoustic._masking_threshold_in_bark(mdct_amplitudes))
  quiet_threshold = psychoacoustic._mappingfrombark(psychoacoustic._quiet_threshold_amplitude_in_bark())

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
