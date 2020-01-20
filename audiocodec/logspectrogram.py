# !/usr/bin/env python

"""Converts a linear spectrogram to a logarithmic spectrogram
"""

import tensorflow as tf


class Spectrogram:
  def __init__(self, sample_rate, filter_bands_n):
    """Switch from a linear spectrogram to a 2D spectrogram made up of notes in octaves
    Usage example::

      spectrum = tf.random.uniform([filter_bands_n], minval=-1, maxval=1)
      log_spectrum = tf.einsum('i,ijk->jk', spectrum, log_spectrogram)
      spectrum_reconstructed = (tf.einsum('jk,ijk->i', log_spectrum, inv_log_spectrogram))

    here i:0 .. filter_bands_n, j: 0 .. octaves_n, k: 0 .. notes_n

    :param sample_rate:       sample_rate
    :param filter_bands_n:    number of filter bands of the filter bank
    """
    max_frequency = sample_rate / 2  # Nyquist frequency: maximum frequency given a sample rate

    filter_band_width = max_frequency / filter_bands_n

    filter_band_min_mid_freq = filter_band_width / 2.
    filter_band_max_mid_freq = filter_band_width * (filter_bands_n - 1) + filter_band_width / 2.

    # express all frequencies as f = 2^{i/notes_n} with i an integer (i = freq_linear_position)
    # notes_n is chosen such that we are not lossy:
    # the two highest frequencies in the filter_bands_n spectrum should be distinguishable in the 2^{i/notes_n} format
    notes_n = tf.math.ceil(-tf.math.log(2.) / (tf.math.log(1. - 1. / filter_bands_n)))

    tone_min_linear_position = tf.math.round(notes_n * tf.math.log(filter_band_min_mid_freq) / tf.math.log(2.))
    tone_max_linear_position = tf.math.round(notes_n * tf.math.log(filter_band_max_mid_freq) / tf.math.log(2.))

    octave_min = tf.math.floor(tone_min_linear_position / notes_n)
    octave_max = tf.math.floor(tone_max_linear_position / notes_n) + 1   # non-inclusive
    octaves_n = octave_max - octave_min

    # make full octave and note range
    octave = tf.math.floor(tf.range(octaves_n * notes_n) / notes_n)
    note = tf.range(octaves_n * notes_n) - notes_n * octave

    frequencies = tf.math.pow(2., octave + octave_min + (note / notes_n))
    filter_band_index = tf.math.round((frequencies - filter_band_width / 2.) / filter_band_width)
    indices = tf.dtypes.cast(tf.stack([filter_band_index, octave, note], axis=1), dtype=tf.int64)
    filter_band_index = tf.dtypes.cast(filter_band_index, dtype=tf.int32)
    # chop off frequencies which are not in the filter_bands index range
    indices = tf.boolean_mask(indices, tf.math.logical_and(tf.math.greater_equal(filter_band_index, 0),
                                                           tf.math.greater(filter_bands_n, filter_band_index)))

    log_spectrogram = tf.SparseTensor(indices, tf.ones(tf.shape(indices)[0]),
                                      dense_shape=[filter_bands_n, octaves_n, notes_n])

    _, idx, count = tf.unique_with_counts(indices[:, 0])
    weights = 1. / tf.gather(tf.dtypes.cast(count, tf.float32), idx)
    inv_log_spectrogram = tf.SparseTensor(indices, weights,
                                          dense_shape=[filter_bands_n, octaves_n, notes_n])

    # convert to dense
    self.log_spectrogram = tf.sparse.to_dense(log_spectrogram)
    self.inv_log_spectrogram = tf.sparse.to_dense(inv_log_spectrogram)

  @tf.function
  def freq_to_note(self, spectrum):
    """Convert from linear spectrogram to octave spectrogram

    :param spectrum:         mdct amplitudes     [#channels, #blocks, filter_bands_n,     ]
    :return:                 mdct amplitudes     [#channels, #blocks, (octaves_n, notes_n)]
    """
    return tf.einsum('bti,ijk->btjk', spectrum, self.log_spectrogram)

  @tf.function
  def note_to_freq(self, octave_spectrum):
    """Convert from octave spectrogram to linear spectrogram

    :param octave_spectrum:  mdct amplitudes     [#channels, #blocks, (octaves_n, notes_n)]
    :return:                 mdct amplitudes     [#channels, #blocks, filter_bands_n      ]
    """
    return tf.einsum('btjk,ijk->bti', octave_spectrum, self.inv_log_spectrogram)
