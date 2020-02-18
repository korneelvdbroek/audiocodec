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

    here i:0 .. filter_bands_n, j: 0 .. self.octaves_n, k: 0 .. self.notes_n

    :param sample_rate:       sample_rate
    :param filter_bands_n:    number of filter bands of the filter bank
    """
    # filter_bands_n <= 2^b-1   (b=octaves_n)
    self.octaves_n = tf.math.ceil(tf.math.log(filter_bands_n + 1.) / tf.math.log(2.))
    self.notes_n = 2**(self.octaves_n-1)

    # filter_bands_n = 6 <= 2**3 - 1
    # ijk = [freq_bin, octave, note]
    #    1--1--1--1
    #    2--2  3--3
    #    4  5  6  X
    # j: range(b); k: range(2**(b-1))
    octave = tf.math.floor(tf.range(self.octaves_n * self.notes_n) / self.notes_n)
    note = tf.range(self.octaves_n * self.notes_n) - octave * self.notes_n
    filter_band_index = 2**octave + tf.math.floor((2**(octave+1) - 2**octave) * note / self.notes_n) - 1

    # remove entries of transformation matrix which are out-of-range for the filter_band
    indices = tf.dtypes.cast(tf.stack([filter_band_index, octave, note], axis=1), dtype=tf.int64)
    filter_band_index = tf.dtypes.cast(filter_band_index, dtype=tf.int32)
    indices = tf.boolean_mask(indices, tf.math.less(filter_band_index, filter_bands_n))

    # define transformation matrix
    log_spectrogram = tf.SparseTensor(indices, tf.ones(tf.shape(indices)[0]),
                                      dense_shape=[filter_bands_n, self.octaves_n, self.notes_n])

    # inverse transformation matrix
    _, idx, count = tf.unique_with_counts(indices[:, 0])
    weights = 1. / tf.gather(tf.dtypes.cast(count, tf.float32), idx)
    inv_log_spectrogram = tf.SparseTensor(indices, weights,
                                          dense_shape=[filter_bands_n, self.octaves_n, self.notes_n])

    # convert to dense
    self.log_spectrogram = tf.sparse.to_dense(log_spectrogram)
    self.inv_log_spectrogram = tf.transpose(tf.sparse.to_dense(inv_log_spectrogram), perm=[1, 2, 0])

  @tf.function
  def freq_to_note(self, spectrum):
    """Convert from linear spectrogram to octave spectrogram

    :param spectrum:         mdct amplitudes     [..., filter_bands_n    ]
    :return:                 mdct amplitudes     [..., self.octaves_n, self.notes_n]
    """
    return tf.tensordot(spectrum, self.log_spectrogram, axes=1)

  @tf.function
  def note_to_freq(self, octave_spectrum):
    """Convert from octave spectrogram to linear spectrogram

    :param octave_spectrum:  mdct amplitudes     [..., self.octaves_n, self.notes_n]
    :return:                 mdct amplitudes     [..., filter_bands_n    ]
    """
    return tf.tensordot(octave_spectrum, self.inv_log_spectrogram, axes=2)
