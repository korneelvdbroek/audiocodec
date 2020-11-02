# !/usr/bin/env python

""" Psycho-acoustic model for lossy audio encoder (inspired by aac encoder)

Loosely based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
See also chapter 9 in "Digital Audio Signal Processing" by Udo Zolzer
"""

import tensorflow as tf
import math

_dB_MAX = 100.           # corresponds with abs(amplitude) = 1.
_AMPLITUDE_EPS = 1e-6    # corresponds with a dB_min of -20dB



def amplitude_to_dB(amplitude):
  """Utility function to convert the amplitude which is normalized in the [-1..1] range to dB.
  The dB scale is set such that an amplitude of 1 (or -1) corresponds to the maximum dB level (_dB_MAX)

  :param amplitude:  amplitude normalized in [-1..1] range
  :return:           corresponding dB scale
  """
  # return 10. * tf.math.log(amplitude ** 2.0) / tf.math.log(10.) + _dB_MAX
  return 10. * tf.math.log(tf.maximum(_AMPLITUDE_EPS ** 2.0, amplitude ** 2.0)) / tf.math.log(10.) + _dB_MAX


_dB_MIN = amplitude_to_dB(_AMPLITUDE_EPS)


class PsychoacousticModel:
  def __init__(self, sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6):
    """Computes required initialization matrices

    :param sample_rate:       sample_rate
    :param alpha:             exponent for non-linear superposition (1.0 is linear superposition, 0.6 is default)
    :param filter_bands_n:    number of filter bands of the filter bank
    :param bark_bands_n:      number of bark bands
    :return:                  tuple with pre-computed required for encoder and decoder
    """
    self.alpha = alpha
    self.sample_rate = sample_rate
    self.bark_bands_n = bark_bands_n
    self.filter_bands_n = filter_bands_n

    # pre-compute some matrices
    self.W, self.W_inv = self._bark_freq_mapping(dtype=tf.float32)
    self.quiet_threshold_amplitude = self._quiet_threshold_amplitude_in_bark(dtype=tf.float32)
    self.spreading_matrix = self._spreading_matrix_in_bark()

  @tf.function
  def tonality(self, mdct_amplitudes):
    """
    Compute tonality (0:white noise ... 1:tonal) from the spectral flatness measure (SFM)
    See equations (9.10)-(9.11) in Digital Audio Signal Processing by Udo Zolzer

    :param mdct_amplitudes:   mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
    :return:                  tonality vector [batches_n, blocks_n, 1, channels_n]
    """
    mdct_intensity = tf.pow(mdct_amplitudes, 2)

    sfm = 10 * tf.math.log(tf.divide(
      tf.exp(tf.reduce_mean(tf.math.log(tf.maximum(_AMPLITUDE_EPS ** 2, mdct_intensity)), axis=2, keepdims=True)),
      tf.reduce_mean(mdct_intensity, axis=2, keepdims=True) + _AMPLITUDE_EPS ** 2)) / math.log(10.0)

    return tf.minimum(sfm / -60., 1.0)

  @tf.function
  def global_masking_threshold(self, mdct_amplitudes, tonality_per_block, drown=0.0):
    """Determines which amplitudes we cannot hear, either since they are too soft
    to hear or because other louder amplitudes are masking it.
    Method uses non-linear superposition determined by factor self.alpha

    :param mdct_amplitudes:     mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
    :param tonality_per_block:  tonality vector associated with the mdct_amplitudes [batches_n, blocks_n, 1, channels_n]
                                can be computed using the method tonality(mdct_amplitudes)
    :param drown:               factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                    masking threshold [batches_n, blocks_n, filter_bands_n, channels_n]
    """
    with tf.name_scope('global_masking_threshold'):
      masking_threshold = self._masking_threshold_in_bark(mdct_amplitudes, tonality_per_block, drown)

      # Take max between quiet threshold and masking threshold
      # Note: even though both thresholds are expressed as amplitudes,
      # they are all positive due to the way they were constructed
      global_masking_threshold_in_bark = tf.maximum(masking_threshold, self.quiet_threshold_amplitude)

      global_mask_threshold = self._mappingfrombark(global_masking_threshold_in_bark)

    return global_mask_threshold

  def _masking_threshold_in_bark(self, mdct_amplitudes, tonality_per_block, drown=0.0):
    """Returns amplitudes that are masked by the sound defined by mdct_amplitudes

    :param mdct_amplitudes:     mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
    :param tonality_per_block:  tonality vector associated with the mdct_amplitudes [batches_n, blocks_n, 1, channels_n]
    :param drown:               factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                    vector of amplitudes in dB for softest audible sounds given a certain sound [#channels, #blocks, bark_bands_n]
    """
    max_frequency = self.sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = self.freq2bark(max_frequency)

    # compute masking offset:
    #    O(i) = tonality (14.5 + i) + (1 - tonality) 5.5
    # with i the bark index
    # see p10 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)
    # in einsum, we tf.squeeze() axis=2 (index i) and take outer product with tf.linspace()
    offset = (1. - drown) * (tf.einsum('nbic,j->nbjc', tonality_per_block, tf.linspace(0.0, max_bark, self.bark_bands_n))
                             + 9. * tonality_per_block
                             + 5.5)

    # add offset to spreading matrix (see (9.18) in "Digital Audio Signal Processing" by Udo Zolzer)
    # note: einsum('.j.,.j.->.j.') multiplies elements on diagonal element-wise (without summing over j)
    masking_matrix = tf.einsum('ij,nbjc->nbijc', self.spreading_matrix, tf.pow(10.0, -self.alpha * offset / 10.0))

    # Transposed version of (9.17) in Digital Audio Signal Processing by Udo Zolzer
    # \Sum_i (amplitude_i^2)^{\alpha} x [ mask^{\alpha}_{i-n} ]_n
    #   = \Sum amplitude_i x mask_{in}                       --> each row is a mask
    # Non-linear superposition (see p13 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)
    # \alpha ~ 0.3 is valid for 94 bark_bands_n; with less bark_bands_n 0.3 leads to (way) too much masking
    amplitudes_in_bark = self._mapping2bark(mdct_amplitudes)
    intensity_in_bark = tf.pow(tf.maximum(_AMPLITUDE_EPS, amplitudes_in_bark), 2. * self.alpha)
    masking_intensity_in_bark = tf.einsum('nbic,nbijc->nbjc', intensity_in_bark, masking_matrix)
    masking_amplitude_in_bark = tf.pow(tf.maximum(_AMPLITUDE_EPS ** 2, masking_intensity_in_bark), 1. / (2. * self.alpha))

    return masking_amplitude_in_bark

  def _spreading_matrix_in_bark(self):
    """Returns (power) spreading matrix, to apply in bark scale to determine masking threshold from sound

    :return:                spreading matrix [bark_bands_n, bark_bands_n]
    """
    max_frequency = self.sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = self.freq2bark(max_frequency)

    # Prototype spreading function [Bark/dB]
    # see equation (9.15) in "Digital Audio Signal Processing" by Udo Zolzer
    f_spreading = tf.map_fn(lambda z: 15.81 + 7.5 * (z + 0.474) - 17.5 * tf.sqrt(1 + tf.pow(z + 0.474, 2)),
                            tf.linspace(-max_bark, max_bark, 2 * self.bark_bands_n))

    # Convert from dB to intensity and include alpha exponent
    f_spreading_intensity = tf.pow(10.0, self.alpha * f_spreading / 10.0)

    # Turns the spreading prototype function into a (bark_bands_n x bark_bands_n) matrix of shifted versions.
    # Transposed version of (9.17) in Digital Audio Signal Processing by Udo Zolzer
    spreading_matrix = tf.stack([f_spreading_intensity[(self.bark_bands_n - row):(2 * self.bark_bands_n - row)]
                                 for row in range(self.bark_bands_n)], axis=0)

    return spreading_matrix

  def _quiet_threshold_amplitude_in_bark(self, dtype):
    """Compute the amplitudes of the softest sounds one can hear
       See (9.3) in "Digital Audio Signal Processing" by Udo Zolzer

    :return:       amplitude vector for softest audible sounds [1, 1, bark_bands_n, 1]
                   returned amplitudes are all positive
    """
    # Threshold in quiet:
    max_frequency = self.sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = self.freq2bark(max_frequency)
    bark_band_width = max_bark / self.bark_bands_n

    bark_bands_mid_bark = bark_band_width * tf.range(self.bark_bands_n, dtype=dtype) + bark_band_width / 2.
    bark_bands_mid_kHz = self.bark2freq(bark_bands_mid_bark) / 1000.

    # Threshold of quiet expressed as amplitude (dB) for each Bark bands
    # (see also p4 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf -- slide has 6.5 typo)
    # see (9.3) in "Digital Audio Signal Processing" by Udo Zolzer
    quiet_threshold_dB = tf.clip_by_value(
      (3.64 * (tf.pow(bark_bands_mid_kHz, -0.8))
       - 6.5 * tf.exp(-0.6 * tf.pow(bark_bands_mid_kHz - 3.3, 2.))
       + 1e-3 * (tf.pow(bark_bands_mid_kHz, 4.))),
      -20, 160)

    # convert dB to amplitude, where _dB_MAX corresponds with an amplitude of 1.0
    quiet_threshold_amplitude = tf.reshape(tf.pow(10.0, (quiet_threshold_dB - _dB_MAX) / 20), shape=[1, 1, -1, 1])

    return quiet_threshold_amplitude

  def _bark_freq_mapping(self, dtype):
    """Compute (static) mapping between MDCT filter bank ranges and Bark bands they fall in

                              ----> bark_bands_n
                   |        [ 1  0  ...  0 ]
      W =          |        [ 0  1  ...  0 ]
                   V        [ 0  1  ...  0 ]
              filter_bank_n [       ...    ]
                            [ 0  0  ...  1 ]

    Inverse transformation, from Bark bins amplitude(!) to MDCT filter bands amplitude(!), consists of transposed
    with normalization factor such that power (square of amplitude) in each Bark band
    gets split equally between the filter bands making up that Bark band:

                              ----> filter_bank_n
                   |        [ 1/\sqrt{1} 0           0          ...  0          ]
      W_inv =      |        [ 0          1/\sqrt{2}  1/\sqrt{2} ...  0          ]
                   V        [                                   ...             ]
              bark_bands_n  [ 0          0           0          ...  1/\sqrt{1} ]

      :return: 2 matrices with shape
                  W      [filter_bank_n , bark_band_n]
                  W_inv  [bark_band_n   , filter_bank_n]
    """
    max_frequency = self.sample_rate / 2  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = self.freq2bark(max_frequency)
    bark_band_width = max_bark / self.bark_bands_n

    filter_band_width = max_frequency / tf.cast(self.filter_bands_n, dtype)
    filter_bands_mid_freq = filter_band_width * tf.range(self.filter_bands_n, dtype=dtype) + filter_band_width / 2
    filter_bands_mid_bark = self.freq2bark(filter_bands_mid_freq)

    column_index = tf.tile(tf.expand_dims(tf.range(self.bark_bands_n, dtype=dtype), axis=0),
                           multiples=[self.filter_bands_n, 1])
    W = tf.dtypes.cast(tf.equal(tf.tile(tf.expand_dims(tf.math.floor(filter_bands_mid_bark / bark_band_width), axis=1),
                                        multiples=[1, self.bark_bands_n]), column_index), dtype=dtype)

    # (bark_band_n x bark_band_n) . (bark_band_n x filter_bank_n)
    W_transpose = tf.transpose(W, perm=[1, 0])
    W_inv = tf.tensordot(tf.linalg.diag(tf.pow(1.0 / tf.maximum(_AMPLITUDE_EPS ** 2, tf.reduce_sum(W_transpose, axis=1)), 0.5)),
                         W_transpose, axes=[[1], [0]])

    return W, W_inv

  def _mapping2bark(self, mdct_amplitudes):
    """Takes MDCT amplitudes and maps it into Bark bands amplitudes.
    Power spectral density of Bark band is sum of power spectral density in
    corresponding filter bands (power spectral density of signal S = X_1^2 + ... + X_n^2)
      (see also slide p9 of ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)

    :param mdct_amplitudes:  vector of mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
    :param W:                matrix to convert from filter bins to bark bins [filter_bands_n, bark_bands_n]
    :return:                 vector of (positive!) signal amplitudes of the Bark bands [batches_n, blocks_n, bark_bands_n, channels_n]
    """
    # tf.maximum() is necessary, to make sure rounding errors don't make the gradient nan!
    mdct_intensity = tf.pow(mdct_amplitudes, 2)
    mdct_intensity_in_bark = tf.einsum('nbic,ij->nbjc', mdct_intensity, self.W)
    mdct_amplitudes_in_bark = tf.pow(tf.maximum(_AMPLITUDE_EPS ** 2, mdct_intensity_in_bark), 0.5)

    return mdct_amplitudes_in_bark

  def _mappingfrombark(self, amplitudes_bark):
    """Takes Bark band amplitudes and maps it to MDCT amplitudes.
    Power spectral density of Bark band is split equally between the
    filter bands making up the Bark band (many-to-one).

    :param amplitudes_bark:  vector of signal amplitudes of the Bark bands [batches_n, blocks_n, bark_bands_n, channels_n]
    :param W_inv:            matrix to convert from filter bins to bark bins [bark_bands_n, filter_bands_n]
    :return:                 vector of mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
    """
    return tf.einsum('nbic,ij->nbjc', amplitudes_bark, self.W_inv)

  def freq2bark(self, frequencies):
    """Empirical Bark scale"""
    return 6. * tf.asinh(frequencies / 600.)

  def bark2freq(self, bark_band):
    """Empirical Bark scale"""
    return 600. * tf.sinh(bark_band / 6.)
