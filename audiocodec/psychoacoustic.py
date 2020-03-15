# !/usr/bin/env python

""" Psycho-acoustic model for lossy audio encoder (inspired by aac encoder)

Loosely based based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
Note: non-private functions are decorated with tf.function.
  When they are invoked as part of a bigger graphs they will be retraced, leading to a retracing warning.
"""

import tensorflow as tf
import math

_dB_MIN = 0.
_dB_MAX = 100.
_EPSILON = 1e-6


class PsychoacousticModel:
  def __init__(self, sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6):
    """Computes required initialization matrices (stateless, no OOP)

    :param sample_rate:       sample_rate
    :param alpha:             exponent for non-linear superposition (~0.6)
    :param filter_bands_n:    number of filter bands of the filter bank
    :param bark_bands_n:      number of bark bands
    :return:                  tuple with pre-computed required for encoder and decoder
    """
    self.alpha = alpha
    self.sample_rate = sample_rate
    self.bark_bands_n = bark_bands_n
    self.filter_bands_n = filter_bands_n

    self.W, self.W_inv = self._bark_freq_mapping()
    self.quiet_threshold_amplitude = self._quiet_threshold_amplitude_in_bark()
    self.spreading_matrix = self._spreading_matrix_in_bark(alpha)

  @tf.function
  def scale_factors(self, mask_thresholds_log_bark):
    """Compute scale-factors from logarithmic masking threshold.

    :param mask_thresholds_log_bark: logarithmic masking threshold [#channels, #blocks, bark_bands_n]
    :return:                         scale factors to be applied on amplitudes [#channels, #blocks, filter_bands_n]
    """
    with tf.name_scope('scale_factors'):
      mask_thresholds_trunc_bark = tf.pow(2., mask_thresholds_log_bark / 4.)

      mask_thresholds_trunc = self._mappingfrombark(mask_thresholds_trunc_bark)

      # maximum of the magnitude of the quantization error is delta/2
      mdct_scale_factors = 1. / (2. * mask_thresholds_trunc)

    return mdct_scale_factors

  @tf.function
  def global_masking_threshold(self, mdct_amplitudes, drown=0.0):
    """Determines which amplitudes we cannot hear, either since they are too soft
    to hear or because other louder amplitudes are masking it.
    Method uses non-linear superposition determined by factor alpha

    :param mdct_amplitudes:   vector of mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    :param drown:             factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                  masking threshold [#channels, #blocks, filter_bands_n]
    """
    global_mask_threshold = self._mappingfrombark(
      self._global_masking_threshold_in_bark(mdct_amplitudes, drown))
    return global_mask_threshold

  @tf.function
  def zero_filter(self, mdct_norm, drown=0.0):
    """Zero out out frequencies which are inaudible

    :param mdct_norm:      vector of normalized mdct amplitudes for each filter [#channels, #blocks, filter_bands_n]
    :param drown:          factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:               modified amplitudes [#channels, #blocks, filter_bands_n]
    """
    mdct_amplitudes = norm_to_ampl(mdct_norm)
    total_threshold = self.global_masking_threshold(mdct_amplitudes, drown)
    threshold_norm = ampl_to_norm(total_threshold)

    # Update spectrum
    # 1. remove anything below masking threshold
    mdct_modified_norm = tf.where(threshold_norm ** 2.0 < mdct_norm ** 2.0,
                                  mdct_norm,
                                  tf.zeros(tf.shape(mdct_norm)))
    # 2. pass-through
    # mdct_modified = mdct_amplitudes
    # 3. put in masking threshold
    # mdct_modified = np.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, mdct_amplitudes, total_threshold)
    # 4. keep only masking threshold
    # mdct_modified = total_threshold
    # 5. keep only sound below masking threshold
    # mdct_modified = np.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, 1e-6, mdct_amplitudes)

    return mdct_modified_norm

  @tf.function
  def lrelu_filter(self, mdct_norm, drown=0.0, beta=0.2):
    """Leaky ReLU suppression of in-audible frequencies

    :param mdct_norm:           mdct amplitudes in rescaled dB         [#channels, #blocks, filter_bands_n]
    :param drown:               factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :param beta:                percent of values below the threshold line which are let through but attenuated
                                lower values will lead to larger gradients...
    :return:                    filtered normalized mdct amplitudes    [#channels, #blocks, filter_bands_n]
    """
    # get masking threshold in norm space
    tf.debugging.assert_less_equal(tf.reduce_max(tf.abs(mdct_norm)), 1.,
                                   "psychoacoustic.lrelu_filter inputs should be in the -1..1 range")

    mdct_amplitudes = norm_to_ampl(mdct_norm)
    total_threshold = self.global_masking_threshold(mdct_amplitudes, drown)
    total_masking_norm = ampl_to_norm(total_threshold)

    # LeakyReLU-based filter: suppress in-audible frequencies (in norm space)
    noise = tf.random.uniform(tf.shape(mdct_norm), minval=0., maxval=1.)
    mdct_abs = tf.abs(mdct_norm)
    mdct_norm_filtered = tf.where(mdct_abs <= 1./(1.+beta) * total_masking_norm,
                                  beta * mdct_norm * noise,
                                  tf.where(total_masking_norm <= mdct_abs,
                                           mdct_norm,
                                           (1./beta * mdct_norm + tf.sign(mdct_norm) * (1. - 1./beta) * total_masking_norm)
                                           * (1. - noise * (total_masking_norm - mdct_abs) / total_masking_norm * ((1. + beta) / beta)) ))

    # This leads to horrible gradients...
    # mdct_abs = tf.abs(mdct_norm)
    # mdct_norm_filtered = tf.where(total_masking_norm <= mdct_abs,
    #                               mdct_norm,
    #                               0.0)

    # LeakyReLU-based filter: suppress in-audible frequencies (in norm space)
    mdct_abs = tf.abs(mdct_norm)
    mdct_norm_filtered = tf.where(mdct_abs <= (1.-beta) * total_masking_norm,
                                  0.,
                                  tf.where(total_masking_norm <= mdct_abs,
                                           mdct_norm,
                                           (1./beta * mdct_norm + tf.sign(mdct_norm) * (1. - 1./beta) * total_masking_norm) ))

    return mdct_norm_filtered

  @tf.function
  def noise_filter(self, mdct_norm):
    """

    :param mdct_norm:
    :return:
    """
    # noise-based overlay: add noise to in-audible frequencies,
    # so learning does not learn anything about these (in norm space)
    # todo: continue here (decide what sigma to use for noise)
    # mdct_norm_noised = tf.where(total_masking_norm <= mdct_abs_fadeout,
    #                           mdct_norm,
    #                           noise)
    pass

  def _global_masking_threshold_in_bark(self, mdct_amplitudes, drown=0.):
    """Determines which amplitudes we cannot hear, either since they are too soft
    to hear or because other louder amplitudes are masking it.
    Method uses non-linear superposition determined by factor alpha

    :param mdct_amplitudes:   vector of mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    :param drown:             factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                  masking threshold [#channels, #blocks, bark_bands_n]
    """
    with tf.name_scope('global_masking_threshold'):
      masking_threshold = self._masking_threshold_in_bark(mdct_amplitudes, drown)

      # Take max between quiet threshold and masking threshold
      # Note: even though both thresholds are expressed as amplitudes,
      # they are all positive due to the way they were constructed
      global_mask_threshold_in_bark = tf.maximum(masking_threshold, self.quiet_threshold_amplitude)

    return global_mask_threshold_in_bark

  def _masking_threshold_in_bark(self, mdct_amplitudes, drown=0.):
    """Returns amplitudes that are masked by the sound defined by mdct_amplitudes

    :param mdct_amplitudes:   mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    :param drown:             factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                  vector of amplitudes in dB for softest audible sounds given a certain sound [#channels, #blocks, bark_bands_n]
    """
    max_frequency = self.sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = self.freq2bark(max_frequency)

    # compute tonality (0:white noise ... 1:tonal) from the spectral flatness measure
    # (SFM = 0dB for white noise, SFM << 0dB for pure tone)
    # tonality = 10./-60. * \log_10  (e^{1/N \sum_{filter_band_i} \ln(a_i^2)}) /
    #                                (1/N \sum_{filter_band_i} a_i^2)
    mdct_intensity = tf.pow(mdct_amplitudes, 2)
    tonality = tf.minimum(1.0, 10 * tf.math.log(tf.divide(
      tf.exp(tf.reduce_mean(tf.math.log(tf.maximum(_EPSILON**2, mdct_intensity)), axis=2, keepdims=True)),
      tf.reduce_mean(mdct_intensity, axis=2, keepdims=True) + _EPSILON**2)) / (-60.0 * math.log(10.0)))
    # add bark_bands_n dimension: [#channels, #blocks, bark_bands_n]
    tonality = tf.tile(tonality, multiples=[1, 1, self.bark_bands_n])

    # compute masking offset: O(i) = tonality (14.5 + i) + (1 - tonality) 5.5
    # note: einsum('.i.,.i.->.i.') does an element-wise multiplication (and no sum) along a specified axes
    offset = (1. - drown) * (tf.einsum('cbj,j->cbj', tonality, tf.linspace(0.0, max_bark, self.bark_bands_n)) + 9. * tonality + 5.5)

    # add offset to spreading matrix
    masking_matrix = tf.einsum('ij,cbj->cbij', self.spreading_matrix, tf.pow(10.0, -self.alpha * offset / 10.0))

    # Transposed version of (9.17) in Digital Audio Signal Processing by Udo Zolzer
    # \Sum_i (amplitude_i^2)^{\alpha} x [ mask^{\alpha}_{i-n} ]_n
    #   = \Sum amplitude_i x mask_{in}                       --> each row is a mask
    # Non-linear superposition (see p13 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)
    # \alpha ~ 0.3 is valid for 94 bark_bands_n; with less bark_bands_n 0.3 leads to (way) too much masking
    amplitudes_in_bark = self._mapping2bark(mdct_amplitudes)
    intensity_in_bark = tf.pow(tf.maximum(_EPSILON, amplitudes_in_bark), 2. * self.alpha)
    masking_intensity_in_bark = tf.einsum('cbi,cbij->cbj', intensity_in_bark, masking_matrix)
    masking_amplitude_in_bark = tf.pow(tf.maximum(_EPSILON**2, masking_intensity_in_bark), 1. / (2. * self.alpha))

    return masking_amplitude_in_bark

  def _spreading_matrix_in_bark(self, alpha):
    """Returns (power) spreading matrix, to apply in bark scale to determine masking threshold from sound

    :param alpha:           exponent for non-linear superposition as applied on amplitudes
    :return:                spreading matrix [bark_bands_n, bark_bands_n]
    """
    max_frequency = self.sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = self.freq2bark(max_frequency)

    # Prototype spreading function [Bark/dB]
    f_spreading = tf.map_fn(lambda z: 15.81 + 7.5 * (z + 0.474) - 17.5 * tf.sqrt(1 + tf.pow(z + 0.474, 2)),
                            tf.linspace(-max_bark, max_bark, 2 * self.bark_bands_n))

    # Convert from dB to intensity and include alpha exponent
    f_spreading_intensity = tf.pow(10.0, alpha * f_spreading / 10.0)

    # Turns the spreading prototype function into a (bark_bands_n x bark_bands_n) matrix of shifted versions.
    # Transposed version of (9.17) in Digital Audio Signal Processing by Udo Zolzer
    spreading_matrix = tf.stack([f_spreading_intensity[(self.bark_bands_n - row):(2 * self.bark_bands_n - row)]
                                 for row in range(self.bark_bands_n)], axis=0)

    return spreading_matrix

  def _quiet_threshold_amplitude_in_bark(self):
    """Compute the amplitudes of the softest sounds one can hear

    :return:              amplitude vector for softest audible sounds [1, 1, bark_bands_n]
    """
    # Threshold in quiet:
    max_frequency = self.sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = self.freq2bark(max_frequency)
    bark_band_width = max_bark / self.bark_bands_n

    bark_bands_mid_bark = bark_band_width * tf.range(self.bark_bands_n, dtype=tf.float32) + bark_band_width / 2.
    bark_bands_mid_kHz = self.bark2freq(bark_bands_mid_bark) / 1000.

    # Threshold of quiet expressed as amplitude (dB) for each Bark bands
    # (see also p4 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf -- slide has 6.5 typo)
    quiet_threshold_dB = tf.clip_by_value(
      (3.64 * (tf.pow(bark_bands_mid_kHz, -0.8))
       - 6.5 * tf.exp(-0.6 * tf.pow(bark_bands_mid_kHz - 3.3, 2.))
       + 1e-3 * (tf.pow(bark_bands_mid_kHz, 4.))),
      -20, 160)

    # convert dB to amplitude
    quiet_threshold_amplitude = tf.expand_dims(tf.expand_dims(tf.pow(10.0, quiet_threshold_dB / 20), axis=0), axis=0)

    return quiet_threshold_amplitude

  def _bark_freq_mapping(self):
    """Compute (static) mapping between MDCT filter bank ranges and Bark bands they fall in

                              ----> bark_bands_n
                   |        [ 1  0  ...  0 ]
      W =          |        [ 0  1  ...  0 ]
                   V        [ 0  1  ...  0 ]
              filter_bank_n [       ...    ]
                            [ 0  0  ...  1 ]

    Inverse transformation, from Bark bins to MDCT filter bands, consists of transposed
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

    filter_band_width = max_frequency / tf.cast(self.filter_bands_n, tf.float32)
    filter_bands_mid_freq = filter_band_width * tf.range(self.filter_bands_n, dtype=tf.float32) + filter_band_width / 2
    filter_bands_mid_bark = self.freq2bark(filter_bands_mid_freq)

    column_index = tf.tile(tf.expand_dims(tf.range(self.bark_bands_n, dtype=tf.float32), axis=0),
                           multiples=[self.filter_bands_n, 1])
    W = tf.dtypes.cast(tf.equal(tf.tile(tf.expand_dims(tf.math.floor(filter_bands_mid_bark / bark_band_width), axis=1),
                                        multiples=[1, self.bark_bands_n]), column_index), dtype=tf.float32)

    # (bark_band_n x bark_band_n) . (bark_band_n x filter_bank_n)
    W_transpose = tf.transpose(W, perm=[1, 0])
    W_inv = tf.tensordot(tf.linalg.diag(tf.pow(1.0 / tf.maximum(_EPSILON**2, tf.reduce_sum(W_transpose, axis=1)), 0.5)),
                         W_transpose, axes=[[1], [0]])

    return W, W_inv

  def _mapping2bark(self, mdct_amplitudes):
    """Takes MDCT amplitudes and maps it into Bark bands amplitudes.
    Power spectral density of Bark band is sum of power spectral density in
    corresponding filter bands (power spectral density of signal S = X_1^2 + ... + X_n^2)
      (see also slide p9 of ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)

    :param mdct_amplitudes:  vector of mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    :param W:                matrix to convert from filter bins to bark bins [filter_bands_n, bark_bands_n]
    :return:                 vector of (positive!) signal amplitudes of the Bark bands [#channels, #blocks, bark_bands_n]
    """
    # tf.maximum() is necessary, to make sure rounding errors don't make the gradient nan!
    mdct_intensity = tf.pow(mdct_amplitudes, 2)
    mdct_intensity_in_bark = tf.tensordot(mdct_intensity, self.W, axes=[[2], [0]])
    mdct_amplitudes_in_bark = tf.pow(tf.maximum(_EPSILON**2, mdct_intensity_in_bark), 0.5)

    return mdct_amplitudes_in_bark

  def _mappingfrombark(self, amplitudes_bark):
    """Takes Bark band amplitudes and maps it to MDCT amplitudes.
    Power spectral density of Bark band is split equally between the
    filter bands making up the Bark band (many-to-one).

    :param amplitudes_bark:  vector of signal amplitudes of the Bark bands [#channels, #blocks, bark_bands_n]
    :param W_inv:            matrix to convert from filter bins to bark bins [bark_bands_n, filter_bands_n]
    :return:                 vector of mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    """
    return tf.tensordot(amplitudes_bark, self.W_inv, axes=[[2], [0]])

  def freq2bark(self, frequencies):
    """Empirical Bark scale"""
    return 6. * tf.asinh(frequencies / 600.)

  def bark2freq(self, bark_band):
    """Empirical Bark scale"""
    return 600. * tf.sinh(bark_band / 6.)


def ampl_to_norm(mdct_amplitudes):
  """Point-wise converts mdct amplitudes to normalized dB scale
      dB = 20 log_10{ampl}

  :param mdct_amplitudes:  -inf..inf          [channels_n, #blocks, filter_bands_n]
  :return:                 -1..1              [channels_n, #blocks, filter_bands_n]
  """
  # this formula clips the very small amplitudes (so log() does not become negative)
  mdct_dB = tf.sign(mdct_amplitudes) * 20. * \
            tf.maximum(
              tf.math.log(tf.maximum(_EPSILON, tf.abs(mdct_amplitudes))),
              0.0) / tf.math.log(10.)

  mdct_norm = dB_to_norm(mdct_dB)

  return mdct_norm


def norm_to_ampl(mdct_norm):
  """Point-wise converts mdct amplitudes in normalized dB scale to actual amplitude values
      ampl = 10^{dB/20}

  :param mdct_norm:   -1..1              [channels_n, #blocks, filter_bands_n]
  :return:            -inf..inf          [channels_n, #blocks, filter_bands_n]
  """
  mdct_dB = norm_to_dB(mdct_norm)
  mdct_amplitudes = tf.sign(mdct_dB) * tf.pow(10., tf.abs(mdct_dB) / 20.)

  return mdct_amplitudes


def dB_to_norm(mdct_dB):
  """Point-wise linearly normalize mdct amplitude expressed in dB (range -inf..inf) to -1..1 range

  :param mdct_dB: -inf..inf [dB]     [channels_n, #blocks, filter_bands_n]
  :return:        -1..1              [channels_n, #blocks, filter_bands_n]
  """
  mdct_norm = (tf.abs(mdct_dB) - _dB_MIN) / (_dB_MAX - _dB_MIN)

  # tf.debugging.assert_less_equal(tf.reduce_max(mdct_norm), 1.,
  #                                message="normalization is not in -1..1 range (clipping...)")

  # this formula clips the very large amplitudes
  return tf.clip_by_value(tf.sign(mdct_dB) * mdct_norm, -1., 1.)


def norm_to_dB(mdct_norm):
  """Point-wise convert normalized mdct amplitudes in -1..1 range to amplitude in dB (range -inf..inf)

  :param mdct_norm: -1..1            [channels_n, #blocks, filter_bands_n]
  :return:          -inf..inf [dB]   [channels_n, #blocks, filter_bands_n]
  """
  mdct_dB = (_dB_MAX - _dB_MIN) * tf.abs(mdct_norm) + _dB_MIN

  return tf.sign(mdct_norm) * mdct_dB
