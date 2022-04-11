# !/usr/bin/env python

""" Psycho-acoustic model for lossy audio encoder (inspired by aac encoder)

Loosely based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
See also chapter 9 in "Digital Audio Signal Processing" by Udo Zolzer
"""

import math
import tensorflow as tf


class PsychoacousticModel:
  def __init__(self, sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6,
               compute_dtype=tf.float32, precompute_dtype=tf.float64):
    """Computes required initialization matrices.

    For standard MP3 encoding, they use filter_bands_n=1024 and bark_bands_n=64.
    If one deviates from these parameters, both the levels of the global masking threshold and the quiet threshold
    can vary, since both are established in the bark scale and then converted to the frequency scale.
    In that conversion process, the bark energy gets dissipated into (more) frequency buckets and
    the frequency by frequency threshold is hence lowered then more frequency buckets or the less bark bands one has.
    todo: add normalization coefficient in definition of W and W_inv to solve this issue

    :param sample_rate:       sample_rate
    :param alpha:             exponent for non-linear superposition
                              lower means lower quality (1.0 is linear superposition, 0.6 is default)
    :param filter_bands_n:    number of filter bands of the filter bank
    :param bark_bands_n:      number of bark bands
    :param compute_dtype      The compute dtype of model. Inputs dtype must match compute_dtype (no implicit casting)
                              Can be tf.float64, tf.float32 or tf.bfloat16 (defaults to tf.float32)
                              Note: tf.float16 is not allowed since float16 does not allow for enough range in
                                the exponent of the amplitudes to get correct results
    :return:                  tuple with pre-computed required for encoder and decoder
    :raises TypeError         when compute_dtype is not
    """
    self.alpha = alpha
    self.sample_rate = sample_rate
    self.bark_bands_n = bark_bands_n
    self.filter_bands_n = filter_bands_n

    if compute_dtype not in [tf.float64, tf.float32, tf.bfloat16]:
      raise TypeError("compute_dtype of PsychoacousticModel should be tf.float64, tf.float32 or tf.bfloat16")
    self.compute_dtype = compute_dtype

    # _dB_MAX corresponds with abs(amplitude) = 1.
    # To calibrate _dB_MAX, we used audio signals which come from mp3 encodings. This audio data does not have the
    # high frequency components (>15.5Hz) since they were removed during the mp3 encoding (high quiet threshold
    # for high frequencies). Since _dB_MAX determines how mdct amplitudes are converted to dB and hence how
    # mdct amplitudes compare against the absolute scale of the quiet threshold, we were able to fix _dB_MAX at same
    # level that was used in the original mp3 encodings
    self._dB_MAX = tf.constant(120., dtype=compute_dtype)

    # _INTENSITY_EPS corresponds with a dB_MIN of -140dB + _dB_MAX
    # _dB_MIN needs to be lower than -9dB, since the quiet threshold at its minimum is -9dB
    self._INTENSITY_EPS = tf.constant(1e-14, dtype=compute_dtype)

    self._dB_MIN = self.amplitude_to_dB(self._INTENSITY_EPS)  # = -20dB

    # pre-compute some values & matrices with higher precision, then down-cast to compute_dtype
    self.max_frequency = tf.cast(self.sample_rate, dtype=precompute_dtype) / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    self.max_bark = self.freq2bark(self.max_frequency)
    self.bark_band_width = self.max_bark / self.bark_bands_n

    W, W_inv = self._bark_freq_mapping(precompute_dtype=precompute_dtype)
    self.W = tf.cast(W, dtype=compute_dtype)
    self.W_inv = tf.cast(W_inv, dtype=compute_dtype)
    self.quiet_threshold_intensity = tf.cast(self._quiet_threshold_intensity_in_bark(precompute_dtype=precompute_dtype), dtype=compute_dtype)
    self.spreading_matrix = tf.cast(self._spreading_matrix_in_bark(), dtype=compute_dtype)

  def amplitude_to_dB(self, mdct_amplitude):
    """Utility function to convert the amplitude which is normalized in the [-1..1] range
    to the dB scale.
    The dB scale is set such that
    1. an amplitude squared (intensity) of 1 corresponds to the maximum dB level (_dB_MAX), and
    2. an amplitude squared (intensity) of _INTENSITY_EPS corresponds with the minimum dB level (_dB_MIN)

    :param mdct_amplitude:  amplitude normalized in [-1, 1] range
                            must be of compute_dtype
    :return:                corresponding dB scale in [_dB_MIN, _dB_MAX] range (positive)
                            output dtype is compute_dtype
    """
    ampl_dB = 10. * tf.math.log(tf.maximum(self._INTENSITY_EPS, mdct_amplitude ** 2.0)) / tf.math.log(
      tf.constant(10., dtype=self.compute_dtype)) + self._dB_MAX
    return ampl_dB

  def amplitude_to_dB_norm(self, mdct_amplitude):
    """Utility function to convert the amplitude which is normalized in the [-1..1] range
    to the normalized dB scale [0, 1]

    output = (self.amplitude_to_dB(mdct_amplitude) - self._dB_MIN) / (self._dB_MAX - self._dB_MIN)
           = 1 - 2 \ln(mdct_amplitude) / \ln(self._INTENSITY_EPS)

    :param mdct_amplitude:  amplitude normalized in [-1, 1] range
                            must be of compute_dtype
    :return:                corresponding dB scale in [0, 1] range (positive)
                            output dtype is compute_dtype
    """
    ampl_dB = self.amplitude_to_dB(mdct_amplitude)
    return (ampl_dB - self._dB_MIN) / (self._dB_MAX - self._dB_MIN)

  @tf.function
  def tonality(self, mdct_amplitudes):
    """
    Compute tonality (0:white noise ... 1:tonal) from the spectral flatness measure (SFM)
    See equations (9.10)-(9.11) in Digital Audio Signal Processing by Udo Zolzer

    :param mdct_amplitudes:     mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
                                must be of compute_dtype
    :return:                    tonality vector. Shape: [batches_n, blocks_n, 1, channels_n]
                                output dtype is compute_dtype
    """
    mdct_intensity = tf.pow(mdct_amplitudes, 2)
    sfm = 10. * tf.math.log(tf.divide(
      tf.exp(tf.reduce_mean(tf.math.log(tf.maximum(self._INTENSITY_EPS, mdct_intensity)), axis=2, keepdims=True)),
      tf.reduce_mean(mdct_intensity, axis=2, keepdims=True) + self._INTENSITY_EPS)) / math.log(10.0)

    sfm = tf.minimum(sfm / -60., 1.0)

    return sfm

  @tf.function
  def global_masking_threshold(self, mdct_amplitudes, tonality_per_block, drown=0.0):
    """Determines which amplitudes we cannot hear, either since they are too soft
    to hear or because other louder amplitudes are masking it.
    Method uses non-linear superposition determined by factor self.alpha

    :param mdct_amplitudes:     mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
                                must be of compute_dtype
    :param tonality_per_block:  tonality vector associated with the mdct_amplitudes [batches_n, blocks_n, 1, channels_n]
                                can be computed using the method tonality(mdct_amplitudes)
                                needs to be of compute_dtype
    :param drown:               factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                    masking threshold in amplitude. Masking threshold is never negative
                                output dtype is compute_dtype
                                [batches_n, blocks_n, filter_bands_n, channels_n]
    """
    with tf.name_scope('global_masking_threshold'):
      masking_intensity = self._masking_intensity_in_bark(mdct_amplitudes, tonality_per_block, drown)

      # Take max between quiet threshold and masking threshold
      # Note: even though both thresholds are expressed as amplitudes,
      # they are all positive due to the way they were constructed
      global_masking_intensity_in_bark = tf.maximum(masking_intensity, self.quiet_threshold_intensity)

      global_mask_threshold = self._bark_intensity_to_freq_ampl(global_masking_intensity_in_bark)

    return global_mask_threshold

  @tf.function
  def add_noise(self, mdct_amplitudes, masking_threshold):
    """
    Adds inaudible noise to amplitudes, using the masking_threshold.
    The noise added is calibrated at a 3-sigma deviation in both directions:
      masking_threshold = 6*sigma
    As such, there is a 0.2% probability that the noise added is bigger than the masking_threshold

    :param mdct_amplitudes:     mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
                                must be of compute_dtype
    :param masking_threshold:   masking threshold in amplitude. Masking threshold is never negative
                                output dtype is compute_dtype
                                [batches_n, blocks_n, filter_bands_n, channels_n]
    :return:                    mdct amplitudes with inaudible noise added [batches_n, blocks_n, filter_bands_n, channels_n]
    """
    noise = masking_threshold * tf.random.normal(shape=mdct_amplitudes.shape, mean=0., stddev=1. / 6., dtype=self.compute_dtype)

    return mdct_amplitudes + noise

  def _masking_intensity_in_bark(self, mdct_amplitudes, tonality_per_block, drown=0.0):
    """Returns amplitudes that are masked by the sound defined by mdct_amplitudes

    :param mdct_amplitudes:     mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
                                Should be of compute_dtype
    :param tonality_per_block:  tonality vector associated with the mdct_amplitudes [batches_n, blocks_n, 1, channels_n]
                                Should be of compute_dtype
    :param drown:               factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                    vector of intensities for softest audible sounds given a certain sound
                                [batches_n, blocks_n, bark_bands_n, channels_n]
    """
    # compute masking offset:
    #    O(i) = tonality (14.5 + i) + (1 - tonality) 5.5
    # with i the bark index
    # see p10 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)
    # in einsum, we tf.squeeze() axis=2 (index i) and take outer product with tf.linspace()
    offset = (1. - drown) * (tf.einsum('nbic,j->nbjc',
                                       tonality_per_block,
                                       tf.linspace(tf.constant(0.0, dtype=self.compute_dtype),
                                                   tf.cast(self.max_bark, dtype=self.compute_dtype),
                                                   self.bark_bands_n))
                             + 9. * tonality_per_block
                             + 5.5)

    # add offset to spreading matrix (see (9.18) in "Digital Audio Signal Processing" by Udo Zolzer)
    # note: einsum('.j.,.j.->.j.') multiplies elements on diagonal element-wise (without summing over j)
    masking_matrix = tf.einsum('ij,nbjc->nbijc',
                               self.spreading_matrix,
                               tf.pow(tf.constant(10.0, dtype=self.compute_dtype), -self.alpha * offset / 10.0))

    # Transposed version of (9.17) in Digital Audio Signal Processing by Udo Zolzer
    # \Sum_i (amplitude_i^2)^{\alpha} x [ mask^{\alpha}_{i-n} ]_n
    #   = \Sum amplitude_i x mask_{in}                       --> each row is a mask
    # Non-linear superposition (see p13 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)
    # \alpha ~ 0.3 is valid for 94 bark_bands_n; with less bark_bands_n 0.3 leads to (way) too much masking
    intensities_in_bark = self._to_bark_intensity(mdct_amplitudes)
    masking_intensity_in_bark = tf.einsum('nbic,nbijc->nbjc',
                                          tf.pow(tf.maximum(self._INTENSITY_EPS, intensities_in_bark), self.alpha),
                                          masking_matrix)
    masking_intensity_in_bark = tf.pow(tf.maximum(self._INTENSITY_EPS, masking_intensity_in_bark), 1. / self.alpha)

    return masking_intensity_in_bark

  def _spreading_matrix_in_bark(self):
    """Returns (power) spreading matrix, to apply in bark scale to determine masking threshold from sound

    :return:                spreading matrix [bark_bands_n, bark_bands_n]
    """
    # Prototype spreading function [Bark/dB]
    # see equation (9.15) in "Digital Audio Signal Processing" by Udo Zolzer
    f_spreading = tf.map_fn(lambda z: 15.81 + 7.5 * (z + 0.474) - 17.5 * tf.sqrt(1 + tf.pow(z + 0.474, 2)),
                            tf.linspace(-self.max_bark, self.max_bark, 2 * self.bark_bands_n))

    # Convert from dB to intensity and include alpha exponent
    f_spreading_intensity = tf.pow(tf.constant(10.0, dtype=f_spreading.dtype), self.alpha * f_spreading / 10.0)

    # Turns the spreading prototype function into a (bark_bands_n x bark_bands_n) matrix of shifted versions.
    # Transposed version of (9.17) in Digital Audio Signal Processing by Udo Zolzer
    spreading_matrix = tf.stack([f_spreading_intensity[(self.bark_bands_n - row):(2 * self.bark_bands_n - row)]
                                 for row in range(self.bark_bands_n)], axis=0)

    return spreading_matrix

  def _quiet_threshold_intensity_in_bark(self, precompute_dtype):
    """Compute the intensity of the softest sounds one can hear
       See (9.3) in "Digital Audio Signal Processing" by Udo Zolzer

    :return:       intensity vector for softest audible sounds [1, 1, bark_bands_n, 1]
                   returned amplitudes are all positive
    """
    # Threshold in quiet:
    bark_bands_mid_bark = self.bark_band_width * tf.range(self.bark_bands_n, dtype=precompute_dtype) + self.bark_band_width / 2.
    bark_bands_mid_kHz = self.bark2freq(bark_bands_mid_bark) / 1000.

    # Threshold of quiet expressed as amplitude in dB-scale for each Bark bands
    # see (9.3) in "Digital Audio Signal Processing" by Udo Zolzer
    quiet_threshold_dB = tf.clip_by_value(
      (3.64 * (tf.pow(bark_bands_mid_kHz, -0.8))
       - 6.5 * tf.exp(-0.6 * tf.pow(bark_bands_mid_kHz - 3.3, 2.))
       + 1e-3 * (tf.pow(bark_bands_mid_kHz, 4.))),
      tf.cast(self._dB_MIN, dtype=precompute_dtype), tf.cast(self._dB_MAX, dtype=precompute_dtype))

    # convert to amplitude scale, where _dB_MAX corresponds with an amplitude of 1.0
    quiet_threshold_intensity = tf.pow(tf.constant(10.0, dtype=precompute_dtype),
                                       (quiet_threshold_dB - tf.cast(self._dB_MAX, dtype=precompute_dtype)) / 10)

    return tf.reshape(quiet_threshold_intensity, shape=[1, 1, -1, 1])

  def _bark_freq_mapping(self, precompute_dtype):
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
    filter_band_width = self.max_frequency / self.filter_bands_n

    def freq_interval_overlap(freq_index, bark_index):
      bark_low = self.bark_band_width * bark_index
      bark_low_in_Hz = tf.broadcast_to(self.bark2freq(bark_low), shape=[self.filter_bands_n, self.bark_bands_n])
      bark_high_in_Hz = tf.broadcast_to(self.bark2freq(bark_low + self.bark_band_width), shape=[self.filter_bands_n, self.bark_bands_n])

      freq_low = filter_band_width * freq_index
      bark_low_in_Hz_clipped = tf.clip_by_value(bark_low_in_Hz, clip_value_min=freq_low, clip_value_max=freq_low + filter_band_width)
      bark_high_in_Hz_clipped = tf.clip_by_value(bark_high_in_Hz, clip_value_min=freq_low, clip_value_max=freq_low + filter_band_width)

      overlap = bark_high_in_Hz_clipped - bark_low_in_Hz_clipped
      return overlap / filter_band_width, overlap / (bark_high_in_Hz - bark_low_in_Hz)

    bark_columns = tf.reshape(tf.range(self.bark_bands_n, dtype=precompute_dtype), shape=[1, -1])
    freq_rows = tf.reshape(tf.range(self.filter_bands_n, dtype=precompute_dtype), shape=[-1, 1])
    W, W_inv_transpose = freq_interval_overlap(freq_rows, bark_columns)

    return W, tf.transpose(W_inv_transpose, perm=[1, 0])

  def _to_bark_intensity(self, mdct_amplitudes):
    """Takes MDCT amplitudes and maps it into Bark bands amplitudes.
    Power spectral density of Bark band is sum of power spectral density in
    corresponding filter bands (power spectral density of signal S = X_1^2 + ... + X_n^2)
      (see also slide p9 of ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)

    :param mdct_amplitudes:  vector of mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
    :param W:                matrix to convert from filter bins to bark bins [filter_bands_n, bark_bands_n]
    :return:                 vector of intensities per Bark band [batches_n, blocks_n, bark_bands_n, channels_n]
    """
    # tf.maximum() is necessary, to make sure rounding errors don't make the gradient nan!
    mdct_intensity = tf.pow(mdct_amplitudes, 2)
    mdct_intensity_in_bark = tf.einsum('nbic,ij->nbjc', mdct_intensity, self.W)

    return mdct_intensity_in_bark

  def _bark_intensity_to_freq_ampl(self, bark_intensity):
    """Takes Bark band intensity and maps it to MDCT amplitudes.
    Power spectral density of Bark band is split equally between the
    filter bands making up the Bark band (one-to-many). As a result,
    intensities in the bark scale get smeared out in the frequency scale.
    For higher frequencies, this smearing effect becomes more important.
    As a result, e.g. the quiet threshold which is specified in the bark scale,
    gets smeared out in the frequency scale (and looks hence lower in value!)

    :param bark_intensity:   vector of signal intensities in the Bark bands [batches_n, blocks_n, bark_bands_n, channels_n]
    :param W_inv:            matrix to convert from filter bins to bark bins [bark_bands_n, filter_bands_n]
    :return:                 vector of mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
    """
    mdct_intensity = tf.einsum('nbic,ij->nbjc', bark_intensity, self.W_inv)
    return tf.pow(tf.maximum(self._INTENSITY_EPS, mdct_intensity), 0.5)

  def freq2bark(self, frequencies):
    """Empirical Bark scale"""
    return 6. * tf.asinh(frequencies / 600.)

  def bark2freq(self, bark_band):
    """Empirical Bark scale"""
    return 600. * tf.sinh(bark_band / 6.)
