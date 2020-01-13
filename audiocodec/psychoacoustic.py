# !/usr/bin/env python

""" Psycho-acoustic model for lossy audio encoder (inspired by aac encoder)

Loosely based based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
"""

import tensorflow as tf

_dB_MIN = 0.
_dB_MAX = 100.


def setup(sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6):
    """Computes required initialization matrices (stateless, no OOP)

    :param sample_rate:       sample_rate
    :param alpha:             exponent for non-linear superposition (~0.6)
    :param filter_bands_n:    number of filter bands of the filter bank
    :param bark_bands_n:      number of bark bands
    :return:                  tuple with pre-computed required for encoder and decoder
    """
    W, W_inv = _bark_freq_mapping(sample_rate, bark_bands_n, filter_bands_n)

    quiet_threshold_amplitude = _quiet_threshold_amplitude_in_bark(sample_rate, bark_bands_n)
    spreading_matrix = _spreading_matrix_in_bark(sample_rate, bark_bands_n, alpha)

    return sample_rate, W, W_inv, quiet_threshold_amplitude, spreading_matrix, alpha


@tf.function
def scale_factors(mask_thresholds_log_bark, W_inv):
    """Compute scale-factors from logarithmic masking threshold.

    :param mask_thresholds_log_bark: logarithmic masking threshold [#channels, #blocks, bark_bands_n]
    :param W_inv:                    matrix to convert from filter bins to bark bins [bark_bands_n, filter_bands_n]
    :return:                         scale factors to be applied on amplitudes [#channels, #blocks, filter_bands_n]
    """
    with tf.name_scope('scale_factors'):
        mask_thresholds_trunc_bark = tf.pow(2., mask_thresholds_log_bark / 4.)

        mask_thresholds_trunc = _mappingfrombark(mask_thresholds_trunc_bark, W_inv)

        # maximum of the magnitude of the quantization error is delta/2
        mdct_scale_factors = 1. / (2. * mask_thresholds_trunc)

    return mdct_scale_factors


@tf.function
def global_masking_threshold(mdct_amplitudes, model_init, drown=0.0):
    """Determines which amplitudes we cannot hear, either since they are too soft
    to hear or because other louder amplitudes are masking it.
    Method uses non-linear superposition determined by factor alpha

    :param mdct_amplitudes:   vector of mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    :param model_init:        initialization data for the psychoacoustic model
    :param drown:             factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                  masking threshold [#channels, #blocks, filter_bands_n]
    """
    _, _, W_inv, _, _, _ = model_init

    global_mask_threshold = _mappingfrombark(
        _global_masking_threshold_in_bark(mdct_amplitudes, model_init, drown), W_inv)
    return global_mask_threshold


@tf.function
def smear_mdct_abs(mdct_abs, model_init):
    """Weaken filter: if amplitude in previous blocks was strong, then keep even weak amplitude unfiltered
    Reasoning: amplitudes oscillate from block to block, and hence passes through zero,
    but might still be helpful to keep as part of strong pattern.

    t_decay derived from temporal post-masking which can last 50-300ms after masker
    (see Ted Painter and Andreas Spanias, Perceptual Coding of Digital Audio)
    """
    sample_rate, W, _, _, _, _ = model_init
    filter_bands_n = tf.shape(W)[0]

    t_decay = 0.100  # 100ms
    t_block = filter_bands_n / sample_rate
    numbers_of_blocks_to_keep = int(t_decay / t_block)
    # set kernel [filter_height, filter_width, in_channels, out_channels]
    block_average_kernel = 1. / numbers_of_blocks_to_keep * tf.ones([numbers_of_blocks_to_keep, 1, 1, 1])
    mdct_norm_fadeout = tf.nn.conv2d(mdct_abs, block_average_kernel, strides=1, padding='same')

    return tf.maximum(mdct_abs, tf.abs(mdct_norm_fadeout))


@tf.function
def psychoacoustic_filter(mdct_amplitudes, model_init, drown=0.):
    """Filters out frequencies which are inaudible

    :param mdct_amplitudes:   vector of mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    :param model_init:        initialization data for the psychoacoustic model
    :param drown:             factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                  modified amplitudes [#channels, #blocks, filter_bands_n]
    """
    total_threshold = global_masking_threshold(mdct_amplitudes, model_init, drown)

    # Update spectrum
    # 1. remove anything below masking threshold
    mdct_modified = tf.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, mdct_amplitudes,
                             tf.fill(tf.shape(mdct_amplitudes), 1e-6))
    # 2. pass-through
    # mdct_modified = mdct_amplitudes
    # 3. put in masking threshold
    # mdct_modified = np.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, mdct_amplitudes, total_threshold)
    # 4. keep only masking threshold
    # mdct_modified = total_threshold
    # 5. keep only sound below masking threshold
    # mdct_modified = np.where(total_threshold ** 2.0 < mdct_amplitudes ** 2.0, 1e-6, mdct_amplitudes)

    return mdct_modified


@tf.function
def lrelu_filter(mdct_norm, psychoacoustic_init, beta = .2):
    """Filter out in-audible frequencies

    :param mdct_norm:           mdct amplitudes in rescaled dB         [batch_size, #blocks, filter_bands_n, #channels=1]
    :param psychoacoustic_init: psychoacoustic initialization data
    :return:                    filtered normalized mdct amplitudes    [batch_size, #blocks, filter_bands_n, #channels=1]
    """
    sample_rate, W, _, _, _, _ = psychoacoustic_init

    # get masking threshold in norm space
    # [batch_size, #blocks, filter_bands_n, 1]
    mdct_amplitudes = norm_to_ampl(mdct_norm)
    total_threshold = global_masking_threshold(mdct_amplitudes, psychoacoustic_init)
    total_masking_norm = ampl_to_norm(total_threshold)
    # [batch_size, #blocks, filter_bands_n, 1]
    # assume hearing threshold is fixed (since the gradients give nan and anyway the threshold moving is only 2nd order)
    total_masking_norm = tf.stop_gradient(total_masking_norm)

    # LeakyReLU-based filter: suppress in-audible frequencies (in norm space)
    mdct_abs = tf.abs(mdct_norm)
    mdct_abs_fadeout = smear_mdct_abs(mdct_abs)
    mdct_norm_filtered = tf.where(mdct_abs_fadeout <= tf.scalar_mul(1./(1.+beta), total_masking_norm),
                                  tf.scalar_mul(beta, mdct_abs),
                         tf.where(total_masking_norm <= mdct_abs_fadeout,
                                  mdct_abs,
                                  tf.scalar_mul(1./beta, mdct_abs) + tf.scalar_mul(1. - 1./beta, total_masking_norm)))
    mdct_norm_filtered = tf.multiply(tf.sign(mdct_norm), mdct_norm_filtered)

    return mdct_norm_filtered


@tf.function
def noise_filter(mdct_norm, psychoacoustic_init):
    """

    :param mdct_norm:
    :param psychoacoustic_init:
    :return:
    """
    # noise-based overlay: add noise to in-audible frequencies,
    # so learning does not learn anything about these (in norm space)
    # todo: continue here (decide what sigma to use for noise)
    # mdct_norm_noised = tf.where(total_masking_norm <= mdct_abs_fadeout,
    #                           mdct_norm,
    #                           noise)
    pass


def _global_masking_threshold_in_bark(mdct_amplitudes, model_init, drown=0.):
    """Determines which amplitudes we cannot hear, either since they are too soft
    to hear or because other louder amplitudes are masking it.
    Method uses non-linear superposition determined by factor alpha

    :param mdct_amplitudes:   vector of mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    :param model_init:        initialization data for the psychoacoustic model
    :param drown:             factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                  masking threshold [#channels, #blocks, bark_bands_n]
    """
    with tf.name_scope('global_masking_threshold'):
        sample_rate, W, _, quiet_threshold_amplitudes, spreading_matrix, alpha = model_init
        masking_threshold = _masking_threshold_in_bark(mdct_amplitudes, W, spreading_matrix, alpha, sample_rate, drown)

        # Take max between quiet threshold and masking threshold
        # Note: even though both thresholds are expressed as amplitudes,
        # they are all positive due to the way they were constructed
        global_mask_threshold_in_bark = tf.maximum(masking_threshold, quiet_threshold_amplitudes)

    return global_mask_threshold_in_bark


def _masking_threshold_in_bark(mdct_amplitudes, W, spreading_matrix, alpha, sample_rate, drown=0.):
    """Returns amplitudes that are masked by the sound defined by mdct_amplitudes

    :param mdct_amplitudes:   mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    :param W:                 matrix to convert from filter bins to bark bins [filter_bands_n, bark_bands_n]
    :param spreading_matrix:  spreading matrix [bark_bands_n, bark_bands_n]
    :param alpha:             exponent for non-linear superposition as applied on amplitudes
    :param drown:             factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                  vector of amplitudes in dB for softest audible sounds given a certain sound [#channels, #blocks, bark_bands_n]
    """
    bark_bands_n = spreading_matrix.shape[0]
    max_frequency = sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = freq2bark(max_frequency)

    # compute tonality (0:white noise ... 1:tonal) from the spectral flatness measure
    # (SFM = 0dB for white noise, SFM << 0dB for pure tone)
    # tonality = 10./-60. * \log_10  (e^{1/N \sum_{filter_band_i} \ln(a_i^2)}) /
    #                                (1/N \sum_{filter_band_i} a_i^2)
    tonality = tf.minimum(1.0, 10 * tf.math.log(tf.divide(
      tf.exp(tf.reduce_mean(tf.math.log(tf.pow(mdct_amplitudes, 2)), axis=2, keepdims=True)),
      tf.reduce_mean(tf.pow(mdct_amplitudes, 2), axis=2, keepdims=True))) / (-60.0 * tf.math.log(10.0)))
    # add bark_bands_n dimension: [#channels, #blocks, bark_bands_n]
    tonality = tf.tile(tonality, multiples=[1, 1, bark_bands_n])

    # compute masking offset: O(i) = tonality (14.5 + i) + (1 - tonality) 5.5
    # note: einsum('.i.,.i.->.i.') does an element-wise multiplication (and no sum) along a specified axes
    offset = (1. - drown) * \
             (tf.einsum('cbj,j->cbj', tonality, tf.linspace(0.0, max_bark, bark_bands_n)) + 9. * tonality + 5.5)

    # add offset to spreading matrix
    masking_matrix = tf.einsum('ij,cbj->cbij', spreading_matrix, tf.pow(10.0, -alpha * offset / 10.0))

    # Transposed version of (9.17) in Digital Audio Signal Processing by Udo Zolzer
    # \Sum_i (amplitude_i^2)^{\alpha} x [ mask^{\alpha}_{i-n} ]_n
    #   = \Sum amplitude_i x mask_{in}                       --> each row is a mask
    # Non-linear superposition (see p13 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)
    # \alpha ~ 0.3 is valid for 94 bark_bands_n; with less bark_bands_n 0.3 leads to (way) too much masking
    amplitudes_in_bark = _mapping2bark(mdct_amplitudes, W)
    return tf.pow(tf.einsum('cbi,cbij->cbj', tf.pow(amplitudes_in_bark, 2. * alpha), masking_matrix), 1. / (2. * alpha))


def _spreading_matrix_in_bark(sample_rate, bark_bands_n, alpha):
    """Returns (power) spreading matrix, to apply in bark scale to determine masking threshold from sound

    :param sample_rate:     sample rate
    :param bark_bands_n:    number of bark bands
    :param alpha:           exponent for non-linear superposition as applied on amplitudes
    :return:                spreading matrix [bark_bands_n, bark_bands_n]
    """
    max_frequency = sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = freq2bark(max_frequency)

    # Prototype spreading function [Bark/dB]
    f_spreading = tf.map_fn(lambda z: 15.81 + 7.5 * (z + 0.474) - 17.5 * tf.sqrt(1 + tf.pow(z + 0.474, 2)),
                            tf.linspace(-max_bark, max_bark, 2 * bark_bands_n))

    # Convert from dB to intensity and include alpha exponent
    f_spreading_intensity = tf.pow(10.0, alpha * f_spreading / 10.0)

    # Turns the spreading prototype function into a (bark_bands_n x bark_bands_n) matrix of shifted versions.
    # Transposed version of (9.17) in Digital Audio Signal Processing by Udo Zolzer
    spreading_matrix = tf.stack([f_spreading_intensity[(bark_bands_n - row):(2 * bark_bands_n - row)]
                                 for row in range(bark_bands_n)], axis=0)

    return spreading_matrix


def _quiet_threshold_amplitude_in_bark(sample_rate, bark_bands_n):
    """Compute the amplitudes of the softest sounds one can hear

    :param sample_rate:   sample rate
    :param bark_bands_n:  number of bark bands
    :return:              amplitude vector for softest audible sounds [1, 1, bark_bands_n]
    """
    # Threshold in quiet:
    max_frequency = sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = freq2bark(max_frequency)
    bark_band_width = max_bark / bark_bands_n

    bark_bands_mid_bark = bark_band_width * tf.range(bark_bands_n, dtype=tf.float32) + bark_band_width / 2.
    bark_bands_mid_kHz = bark2freq(bark_bands_mid_bark) / 1000.

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


def _bark_freq_mapping(sample_rate, bark_bands_n, filter_bands_n):
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

      :param sample_rate:     sample rate
      :param bark_bands_n:    number of bark bands
      :param filter_bands_n:  number of mdct filters
      :return: 2 matrices with shape
                  W      [filter_bank_n , bark_band_n]
                  W_inv  [bark_band_n   , filter_bank_n]
    """
    max_frequency = sample_rate / 2  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = freq2bark(max_frequency)
    bark_band_width = max_bark / bark_bands_n

    filter_band_width = max_frequency / tf.cast(filter_bands_n, tf.float32)
    filter_bands_mid_freq = filter_band_width * tf.range(filter_bands_n, dtype=tf.float32) + filter_band_width / 2
    filter_bands_mid_bark = freq2bark(filter_bands_mid_freq)

    column_index = tf.tile(tf.expand_dims(tf.range(bark_bands_n, dtype=tf.float32), axis=0),
                           multiples=[filter_bands_n, 1])
    W = tf.dtypes.cast(tf.equal(tf.tile(tf.expand_dims(tf.math.floor(filter_bands_mid_bark / bark_band_width), axis=1),
                                        multiples=[1, bark_bands_n]), column_index), dtype=tf.float32)

    # (bark_band_n x bark_band_n) . (bark_band_n x filter_bank_n)
    W_transpose = tf.transpose(W, perm=[1, 0])
    W_inv = tf.tensordot(tf.linalg.diag(tf.pow(1.0 / (1e-6 + tf.reduce_sum(W_transpose, axis=1)), 0.5)), W_transpose,
                         axes=[[1], [0]])

    return W, W_inv


def _mapping2bark(mdct_amplitudes, W):
    """Takes MDCT amplitudes and maps it into Bark bands amplitudes.
    Power spectral density of Bark band is sum of power spectral density in
    corresponding filter bands (power spectral density of signal S = X_1^2 + ... + X_n^2)
      (see also slide p9 of ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)

    :param mdct_amplitudes:  vector of mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    :param W:                matrix to convert from filter bins to bark bins [filter_bands_n, bark_bands_n]
    :return:                 vector of signal amplitudes of the Bark bands [#channels, #blocks, bark_bands_n]
    """
    return tf.pow(tf.tensordot(tf.pow(mdct_amplitudes, 2), W, axes=[[2], [0]]), 0.5)


def _mappingfrombark(amplitudes_bark, W_inv):
    """Takes Bark band amplitudes and maps it to MDCT amplitudes.
    Power spectral density of Bark band is split equally between the
    filter bands making up the Bark band (many-to-one).

    :param amplitudes_bark:  vector of signal amplitudes of the Bark bands [#channels, #blocks, bark_bands_n]
    :param W_inv:            matrix to convert from filter bins to bark bins [bark_bands_n, filter_bands_n]
    :return:                 vector of mdct amplitudes (spectrum) for each filter [#channels, #blocks, filter_bands_n]
    """
    return tf.tensordot(amplitudes_bark, W_inv, axes=[[2], [0]])


def freq2bark(frequencies):
    """Empirical Bark scale"""
    return 6. * tf.asinh(frequencies / 600.)


def bark2freq(bark_band):
    """Empirical Bark scale"""
    return 600. * tf.sinh(bark_band / 6.)


def ampl_to_norm(mdct_amplitudes):
    """Convert mdct amplitudes to normalized dB scale

    :param mdct_amplitudes:  -inf..inf          [channels_n, #blocks, filter_bands_n]
    :return:                 -1..1              [channels_n, #blocks, filter_bands_n]
    """
    # this formula clips the very small amplitudes (so log() does not become negative)
    mdct_dB = tf.sign(mdct_amplitudes) * 20. * \
              tf.maximum(
                  tf.math.log1p(tf.abs(mdct_amplitudes) - 1.),
                  0.0) / tf.math.log(10.)

    mdct_norm = dB_to_norm(mdct_dB)

    return mdct_norm


def norm_to_ampl(mdct_norm):
    """Convert mdct amplitudes in normalized dB scale to actual amplitude values

    :param mdct_norm:   -1..1              [channels_n, #blocks, filter_bands_n]
    :return:            -inf..inf          [channels_n, #blocks, filter_bands_n]
    """
    mdct_dB = norm_to_dB(mdct_norm)
    mdct_amplitudes = tf.sign(mdct_dB) * tf.pow(10., tf.abs(mdct_dB) / 20.)

    return mdct_amplitudes


def dB_to_norm(mdct_dB):
    """Linearly normalize mdct amplitude expressed in dB (range -inf..inf) to -1..1 range

    :param mdct_dB: -inf..inf [dB]     [channels_n, #blocks, filter_bands_n]
    :return:        -1..1              [channels_n, #blocks, filter_bands_n]
    """
    mdct_norm = (mdct_dB - _dB_MIN) / (_dB_MAX - _dB_MIN)

    # this formula clips the very large amplitudes
    return tf.clip_by_value(mdct_norm, -1., 1.)


def norm_to_dB(mdct_norm):
    """Convert normalized mdct amplitudes in -1..1 range to amplitude in dB (range -inf..inf)

    :param mdct_norm: -1..1            [channels_n, #blocks, filter_bands_n]
    :return:          -inf..inf [dB]   [channels_n, #blocks, filter_bands_n]
    """
    mdct_dB = (_dB_MAX - _dB_MIN) * mdct_norm + _dB_MIN

    return mdct_dB
