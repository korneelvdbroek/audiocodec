# !/usr/bin/env python

""" Psycho-acoustic model for lossy audio encoder (inspired by aac encoder)

Based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
"""

import numpy as np


def bark_freq_mapping(sample_rate, bark_bands_n, filter_bands_n):
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
                  W      (filter_bank_n x bark_band_n)
                  W_inv  (bark_band_n   x filter_bank_n)
    """
    max_frequency = sample_rate / 2  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = freq2bark(max_frequency)
    bark_band_width = max_bark / bark_bands_n

    filter_band_width = max_frequency / filter_bands_n
    filter_bands_mid_freq = np.arange(filter_bands_n) * filter_band_width + filter_band_width / 2
    filter_bands_mid_bark = freq2bark(filter_bands_mid_freq)

    W = np.zeros((filter_bands_n, bark_bands_n))
    for i in range(bark_bands_n):
        # copy in column by column
        W[:, i] = (np.trunc(filter_bands_mid_bark / bark_band_width) == i)

    # (bark_band_n x bark_band_n) . (bark_band_n x filter_bank_n)
    W_inv = np.dot(np.diag(np.power(1.0 / (np.sum(W.T, axis=1) + 1e-6), 0.5)), W.T)

    return W, W_inv


def global_masking_threshold_in_bark(mdct_amplitudes, W, spreading_matrix, quiet_threshold, alpha, sample_rate):
    """Determines which amplitudes we cannot hear, either since they are too soft
    to hear or because other louder amplitudes are masking it.
    Method uses non-linear superposition determined by factor alpha

    :param mdct_amplitudes:   vector of mdct amplitudes (spectrum) for each filter (#channels, filter_bands_n, #blocks+1)
    :param W:                 matrix to convert from filter bins to bark bins (filter_bands_n x bark_bands_n)
    :param spreading_matrix:  spreading matrix (bark_bands_n x bark_bands_n)
    :param quiet_threshold:   amplitude vector for softest audible sounds given a certain sound (bark_bands_n)
    :param alpha:             exponent for non-linear superposition
    :param sample_rate:       sample rate
    :return:                  masking threshold (#channels x #blocks x bark_bands_n)
    """
    masking_threshold = _masking_threshold_in_bark(mdct_amplitudes, W, spreading_matrix, alpha, sample_rate)

    # Take max between quiet threshold and masking threshold
    # Note: even though both thresholds are expressed as amplitudes,
    # they are all positive due to the way they were constructed
    return np.maximum(masking_threshold, quiet_threshold[np.newaxis, np.newaxis, :])


def _masking_threshold_in_bark(mdct_amplitudes, W, spreading_matrix, alpha, sample_rate):
    """Returns amplitudes that are masked by the sound defined by mdct_amplitudes

    :param mdct_amplitudes:   vector of mdct amplitudes (spectrum) for each filter (#channels, filter_bands_n, #blocks+1)
    :param W:                 matrix to convert from filter bins to bark bins (filter_bands_n x bark_bands_n)
    :param spreading_matrix:  spreading matrix (bark_bands_n x bark_bands_n)
    :param alpha:             exponent for non-linear superposition as applied on amplitudes
    :return:                  amplitude vector for softest audible sounds given a certain sound (bark_bands_n)
    """
    filter_bands_n = mdct_amplitudes.shape[1]
    bark_bands_n = spreading_matrix.shape[0]
    max_frequency = sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = freq2bark(max_frequency)

    amplitudes_in_bark = _mapping2bark(mdct_amplitudes, W)

    # compute tonality from the spectral flatness measure (SFM = 0dB for noise, SFM << 0dB for tone)
    tonality = np.minimum(1.0, 10 * np.log10(np.divide(
      np.exp(1 / filter_bands_n * np.sum(np.log(mdct_amplitudes ** 2.0), axis=1)),
      1 / filter_bands_n * np.sum(mdct_amplitudes ** 2.0, axis=1))) / (-60.0))
    tonality = np.repeat(np.expand_dims(tonality, axis=2), bark_bands_n, axis=2)

    # compute masking offset O(i) = \alpha (14.5 + z) + (1 - \alpha) 5.5
    # note: einsum('.i.,.i.->.i.') does an element-wise multiplication (and no sum) along a specified axes
    offset = np.einsum('cbj,j->cbj', tonality, np.linspace(0, max_bark, bark_bands_n)) + 9. * tonality + 5.5

    # add offset to spreading matrix
    masking_matrix = np.einsum('ij,cbj->cbij', spreading_matrix, 10.0 ** (-alpha * offset / 10.0))

    # Transposed version of (9.17) in Digital Audio Signal Processing by Udo Zolzer
    # \Sum_i (amplitude_i^2)^{\alpha} x [ mask^{\alpha}_{i-n} ]_n
    #   = \Sum amplitude_i x mask_{in}                       --> each row is a mask
    # Non-linear superposition (see p13 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)
    # \alpha ~ 0.3 is valid for 94 bark_bands_n; with less bark_bands_n 0.3 leads to (way) too much masking
    return np.einsum('cbi,cbij->cbj', amplitudes_in_bark ** (2 * alpha), masking_matrix) ** (1 / (2 * alpha))


def spreading_matrix_in_bark(sample_rate, bark_bands_n, alpha):
    """Returns (power) spreading matrix, to apply in bark scale to determine masking threshold from sound

    :param sample_rate:     sample rate
    :param bark_bands_n:    number of bark bands
    :param alpha:           exponent for non-linear superposition as applied on amplitudes
    :return:                spreading matrix (bark_bands_n x bark_bands_n)
    """
    max_frequency = sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = freq2bark(max_frequency)

    # Prototype spreading function [Bark/dB]
    f_spreading = np.array([15.81 + 7.5 * (z + 0.474) - 17.5 * np.sqrt(1 + (z + 0.474) ** 2)
                            for z in np.linspace(-max_bark, max_bark, 2 * bark_bands_n)])

    # Convert from dB to intensity and include alpha exponent
    f_spreading_intensity = 10.0 ** (alpha * f_spreading / 10.0)

    # Turns the spreading prototype function into a (bark_bands_n x bark_bands_n) matrix of shifted versions.
    # Transposed version of (9.17) in Digital Audio Signal Processing by Udo Zolzer
    spreading_matrix = np.zeros((bark_bands_n, bark_bands_n))
    for row in range(bark_bands_n):
        # copy in row k
        spreading_matrix[row, :] = f_spreading_intensity[(bark_bands_n - row):(2 * bark_bands_n - row)]

    return spreading_matrix


def quiet_threshold_in_bark(sample_rate, bark_bands_n):
    """Compute the amplitudes of the softest sounds one can hear

    :param sample_rate:   sample rate
    :param bark_bands_n:  number of bark bands
    :return:              amplitude vector for softest audible sounds (bark_bands_n)
    """
    # Threshold in quiet:
    max_frequency = sample_rate / 2.0  # Nyquist frequency: maximum frequency given a sample rate
    max_bark = freq2bark(max_frequency)
    bark_band_width = max_bark / bark_bands_n

    bark_bands_mid_bark = np.arange(bark_bands_n) * bark_band_width + bark_band_width / 2.
    bark_bands_mid_kHz = bark2freq(bark_bands_mid_bark) / 1000.

    # Threshold of quiet expressed as amplitude (dB) for each Bark bands
    # (see also p4 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)
    quiet_threshold_dB = np.clip(
      (3.64 * (bark_bands_mid_kHz ** (-0.8))
       - 6.5 * np.exp(-0.6 * (bark_bands_mid_kHz - 3.3) ** 2.)
       + 1e-3 * (bark_bands_mid_kHz ** 4.)),
      -20, 160)

    # convert dB to amplitude
    return 10.0 ** (quiet_threshold_dB / 20)


def scale_factors(mask_thresholds_log_bark, W_inv):
    """Compute scale-factors from logarithmic masking threshold.

    :param mask_thresholds_log_bark: logarithmic masking threshold (#channels x #blocks x bark_bands_n)
    :param W_inv:                    matrix to convert from filter bins to bark bins (bark_bands_n x filter_bands_n)
    :return:                         scale factors to be applied on amplitudes (#channels x filter_bands_n x #blocks)
    """
    mask_thresholds_trunc_bark = np.power(2, mask_thresholds_log_bark / 4)

    mask_thresholds_trunc = _mappingfrombark(mask_thresholds_trunc_bark, W_inv)

    # maximum of the magnitude of the quantization error is delta/2
    return 1 / (2 * np.transpose(mask_thresholds_trunc, axes=(0, 2, 1)))


def _mapping2bark(mdct_amplitudes, W):
    """Takes MDCT amplitudes and maps it into Bark bands amplitudes.
    Power spectral density of Bark band is sum of power spectral density in
    corresponding filter bands (power spectral density of signal S = X_1^2 + ... + X_n^2)
      (see also slide p9 of ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)

    :param mdct_amplitudes:  vector of mdct amplitudes (spectrum) for each filter (#channels x filter_bands_n x #blocks)
    :param W:                matrix to convert from filter bins to bark bins (filter_bands_n x bark_bands_n)
    :return:                 vector of signal amplitudes of the Bark bands (#channels x #blocks x bark_bands_n)
    """
    return (np.tensordot(mdct_amplitudes ** 2.0, W, axes=(1, 0))) ** 0.5


def _mappingfrombark(amplitudes_bark, W_inv):
    """Takes Bark band amplitudes and maps it to MDCT amplitudes.
    Power spectral density of Bark band is split equally between the
    filter bands making up the Bark band (many-to-one).

    :param amplitudes_bark:  vector of signal amplitudes of the Bark bands (#channels x bark_bands_n x #blocks)
    :param W_inv:            matrix to convert from filter bins to bark bins (bark_bands_n x filter_bands_n)
    :return:                 vector of mdct amplitudes (spectrum) for each filter (#channels x #blocks x filter_bands_n)
    """
    return np.dot(amplitudes_bark, W_inv)


def freq2bark(frequencies):
    """Empirical Bark scale"""
    return 6. * np.arcsinh(frequencies / 600.)


def bark2freq(bark_band):
    """Empirical Bark scale"""
    return 600. * np.sinh(bark_band / 6.)
