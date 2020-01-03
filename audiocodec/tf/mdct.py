#!/usr/bin/env python

""" Implements a MDCT transformation and inverse transformation on (channel x signal data)

Loosely based based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
"""

import tensorflow as tf
import math

_LOG_EPS = 1e-6
_dB_MIN = -20
_dB_MAX = 120


@tf.function
def setup(filters_n=1024):
    """Computes required initialization matrices (stateless, no OOP)

    :param filters_n:   number of filter bands of the filter bank
    :return:            tuple with pre-computed required for encoder and decoder
    """
    H = _polyphase_matrix(filters_n)
    H_inv = _inv_polyphase_matrix(filters_n)

    return filters_n, H, H_inv


@tf.function
def _polyphase_matrix(filters_n):
    """Decomposed part of poly-phase matrix of an MDCT filter bank, with a sine modulated window:
      H(z) = F_{analysis} x D x DCT4 =

    :param filters_n:  number of MDCT filters
    :return:           F_{analysis} x D           [filters_n, filters_n, 2]
    """
    with tf.name_scope('poly_phase_matrix'):
        F_analysis = tf.expand_dims(_filter_window_matrix(filters_n), axis=1)     # [filters_n, 1, filters_n]
        D = _delay_matrix(filters_n)                                              # [2, filters_n, filters_n]
        polyphase_matrix = _polmatmult(F_analysis, D)                             # [filters_n, 2, filters_n]

    return tf.transpose(polyphase_matrix, perm=[1, 0, 2])                         # [2, filters_n, filters_n]


@tf.function
def _inv_polyphase_matrix(filters_n):
    """Decomposed part of inverse poly-phase matrix of an MDCT filter bank, with a sine modulated window:
      G(z) = DCT4 x D^-1 x F_{synthesis}

    :param filters_n:   number of MDCT filters
    :return:            D^-1 x F_{synthesis}       [2, filters_n, filters_n]
    """
    with tf.name_scope('inv_poly_phase_matrix'):
        # invert Fa matrix for synthesis after removing last dim:
        F_synthesis = tf.expand_dims(
                        tf.linalg.inv(_filter_window_matrix(filters_n)), axis=0)        # [1, filters_n, filters_n]
        D_inv = _inverse_delay_matrix(filters_n)                                        # [filters_n, 2, filters_n]
        inv_polyphase_matrix = _polmatmult(D_inv, F_synthesis)                          # [filters_n, 2, filters_n]

    return tf.transpose(inv_polyphase_matrix, perm=[1, 0, 2])                           # [2, filters_n, filters_n]


@tf.function
def transform(x, model_init):
    """MDCT (Modulated Discrete Cosine Transform) analysis filter bank. Filters in MDCT use window-modulated filters.

    Basic structure:

      audio --+---> filter 1         --> down-sample by filters_n -->
              |
              +---> filter 2         --> down-sample by filters_n -->
              |
             ...
              |
              +---> filter filters_n --> down-sample by filters_n -->

    Down-sampling of the output of the filters_n filters by factor filters_n is
    without loss of information (Nyquist Theorem).
    Input has #samples, output has same amount of data (#samples = filters_n x #blocks).

    To improve computational efficiency, order of filter and down-sampling can be switched around (Noble Identities).
    Each filter itself needs to be split into its filters_n poly-phase decompositions (see p25 ./docs/mrate.pdf):

      x ---+-----> down-sample by filters_n --> [[     ]  [     ]     [             ]]
           |z^-1                                [[ fil ]  [ fil ]     [             ]]
           +-----> down-sample by filters_n --> [[ ter ]  [ ter ] ... [   filter    ]]
           |z^-1                                [[  1  ]  [  2  ]     [  filters_n  ]] = H(z) = F_{analysis} x D x DCT4
          ...                                   [[     ]  [     ]     [             ]]
           |z^-1                                [[     ]  [     ]     [             ]]
           +-----> down-sample by filters_n --> [[     ]  [     ]     [             ]]

    with F_{analysis}: filter window matrix
         D:            delay matrix
         DCT4:         discrete cosine transformation matrix

    read more:
      ./docs/02_shl_Filterbanks1_NobleID_WS2016-17_o.pdf
      ./docs/03_shl_FilterBanks2_WS2016-17.pdf
      ./docs/mrate.pdf

    :param x:            signal data assumed in -1..1 range [channels, #samples]
    :param model_init:   initialization data for the mdct model
    :return:             filters_n coefficients of MDCT transform for each block [#channels, #blocks+1, filters_n]
                         where #samples = filters_n x #blocks
    """
    filters_n, H, _ = model_init

    with tf.name_scope('mdct_transform'):
        # split signal into blocks
        x_pp = _x2polyphase(x, filters_n)               # [#channels, #blocks, filters_n]

        # put x through filter bank
        mdct_amplitudes = _dct4(_polmatmult(x_pp, H))   # [#channels, #blocks+1, filters_n]

    return mdct_amplitudes


@tf.function
def inverse_transform(y, model_init):
    """MDCT synthesis filter bank.

    :param y:            encoded signal in 3d array [#channels, #blocks, filters_n]
    :param model_init:   initialization data for the mdct model
    :return:             restored signal (#channels x #samples)
    """
    _, _, H_inv = model_init

    with tf.name_scope('inv_mdct_transform'):
        # put y through inverse filter bank
        x_pp = _polmatmult(_dct4(y), H_inv)

        # glue back the blocks to one signal
        x = _polyphase2x(x_pp)

    return x


@tf.function
def _filter_window_matrix(filters_n):
    """Produces a diamond shaped folding matrix F from the sine window which leads to identical analysis and
    synthesis base-band impulse responses. Hence has det 1 or -1.

    :param filters_n:   number of MDCT filters (needs to be even!)
    :return:            F of shape (filters_n, filters_n)
    """
    # Sine window:
    filter_bank_windows = tf.sin(math.pi / (2 * filters_n) * (tf.range(0.5, int(1.5 * filters_n) + 0.5)))

    # lace window coefficients around diamond matrix
    F_upper_left = tf.reverse(tf.linalg.diag(filter_bank_windows[0:int(filters_n / 2)]), axis=[1])
    F_lower_left = tf.linalg.diag(filter_bank_windows[int(filters_n / 2):filters_n])
    F_upper_right = tf.linalg.diag(filter_bank_windows[filters_n:(filters_n + int(filters_n / 2))])
    # F matrix is completed via consistency rule (hence no need for filter_bank_windows range to extend to 2filters_n-1
    sym = 1.0  # The kind of symmetry: +-1
    ff = tf.reverse((sym * tf.ones((int(filters_n / 2)))
                     - filter_bank_windows[filters_n:(int(1.5 * filters_n))] * filter_bank_windows[filters_n - 1:int(filters_n / 2) - 1:-1])
                    / filter_bank_windows[0:int(filters_n / 2)], axis=[0])
    # note:
    # ff entry i (i=0..filters_n/2) = (1 - sin(pi/(2filters_n)(filters_n+i+.5)) * sin(pi/(2filters_n)(filters_n-i-.5))) / sin(pi/(2filters_n)(i+.5))
    #    = sin(pi/(2filters_n) [2filters_n - i+.5])
    F_lower_right = -tf.reverse(tf.linalg.diag(ff), axis=[1])

    return tf.concat([tf.concat([F_upper_left, F_upper_right], axis=1),
                      tf.concat([F_lower_left, F_lower_right], axis=1)], axis=0)


@tf.function
def _delay_matrix(filters_n):
    """Delay matrix D(z), which has delay z^-1 on the upper half of its diagonal
    in a 3D polynomial representation (exponents of z^-1 are in the third dimension)

    :param filters_n:  number of MDCT filters (should be even!)
    :return:           delay matrix [2, filters_n, filters_n]
    """
    a = tf.linalg.diag(tf.concat([tf.zeros(int(filters_n / 2)), tf.ones(int(filters_n / 2))], axis=0))
    b = tf.linalg.diag(tf.concat([tf.ones(int(filters_n / 2)), tf.zeros(int(filters_n / 2))], axis=0))
    return tf.stack([a, b], axis=0)


@tf.function
def _inverse_delay_matrix(filters_n):
    """Causal inverse delay matrix D^{-1}(z), which has delays z^-1  on the lower
    half in 3D polynomial representation (exponents of z^-1 are in third dimension)

    :param filters_n:  number of MDCT filters (should be even!)
    :return:   inverse delay matrix [filters_n, 2, filters_n]
    """
    a = tf.linalg.diag(tf.concat([tf.ones(int(filters_n / 2)), tf.zeros(int(filters_n / 2))], axis=0))
    b = tf.linalg.diag(tf.concat([tf.zeros(int(filters_n / 2)), tf.ones(int(filters_n / 2))], axis=0))
    return tf.stack([a, b], axis=1)


@tf.function
def _x2polyphase(x, filters_n):
    """Split signal in each channel into blocks of size filters_n.
    Last part of signal is ignored, if it does not fit into a block.

      x ---+-----> down-sample by filters_n -->
           |z^-1
           +-----> down-sample by filters_n -->
           |z^-1
          ...
           |z^-1
           +-----> down-sample by filters_n -->

    Example:
        filters_n = 3
        in  = [[5,6,7,8,9,10],
               [...]]
        out = [[[5, 6, 7],
                [8, 9, 10]],
               [[...],
                [...]]]
            = [[[  5       6       7   ],
                   +       +       +
                [8z^-1   9z^-1  10z^-1]],
               [[...],
                [...]]]
           with z^-1 being the operator of a 1-block delay on the signal

    :param x:           multi-channel input signal [#channel, #samples]
    :param filters_n:   size of blocks
    :return:            multi-channel signal split in blocks [#channel, #blocks, filters_n]

    :raises InvalidArgumentError when #samples is not a multiple of filters_n
    """
    return tf.reshape(x, [tf.shape(x)[0], -1, filters_n])


@tf.function
def _polyphase2x(xp):
    """Glues back together the blocks (on axis=1)

    :param xp:  multi-channel signal split in blocks [#channel, #blocks, filters_n]
    :return:    multi-channel signal [#channel, #samples]
    """
    return tf.reshape(xp, [tf.shape(xp)[0], -1])


@tf.function
def _dct4(samples):
    """DCT4 transformation on axis=1 of samples

        y_k = \\sqrt{2/N} \\sum_{n=0}^{N-1} x_n \\cos( \\pi/N (n+1/2) (k+1/2) )

    Note: DCT4 is its own reverse

    :param samples: 3d array of samples already in shape [#channels, #blocks, filters_n]
                    axis=-1 are filter_n samples from the signal, on which the DCT4 is performed
    :return: 3-D array where axis=1 is DCT4-transformed, orthonormal with shape [#channel, #blocks, filters_n]
             axis=-1 thus contains the coefficients of the cosine harmonics which compose the filters_n block signal
    """
    # up-sample by inserting zeros for all even entries: this allows us to express DCT-IV as a DCT-III
    filters_n = tf.shape(samples)[-1]
    upsampled = tf.reshape(
                  tf.stack([tf.zeros(tf.shape(samples)), samples], axis=-1),
                  shape=[tf.shape(samples)[0], tf.shape(samples)[1], 2 * filters_n])

    y = math.sqrt(2) * tf.signal.dct(upsampled, type=3, axis=-1, norm='ortho')

    # down-sample again
    return y[:, :, 0:filters_n]


@tf.function
def _polmatmult(A, F):
    """Matrix multiplication of matrices where each entry is a polynomial.

    :param A: 3D matrix A, with 2nd dimension the polynomial coefficients
    :param F: 3D matrix F, with 1st dimension the polynomial coefficients
    :return:  3D matrix C, with 2nd dimension the polynomial coefficients
          C_{i n k} = \\sum_{m=0}^n    \\sum_j A{i m j}           F_{n-m jk}
                    = \\sum_{m=0}^n    \\sum_j A{i m j}           F_flip{f-n+m jk}   with f=tf.shape(F)[2]-1=degree(F)
                    = \\sum_{mm=f-n}^f \\sum_j A{i mm-f+n j}      F_flip{mm jk}      with mm=m+(f-n)
                    = \\sum {mm=f-n}^f \\sum_j A_padded{i mm+n j} F_flip{mm jk}      A padded with degree(F)
        where F plays role of filter (kernel) which is dragged over the padded A.
    """
    F_flip = tf.reverse(F, axis=[0])

    # add padding: degree(F) zeros at start and end along the 3rd (polynomial) dimension
    A_padded = tf.concat([tf.zeros([tf.shape(A)[0], tf.shape(F)[1]-1, tf.shape(A)[2]]),
                          A,
                          tf.zeros([tf.shape(A)[0], tf.shape(F)[1]-1, tf.shape(A)[2]])], axis=1)

    return tf.nn.convolution(A_padded, F_flip, padding="VALID")


@tf.function
def normalize_mdct(mdct_amplitudes):
    """With an audio signal in the -1..1 range, the mdct amplitudes is maximally of the order of \sqrt{2 filter_n}
    as can be seen from the dct4 formula

    .. math:
        mdct_{dB} = 20 \\log_{10} |mdct_{amplitude}| / 2^{-15}

    We can then normalize the mdct amplitudes in dB in the -1..1 range, by assuming they are
    within dB_MIN = -20 and dB_MAX = 160

    Normalize mdct amplitudes from -inf..inf range to -1..1 range
    Converts absolute value of amplitude to dB and maps dB_MIN..dB_MAX to 0..1
    Sign of amplitude is used as sign of normalized output.

    :param mdct_amplitudes: -inf..inf  [channels_n, filter_bands_n, #blocks]
    :return:                -1..1      [channels_n, filter_bands_n, #blocks]
    """
    # convert to dB, clip, normalize and add in sign, so -1 <= X_lmag <= 1
    mdct_db = 20. * tf.math.log(tf.abs(mdct_amplitudes) * (2**15) + _LOG_EPS) / tf.math.log(10.)
    tf.print(tf.reduce_max(mdct_db))
    tf.print(tf.reduce_min(mdct_db))
    mdct_clipped = tf.clip_by_value(mdct_db, _dB_MIN, _dB_MAX)
    mdct_norm = tf.sign(mdct_amplitudes) * (mdct_clipped - _dB_MIN) / (_dB_MAX - _dB_MIN)

    return mdct_norm


@tf.function
def inv_normalize_mdct(mdct_norm):
    """Convert normalized mdct amplitudes to actual amplitude value

    :param mdct_norm: -1..1      [channels_n, filter_bands_n, #blocks]
    :return:          -inf..inf  [channels_n, filter_bands_n, #blocks]
    """
    # remove dB rescaling
    mdct_db = (_dB_MAX - _dB_MIN) * tf.abs(mdct_norm) + _dB_MIN
    mdct_amplitudes = tf.sign(mdct_norm) * tf.exp(tf.math.log(10.) * (mdct_db/2**15) / 20.)

    return mdct_amplitudes
