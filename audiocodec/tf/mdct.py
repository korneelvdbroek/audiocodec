#!/usr/bin/env python

""" Implements a MDCT transformation and inverse transformation on (channel x signal data)

Loosely based based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
"""

import tensorflow as tf
import math


def setup(N=1024):
    """Computes required initialization matrices (stateless, no OOP)

    :param N:    number of filter bands of the filter bank
    :return:     tuple with pre-computed required for encoder and decoder
    """
    H = polyphase_matrix(N)
    H_inv = inverse_polyphase_matrix(N)

    return N, H, H_inv


def polyphase_matrix(N):
    """Decomposed part of poly-phase matrix of an MDCT filter bank, with a sine modulated window:
      H(z) = F_{analysis} x D x DCT4

    :param N: number of MDCT filters
    :return:  F_{analysis} x D           (N x N x 2)
    """
    with tf.name_scope('poly_phase_matrix'):
      with tf.name_scope('F_analysis'):
          F_analysis = tf.expand_dims(_filter_window_matrix(N), axis=-1)    # N x N x 1
      with tf.name_scope('delay_matrix'):
          D = _delay_matrix(N)                                              # N x N x 2

      with tf.name_scope('polyphase_matrix'):
          polyphase_matrix = _polmatmult(F_analysis, D)                     # N x N x 2

    return polyphase_matrix


def inverse_polyphase_matrix(N):
    """Decomposed part of inverse poly-phase matrix of an MDCT filter bank, with a sine modulated window:
      G(z) = DCT4 x D^-1 x F_{synthesis}

    :param N: number of MDCT filters
    :return:  D^-1 x F_{synthesis}       (N x N x 2)
    """
    with tf.name_scope('inv_poly_phase_matrix'):
        with tf.name_scope('F_synthesis'):
            # invert Fa matrix for synthesis after removing last dim:
            F_synthesis = tf.expand_dims(tf.linalg.inv(_filter_window_matrix(N)), axis=-1)  # N x N x 1
        with tf.name_scope('inv_delay_matrix'):
            Dinv = _inverse_delay_matrix(N)                                                 # N x N x 2
        with tf.name_scope('inv_polyphase_matrix'):
            inv_polyphase_matrix = _polmatmult(Dinv, F_synthesis)                           # N x N x 2

    return inv_polyphase_matrix


def transform(x, H):
    """MDCT (Modulated Discrete Cosine Transform) analysis filter bank. Filters in MDCT use window-modulated filters.

    Basic structure:

      audio --+---> filter 1 --> down-sample by N -->
              |
              +---> filter 2 --> down-sample by N -->
              |
             ...
              |
              +---> filter N --> down-sample by N -->

    Down-sampling of the output of the N filters by factor N is without loss of information (Nyquist Theorem).
    Input has #samples, output has same amount of data.

    To improve computational efficiency, order of filter and down-sampling can be switched around (Noble Identities).
    Each filter itself needs to be split into its N poly-phase decompositions (see p25 ./docs/mrate.pdf):

      x ---+-----> down-sample by N --> [[     ]  [     ]     [     ]]
           |z^-1                        [[ fil ]  [ fil ]     [ fil ]]
           +-----> down-sample by N --> [[ ter ]  [ ter ] ... [ ter ]]
           |z^-1                        [[  1  ]  [  2  ]     [  N  ]] = H(z) = F_{analysis} x D x DCT4
          ...                           [[     ]  [     ]     [     ]]
           |z^-1                        [[     ]  [     ]     [     ]]
           +-----> down-sample by N --> [[     ]  [     ]     [     ]]

    with F_{analysis}: filter window matrix
         D:            delay matrix
         DCT4:         discrete cosine transformation matrix

    read more:
      ./docs/02_shl_Filterbanks1_NobleID_WS2016-17_o.pdf
      ./docs/03_shl_FilterBanks2_WS2016-17.pdf
      ./docs/mrate.pdf

    :param x: signal wave data: each row is a channel (#channels x #samples)
    :param H: poly-phase matrix (N x N x 2)
    :return:  N coefficients of MDCT transform for each block (#channels, N, #blocks+1)
    """
    with tf.name_scope('mdct_transform'):
        N = tf.dtypes.cast(tf.shape(H)[0], dtype=tf.int32)

        # split signal into blocks
        x_pp = _x2polyphase(x, N)               # #channels x N x #blocks

        # put x through filter bank
        mdct_amplitudes = _dct4(_polmatmult(x_pp, H))   # #channels x N x #blocks+1

    return mdct_amplitudes


def inverse_transform(y, Hinv):
    """MDCT synthesis filter bank.

    :param y:     encoded signal in 3d array (#channels x N x #blocks+1)
    :param Hinv:  inverse poly-phase matrix (N x N x 2)
    :return:      restored signal (#channels x #samples)
    """
    with tf.name_scope('inv_mdct_transform'):
        # put y through inverse filter bank
        x_pp = _polmatmult(_dct4(y), Hinv)

        # glue back the blocks to one signal
        x = _polyphase2x(x_pp)

    return x


def _filter_window_matrix(N):
    """Produces a diamond shaped folding matrix F from the sine window which leads to identical analysis and
    synthesis base-band impulse responses. Hence has det 1 or -1.

    :param N: number of MDCT filters (needs to be even!)
    :return:  F of shape (N, N)
    """
    # Sine window:
    filter_bank_windows = tf.sin(math.pi / (2 * N) * (tf.range(0.5, int(1.5 * N) + 0.5)))

    # lace window coefficients around diamond matrix
    F_upper_left = tf.reverse(tf.diag(filter_bank_windows[0:int(N / 2)]), axis=[1])
    F_lower_left = tf.diag(filter_bank_windows[int(N / 2):N])
    F_upper_right = tf.diag(filter_bank_windows[N:(N + int(N / 2))])
    # F matrix is completed via consistency rule (hence no need for filter_bank_windows range to extend to 2N-1
    sym = 1.0  # The kind of symmetry: +-1
    ff = tf.reverse((sym * tf.ones((int(N / 2)))
                     - filter_bank_windows[N:(int(1.5 * N))] * filter_bank_windows[N - 1:int(N / 2) - 1:-1])
                     / filter_bank_windows[0:int(N / 2)], axis=[0])
    # note:
    # ff entry i (i=0..N/2) = (1 - sin(pi/(2N)(N+i+.5)) * sin(pi/(2N)(N-i-.5))) / sin(pi/(2N)(i+.5))
    #    = sin(pi/(2N) [2N - i+.5])
    F_lower_right = -tf.reverse(tf.diag(ff), axis=[1])

    return tf.concat([tf.concat([F_upper_left, F_upper_right], axis=1),
                      tf.concat([F_lower_left, F_lower_right], axis=1)], axis=0)


def _delay_matrix(N):
    """Delay matrix D(z), which has delay z^-1 on the upper half of its diagonal
    in a 3D polynomial representation (exponents of z^-1 are in the third dimension)

    :param N:  number of MDCT filters (should be even!)
    :return:   delay matrix (N x N x 2)
    """
    a = tf.diag(tf.concat([tf.zeros(int(N / 2)), tf.ones(int(N / 2))], axis=0))
    b = tf.diag(tf.concat([tf.ones(int(N / 2)), tf.zeros(int(N / 2))], axis=0))
    return tf.stack([a, b], axis=-1)


def _inverse_delay_matrix(N):
    """Causal inverse delay matrix D^{-1}(z), which has delays z^-1  on the lower
    half in 3D polynomial representation (exponents of z^-1 are in third dimension)

    :param N:  number of MDCT filters (should be even!)
    :return:   inverse delay matrix (N x N x 2)
    """
    a = tf.diag(tf.concat([tf.ones(int(N / 2)), tf.zeros(int(N / 2))], axis=0))
    b = tf.diag(tf.concat([tf.zeros(int(N / 2)), tf.ones(int(N / 2))], axis=0))
    return tf.stack([a, b], axis=-1)


def _x2polyphase(x, N):
    """Split signal in each channel into blocks of size N.
    Last part of signal is ignored, if it does not fit into a block.

      x ---+-----> down-sample by N -->
           |z^-1
           +-----> down-sample by N -->
           |z^-1
          ...
           |z^-1
           +-----> down-sample by N -->

    Example:
        N = 3
        in  = [[5,6,7,8,9,10,11],
               [...]]
        out = [[[5, 8],
                [6, 9],
                [7, 10]],
               [[...]]]
            = [[5+8z^-1
                6+9z^-1
                7+10z^-1],
               [...]]
           with z^-1 being the operator of a 1-block delay on the signal

    :param x: multi-channel input signal (#channel x #samples)
    :param N: size of blocks
    :return:  multi-channel signal split in blocks (#channel x N x #blocks)
    """
    # truncate x: limit #samples so it can be split into blocks of size N
    blocks_n = tf.dtypes.cast(tf.math.floor(tf.shape(x)[1] / N), dtype=tf.int32)

    x_trunc = x[:, :blocks_n * N]

    x_polyphase = tf.transpose(tf.reshape(x_trunc, [tf.shape(x_trunc)[0], -1, N]), perm=[0, 2, 1])

    return x_polyphase


def _polyphase2x(xp):
    """Glues back together the blocks (on axis=1)

    :param xp:  multi-channel signal split in blocks (#channel x N x #blocks)
    :return:    multi-channel signal (#channel x #samples)
    """
    return tf.reshape(tf.transpose(xp, perm=[0, 2, 1]), [tf.shape(xp)[0], -1])


def _dct4(samples):
    """DCT4 transformation on axis=1 of samples
    Note: DCT4 is its own reverse

    :param samples: 3d array of samples (#channels x N x #blocks)
                    Axis=1 is a block, on which the DCT4 is performed
    :return: 3-D array where axis=1 is DCT4-transformed, orthonormal with shape (#channel, N, #blocks)
             Axis=1 thus contains the coefficients of the cos harmonics which compose the N block signal
    """
    N = tf.shape(samples)[1]

    # up-sample to use the DCT3 implementation for our DCT4 transformation:
    upsampled = tf.reshape(tf.stack([tf.zeros(tf.shape(samples)), samples], axis=2),
                           shape=[tf.shape(samples)[0], 2 * N, tf.shape(samples)[2]])

    y = tf.transpose(
          tf.signal.dct(
            tf.transpose(
              tf.dtypes.cast(upsampled, dtype=tf.float32),
              perm=[0, 2, 1]),
            type=3, axis=-1, norm='ortho') * math.sqrt(2),
          perm=[0, 2, 1])

    # down-sample again
    return y[:, 0:N, :]


def _polmatmult(A, B):
    """Matrix multiplication of matrices where each entry is a polynomial. Polynomial coefficient are on axis=2.

    :param A: 3D matrix A, with third dimension is coefficient of polynomial
    :param B: 3D matrix B, with third dimension is coefficient of polynomial
    :return:  C_ijn = \Sum_{m=0}^n \Sum_j A_{ij n-m} B{jk m}
              A multiplied by B in first 2 indices, with convolution (ie. polynomial multiplication) over 3rd dimension
    """
    B_flip = tf.reverse(B, axis=[2])

    # add padding zeros at beginning of signal
    A_padded = tf.concat([tf.zeros([tf.shape(A)[0], tf.shape(A)[1], 1]), A], -1)

    # position all indices in right position and then let tf do its magic in cuda
    C = tf.nn.convolution(tf.transpose(A_padded, perm=[0, 2, 1]), tf.transpose(B_flip, perm=[2, 0, 1]),
                          padding="SAME")

    return tf.transpose(C, perm=[0, 2, 1])
