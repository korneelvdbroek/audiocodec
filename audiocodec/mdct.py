#!/usr/bin/env python

""" Implements a MDCT transformation and inverse transformation on (channel x signal data)

Based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
"""

import numpy as np
import scipy.fftpack as fft


def polyphase_matrix(N):
    """Decomposed part of poly-phase matrix of an MDCT filter bank, with a sine modulated window:
      H(z) = F_{analysis} x D x DCT4

    :param N: number of MDCT filters
    :return:  F_{analysis} x D           (N x N x 2)
    """
    F_analysis = np.expand_dims(_filter_window_matrix(N), axis=-1)    # N x N x 1
    D = _delay_matrix(N)                                              # N x N x 2

    return _polmatmult(F_analysis, D)                                 # N x N x 2


def inverse_polyphase_matrix(N):
    """Decomposed part of inverse poly-phase matrix of an MDCT filter bank, with a sine modulated window:
      G(z) = DCT4 x D^-1 x F_{synthesis}

    :param N: number of MDCT filters
    :return:  D^-1 x F_{synthesis}       (N x N x 2)
    """
    # invert Fa matrix for synthesis after removing last dim:
    F_synthesis = np.expand_dims(np.linalg.inv(_filter_window_matrix(N)), axis=-1)  # N x N x 1
    Dinv = _inverse_delay_matrix(N)                                                 # N x N x 2

    return _polmatmult(Dinv, F_synthesis)  # N x N x 2


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
    N = H.shape[0]

    # split signal into blocks
    x_pp = _x2polyphase(x, N)           # #channels x N x #blocks

    # put x through filter bank
    return _dct4(_polmatmult(x_pp, H))  # #channels x N x #blocks+1


def inverse_transform(y, Hinv):
    """MDCT synthesis filter bank.

    :param y:     encoded signal in 3d array (#channels x N x #blocks+1)
    :param Hinv:  inverse poly-phase matrix (N x N x 2)
    :return:      restored signal (#channels x #samples)
    """
    # put y through inverse filter bank
    x_pp = _polmatmult(_dct4(y), Hinv)

    # glue back the blocks to one signal
    return _polyphase2x(x_pp)


def _filter_window_matrix(N):
    """Produces a diamond shaped folding matrix F from the sine window which leads to identical analysis and
    synthesis base-band impulse responses. Hence has det 1 or -1.

    :param N: number of MDCT filters (needs to be even!)
    :return:  F of shape (N, N)
    """
    # Sine window:
    filter_bank_windows = np.sin(np.pi / (2 * N) * (np.arange(int(1.5 * N)) + 0.5))

    # lace window coefficients around diamond matrix
    F = np.zeros((N, N))
    F[0:int(N / 2), 0:int(N / 2)] = np.fliplr(np.diag(filter_bank_windows[0:int(N / 2)]))
    F[int(N / 2):N, 0:int(N / 2)] = np.diag(filter_bank_windows[int(N / 2):N])
    F[0:int(N / 2), int(N / 2):N] = np.diag(filter_bank_windows[N:(N + int(N / 2))])
    # F matrix is completed via consistency rule (hence no need for filter_bank_windows range to extend to 2N-1
    sym = 1.0  # The kind of symmetry: +-1
    ff = np.flipud((sym * np.ones((int(N / 2)))
                    - filter_bank_windows[N:(int(1.5 * N))] * filter_bank_windows[N - 1:int(N / 2) - 1:-1])
                    / filter_bank_windows[0:int(N / 2)])
    # note:
    # ff entry i (i=0..N/2) = (1 - sin(pi/(2N)(N+i+.5)) * sin(pi/(2N)(N-i-.5))) / sin(pi/(2N)(i+.5))
    #    = sin(pi/(2N) [2N - i+.5])
    F[int(N / 2):N, int(N / 2):N] = -np.fliplr(np.diag(ff))

    return F


def _delay_matrix(N):
    """Delay matrix D(z), which has delay z^-1 on the upper half of its diagonal
    in a 3D polynomial representation (exponents of z^-1 are in the third dimension)

    :param N:  number of MDCT filters (should be even!)
    :return:   delay matrix (N x N x 2)
    """
    D = np.zeros((N, N, 2))
    D[:, :, 0] = np.diag(np.append(np.zeros((1, int(N / 2))), np.ones((1, int(N / 2)))))
    D[:, :, 1] = np.diag(np.append(np.ones((1, int(N / 2))), np.zeros((1, int(N / 2)))))
    return D


def _inverse_delay_matrix(N):
    """Causal inverse delay matrix D^{-1}(z), which has delays z^-1  on the lower
    half in 3D polynomial representation (exponents of z^-1 are in third dimension)

    :param N:  number of MDCT filters (should be even!)
    :return:   inverse delay matrix (N x N x 2)
    """
    D = np.zeros((N, N, 2))
    D[:, :, 0] = np.diag((np.append(np.ones((1, int(N / 2))), np.zeros((1, int(N / 2))))))
    D[:, :, 1] = np.diag((np.append(np.zeros((1, int(N / 2))), np.ones((1, int(N / 2))))))
    return D


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
    x = x[:, :int(x.shape[1] / N) * N]

    return np.reshape(x, (x.shape[0], N, -1), order='F')  # order=F: first index changes fastest


def _polyphase2x(xp):
    """Glues back together the blocks (on axis=1)

    :param xp:  multi-channel signal split in blocks (#channel x N x #blocks)
    :return:    multi-channel signal (#channel x #samples)
    """
    x = np.reshape(xp, (xp.shape[0], 1, -1), order='F')  # order=F: first index changes fastest
    x = np.squeeze(x, axis=1)
    return x


def _dct4(samples):
    """DCT4 transformation on axis=1 of samples
    Note: DCT4 is its own reverse

    :param samples: 3d array of samples (#channels x N x #blocks)
                    Axis=1 is a block, on which the DCT4 is performed
    :return: 3-D array where axis=1 is DCT4-transformed, orthonormal with shape (#channel, N, #blocks)
             Axis=1 thus contains the coefficients of the cos harmonics which compose the N block signal
    """
    channels, N, blocks = samples.shape

    # up-sample to use the DCT3 implementation for our DCT4 transformation:
    upsampled = np.zeros((channels, 2 * N, blocks))
    upsampled[:, 1::2, :] = samples  # each time add in a zero for each element in the row
    y = fft.dct(upsampled, type=3, axis=1, norm='ortho') * np.sqrt(2)

    # down-sample again
    return y[:, 0:N, :]


def _polmatmult(A, B):
    """Matrix multiplication of matrices where each entry is a polynomial. Polynomial coefficient are on axis=2.

    :param A: 3d matrix A
    :param B: 3d matrix B
    :return:  A*B
    """
    A_degree = A.shape[2] - 1    # A_degree = 2, if A contains z^-2
    B_degree = B.shape[2] - 1

    C_degree = A_degree + B_degree

    C = np.zeros((A.shape[0], B.shape[1], C_degree + 1))
    # Convolution of matrices (can be written as a flipped trace, though that is not faster)
    #   Hflip = np.flip(H, axis=2)
    #   big = np.einsum("ijm,jkn->ikmn", a, Hflip)
    #   for n in range(C_degree + 1):
    #       C[:, :, n] = np.trace(big, offset=H_degree - n, axis1=2, axis2=3)
    for n in range(C_degree + 1):
        for m in range(n + 1):
            if (n - m) <= A_degree and m <= B_degree:
                C[:, :, n] = C[:, :, n] + np.dot(A[:, :, (n - m)], B[:, :, m])
    # \Sum_m=0^n \Sum_j A_{ij(n-m)} B_{jkm}
    return C
