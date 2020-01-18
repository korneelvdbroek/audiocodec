#!/usr/bin/env python

""" Implements a MDCT transformation and inverse transformation on (channel x signal data)

Loosely based based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
"""

import tensorflow as tf
import math


@tf.function
def setup(filters_n=1024, dB_max=100, window_type='vorbis'):
  """Computes required initialization matrices (stateless, no OOP)
  Note: H and H_inv are very sparse (2xfilter_n non-zero elements, arranged in diamond shape),
  yet TF2.0 does not support tf.nn.convolution for SparseTensor
  todo: work out polymatmul(x_pp, H) and polymatmul(y, H_inv) in more efficient way

  :param filters_n:   number of filter bands of the filter bank (needs to be even)
  :param window_type: None, 'sine' or 'vorbis' (default) to select window type
  :return:            tuple with pre-computed required for encoder and decoder
  """
  assert (filters_n % 2) == 0, "number of filters used in mdct transformation needs to be even"

  H = _polyphase_matrix(filters_n, window_type)
  H_inv = _inv_polyphase_matrix(filters_n, window_type)
  dB_factor = 10.**(dB_max / 20.)

  return filters_n, H, H_inv, dB_factor


def _polyphase_matrix(filters_n, window_type='vorbis'):
  """Decomposed part of poly-phase matrix of an MDCT filter bank, with a sine modulated window:
    H(z) = F_{analysis} x D x DCT4 =

  :param filters_n:  number of MDCT filters
  :return:           F_{analysis} x D           [filters_n, filters_n, 2]
  """
  with tf.name_scope('poly_phase_matrix'):
    F_analysis = tf.expand_dims(
      _filter_window_matrix(filters_n, window_type), axis=1)                # [filters_n, 1, filters_n]
    D = _delay_matrix(filters_n)                                              # [2, filters_n, filters_n]
    polyphase_matrix = _polymatmul(F_analysis, D)                             # [filters_n, 2, filters_n]

  return tf.transpose(polyphase_matrix, perm=[1, 0, 2])                         # [2, filters_n, filters_n]


def _inv_polyphase_matrix(filters_n, window_type='vorbis'):
  """Decomposed part of inverse poly-phase matrix of an MDCT filter bank, with a sine modulated window:
    G(z) = DCT4 x D^-1 x F_{synthesis}

  :param filters_n:   number of MDCT filters
  :return:            D^-1 x F_{synthesis}       [2, filters_n, filters_n]
  """
  with tf.name_scope('inv_poly_phase_matrix'):
    # invert Fa matrix for synthesis after removing last dim:
    F_synthesis = tf.expand_dims(
      tf.linalg.inv(
        _filter_window_matrix(filters_n, window_type)), axis=0)   # [1, filters_n, filters_n]
    D_inv = _inverse_delay_matrix(filters_n)                                        # [filters_n, 2, filters_n]
    inv_polyphase_matrix = _polymatmul(D_inv, F_synthesis)                          # [filters_n, 2, filters_n]

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
  Input has #samples, output has same amount of data (#samples = #blocks x filters_n).

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

  Return amplitude scaling:
  The dct4() amplitudes y_k are maximally of the order of
     x_{max} \\sqrt{2 filter_n}
  as can be seen from the dct4 formula above, with x_{max} \approx 1.
  These dct4() amplitudes are then rescaled by
     10^{_dB_MAX / 20.} / \\sqrt{2 filter_n}
  so the maximum output amplitudes are of the order of
     10^{_dB_MAX / 20.}

  :param x:            signal data assumed in -1..1 range [channels, #samples]
  :param model_init:   initialization data for the mdct model
  :return:             filters_n coefficients of MDCT transform for each block [#channels, #blocks+1, filters_n]
                       where #samples = #blocks x filters_n
                       amplitudes are rescaled such that largest possible amplitude is 10^{_dB_MAX / 20.}
  """
  filters_n, H, _, dB_factor = model_init
  with tf.name_scope('mdct_transform'):
    # split signal into blocks
    x_pp = _x2polyphase(x, filters_n)               # [#channels, #blocks, filters_n]

    # put x through filter bank
    mdct_amplitudes = _dct4(_polymatmul(x_pp, H), filters_n)   # [#channels, #blocks+1, filters_n]

  # up-scale
  return (dB_factor / tf.sqrt(2. * tf.cast(filters_n, dtype=tf.float32))) * mdct_amplitudes


@tf.function
def inverse_transform(mdct_amplitudes, model_init):
  """MDCT synthesis filter bank.

  :param mdct_amplitudes: mdct amplitudes with shape       [#channels, #blocks, filters_n]
                          amplitudes should be scaled such that maximum possible amplitude has size 10^{_dB_MAX / 20.}
  :param model_init:      initialization data for the mdct model
  :return:                restored signal in range -1..1   [#channels, #samples]
                          where #samples = (#blocks + 1) x filters_n
  """
  filters_n, _, H_inv, dB_factor = model_init
  with tf.name_scope('inv_mdct_transform'):
    mdct_rescaled = (tf.sqrt(2. * tf.cast(filters_n, dtype=tf.float32)) / dB_factor) * mdct_amplitudes

    # put y through inverse filter bank
    x_pp = _polymatmul(_dct4(mdct_rescaled, filters_n), H_inv)

    # glue back the blocks to one signal
    x = _polyphase2x(x_pp)

  return x


def _filter_window_matrix(filters_n, window_type="vorbis"):
  """Produces a diamond shaped folding matrix F from the sine window which leads to identical analysis and
  synthesis base-band impulse responses. Hence has det 1 or -1.

  :param filters_n:   number of MDCT filters (needs to be even!)
  :param window_type: None, 'sine' or 'vorbis' (default) to select window type
  :return:            F of shape (filters_n, filters_n)
  """
  if window_type.lower() == 'sine':
    # Sine window:
    filter_bank_windows = tf.sin(math.pi / (2 * filters_n) * (tf.range(0.5, int(1.5 * filters_n) + 0.5)))
  elif window_type.lower() == 'vorbis':
    filter_bank_windows = tf.sin(
      math.pi / 2. * tf.sin(math.pi / (2. * filters_n) * tf.range(0.5, int(1.5 * filters_n) + 0.5)) ** 2)
  else:
    # no modified window (issues with stopband attenuation)
    filter_bank_windows = tf.ones(shape=[filters_n + int(filters_n / 2)])

  # lace window coefficients around diamond matrix
  F_upper_left = tf.reverse(tf.linalg.diag(filter_bank_windows[0:int(filters_n / 2)]), axis=[1])
  F_lower_left = tf.linalg.diag(filter_bank_windows[int(filters_n / 2):filters_n])
  F_upper_right = tf.linalg.diag(filter_bank_windows[filters_n:(filters_n + int(filters_n / 2))])
  # F matrix is completed via consistency rule (hence no need for filter_bank_windows range to extend to 2filters_n-1
  sym = 1.0  # The kind of symmetry: +-1
  ff = tf.reverse((sym * tf.ones((int(filters_n / 2)))
                   - filter_bank_windows[filters_n:(int(1.5 * filters_n))] * filter_bank_windows[filters_n - 1:int(filters_n / 2) - 1:-1])
                  / filter_bank_windows[0:int(filters_n / 2)], axis=[0])
  # note for sine window:
  # ff entry i (i=0..filters_n/2)
  #    = (1-sin(pi/(2filters_n)(filters_n+i+.5)) * sin(pi/(2filters_n)(filters_n-i-.5))) / sin(pi/(2filters_n)(i+.5))
  #    = sin(pi/(2filters_n) [2filters_n - i+.5])
  F_lower_right = -tf.reverse(tf.linalg.diag(ff), axis=[1])

  return tf.concat([tf.concat([F_upper_left, F_upper_right], axis=1),
                    tf.concat([F_lower_left, F_lower_right], axis=1)], axis=0)


def _delay_matrix(filters_n):
  """Delay matrix D(z), which has delay z^-1 on the upper half of its diagonal
  in a 3D polynomial representation (exponents of z^-1 are in the third dimension)

  :param filters_n:  number of MDCT filters (should be even!)
  :return:           delay matrix [2, filters_n, filters_n]
  """
  a = tf.linalg.diag(tf.concat([tf.zeros(int(filters_n / 2)), tf.ones(int(filters_n / 2))], axis=0))
  b = tf.linalg.diag(tf.concat([tf.ones(int(filters_n / 2)), tf.zeros(int(filters_n / 2))], axis=0))
  return tf.stack([a, b], axis=0)


def _inverse_delay_matrix(filters_n):
  """Causal inverse delay matrix D^{-1}(z), which has delays z^-1  on the lower
  half in 3D polynomial representation (exponents of z^-1 are in third dimension)

  :param filters_n:  number of MDCT filters (should be even!)
  :return:   inverse delay matrix [filters_n, 2, filters_n]
  """
  a = tf.linalg.diag(tf.concat([tf.ones(int(filters_n / 2)), tf.zeros(int(filters_n / 2))], axis=0))
  b = tf.linalg.diag(tf.concat([tf.zeros(int(filters_n / 2)), tf.ones(int(filters_n / 2))], axis=0))
  return tf.stack([a, b], axis=1)


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


def _polyphase2x(xp):
  """Glues back together the blocks (on axis=1)

  :param xp:  multi-channel signal split in blocks [#channel, #blocks, filters_n]
  :return:    multi-channel signal [#channel, #samples]
  """
  return tf.reshape(xp, [tf.shape(xp)[0], -1])


def _dct4(samples, filters_n):
  """Orthogonal DCT4 transformation on axis=1 of samples

      y_k = \\sqrt{2/N} \\sum_{n=0}^{N-1} x_n \\cos( \\pi/N (n+1/2) (k+1/2) )

  The dct4() amplitudes y_k are maximally of the order of
     x_{max} \\sqrt{2 filter_n}
  as can be seen from the dct4 formula above.
  Note: DCT4 is its own reverse

  :param samples:     3d array of samples already in shape [#channels, #blocks, filters_n]
                      axis=-1 are filter_n samples from the signal, on which the DCT4 is performed
  :return:            3-D array where axis=1 is DCT4-transformed, orthonormal with shape [#channel, #blocks, filters_n]
                      axis=2 contains coefficients of the cosine harmonics which compose the filters_n block signal
                      amplitudes are rescaled, such that largest possible amplitude has dB_MAX decibels
  """
  # up-sample by inserting zeros for all even entries: this allows us to express DCT-IV as a DCT-III
  upsampled = tf.reshape(
    tf.stack([tf.zeros(tf.shape(samples)), samples], axis=-1),
    shape=[tf.shape(samples)[0], tf.shape(samples)[1], 2 * filters_n])

  # conversion factor \sqrt{2} is needed to go from orthogonal DCT-III to orthogonal DCT-IV,
  y = math.sqrt(2.) * tf.signal.dct(upsampled, type=3, axis=-1, norm='ortho')

  # down-sample
  return y[:, :, 0:filters_n]


@tf.function
def _polymatmul(A, F):
  """Matrix multiplication of matrices where each entry is a polynomial.

  :param A: 3D matrix A, with 2nd dimension the polynomial coefficients
  :param F: 3D matrix F, with 1st dimension the polynomial coefficients
  :return:  3D matrix C, with 2nd dimension the polynomial coefficients
        C_{i n k} = \\sum_{m=0}^n    \\sum_j A{i m j}           F_{n-m jk}
                  = \\sum_{m=0}^n    \\sum_j A{i m j}           F_flip{f-n+m jk}   with f=tf.shape(F)[2]-1=degree(F)
                  = \\sum_{mm=f-n}^f \\sum_j A{i mm-f+n j}      F_flip{mm jk}      with mm=m+(f-n)
                  = \\sum {mm=f-n}^f \\sum_j A_padded{i mm+n j} F_flip{mm jk}      A padded with degree(F)
                  = tf.nn.convolution(A_padded, F_flip)
      where F plays role of filter (kernel) which is dragged over the padded A.
  """
  F_flip = tf.reverse(F, axis=[0])

  # add padding: degree(F) zeros at start and end along A's polynomial dimension
  F_degree = tf.shape(F)[0] - 1
  A_padded = tf.pad(A, paddings=tf.convert_to_tensor([[0, 0], [F_degree, F_degree], [0, 0]]))

  return tf.nn.convolution(A_padded, F_flip, padding="VALID")

