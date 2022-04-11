#!/usr/bin/env python

"""Implements a MDCT transformation and inverse

Loosely based based on code from Gerald Schuller, June 2018 (https://github.com/TUIlmenauAMS/Python-Audio-Coder)
"""

import tensorflow as tf
import math


class MDCTransformer:
  def __init__(self, filters_n=1024, window_type='vorbis',
               compute_dtype=tf.float32, precompute_dtype=tf.float64):
    """Computes required initialization matrices
    Note: H and H_inv are very sparse (2xfilter_n non-zero elements, arranged in diamond shape),
    yet TF2.0 does not support tf.nn.convolution for SparseTensor
    todo: work out polymatmul(x_pp, H) and polymatmul(y, H_inv) in more efficient way

    :param filters_n:        number of filter bands of the filter bank (needs to be even)
    :param window_type:      None, 'sine' or 'vorbis' (default) to select window type
    :param compute_dtype     dtype of input and output of the layer. Defaults to tf.float32
                             Inputs dtype must match compute_dtype
    :return:                 tuple with pre-computed required for encoder and decoder
    """
    assert (filters_n % 2) == 0, "number of filters used in mdct transformation needs to be even"

    self.filters_n = filters_n
    self.window_type = window_type

    # Pre-compute some (kernel) matrices with higher precision (precompute_dtype) then down-cast to compute_dtype.
    # This is similar to the Variable treatment in a custom layer implementation which is mixed_precision capable.
    # The variables are stored in variable_dtype, which corresponds to our precompute_dtype.
    # Variables then are down-cast before they enter the calculation with input data
    #   see https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s91029-automated-mixed-precision-tools-for-tensorflow-training-v2.pdf

    # example of self.H for filters_n = 8
    # self.H[0, :, :] =
    #   [[ 0.          0.          0.          0.          0.99988616  0.          0.          0.        ]
    #    [ 0.          0.          0.          0.          0.          0.9912527   0.          0.        ]
    #    [ 0.          0.          0.          0.          0.          0.          0.93969655  0.        ]
    #    [ 0.          0.          0.          0.          0.          0.          0.          0.80674446]
    #    [ 0.          0.          0.          0.          0.          0.          0.         -0.5909005 ]
    #    [ 0.          0.          0.          0.          0.          0.         -0.3420093   0.        ]
    #    [ 0.          0.          0.          0.          0.         -0.13197729  0.          0.        ]
    #    [ 0.          0.          0.          0.         -0.01509063  0.          0.          0.        ]]
    #
    # self.H[1, :, :] =
    #   [[0.         0.         0.         0.01509063 0.         0.         0.         0.        ]
    #    [0.         0.         0.13197729 0.         0.         0.         0.         0.        ]
    #    [0.         0.3420093  0.         0.         0.         0.         0.         0.        ]
    #    [0.5909005  0.         0.         0.         0.         0.         0.         0.        ]
    #    [0.80674446 0.         0.         0.         0.         0.         0.         0.        ]
    #    [0.         0.93969655 0.         0.         0.         0.         0.         0.        ]
    #    [0.         0.         0.9912527  0.         0.         0.         0.         0.        ]
    #    [0.         0.         0.         0.99988616 0.         0.         0.         0.        ]]

    self.H = tf.cast(self._polyphase_matrix(precompute_dtype=precompute_dtype), dtype=compute_dtype)
    self.H_inv = tf.cast(self._inv_polyphase_matrix(precompute_dtype=precompute_dtype), dtype=compute_dtype)

  @tf.function
  def transform(self, x):
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
    Each filter itself needs to be split into its filters_n poly-phase decompositions:

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

    Return amplitude scaling:
    The dct4() amplitudes y_k are maximally of the order of
       x_{max} \\sqrt{2 filter_n}
    as can be seen from the dct4 formula, with x_{max} \approx 1.
    The poly-phase matrix H adds another factor of \\sqrt{2}
    These dct4() amplitudes are then rescaled by
       1. / (\\sqrt{2} \\sqrt{2 filter_n})
    so the maximum output amplitudes are of the order of
       1.

    :param x:                 signal data assumed in -1..1 range [batches_n, samples_n, channels_n]
                              must be of compute_dtype
    :return:                  filters_n coefficients of MDCT transform for each block
                              output shape is [batches_n, blocks_n+1, filter_bands_n, channels_n]
                              where samples_n = blocks_n x filter_bands_n
                              amplitudes are normalized to be in the ]-1, 1[ range
    """
    batches_n = tf.shape(x)[0]
    channels_n = x.shape[2]

    # split signal into blocks
    x_pp = self._split_in_blocks(x)                                    # [batches_n x channels_n, blocks_n, filter_bands_n]

    # put x_pp through filter bank
    mdct_amplitudes = self._dct4(self._polymatmul(x_pp, self.H))       # [batches_n x channels_n, blocks_n+1, filter_bands_n]

    # un-fold channels dimension
    mdct_amplitudes = tf.reshape(mdct_amplitudes, shape=[batches_n, channels_n, tf.shape(mdct_amplitudes)[1], tf.shape(mdct_amplitudes)[2]])
    mdct_amplitudes = tf.transpose(mdct_amplitudes, perm=[0, 2, 3, 1]) # [batches_n, blocks_n+1, filter_bands_n, channels_n]

    # up-scale
    return (1. / tf.sqrt(4. * tf.cast(self.filters_n, dtype=x.dtype))) * mdct_amplitudes

  @tf.function
  def inverse_transform(self, mdct_amplitudes):
    """MDCT synthesis filter bank.

    :param mdct_amplitudes:   mdct amplitudes with shape       [batches_n, blocks_n, filter_bands_n, channels_n]
                              amplitudes should be in -1..1 range
                              must be of compute_dtype
    :return:                  restored signal in range -1..1   [batches_n, samples_n, channels_n]
                              where samples_n = (blocks_n + 1) x filters_n
    """
    # fold channels dimension into batches
    batches_n = tf.shape(mdct_amplitudes)[0]
    channels_n = tf.shape(mdct_amplitudes)[3]

    mdct_amplitudes = tf.transpose(mdct_amplitudes, perm=[0, 3, 1, 2])
    mdct_amplitudes = tf.reshape(mdct_amplitudes, shape=[-1, mdct_amplitudes.shape[2], mdct_amplitudes.shape[3]])
    # [batches_n x channels_n, blocks_n, filter_bands_n]

    mdct_rescaled = tf.sqrt(4. * tf.cast(self.filters_n, dtype=mdct_amplitudes.dtype)) * mdct_amplitudes

    # put y through inverse filter bank
    x_pp = self._polymatmul(self._dct4(mdct_rescaled), self.H_inv)

    # glue back the blocks to one signal
    x = self._merge_blocks(x_pp, batches_n, channels_n)

    return x

  def _polyphase_matrix(self, precompute_dtype):
    """Decomposed part of poly-phase matrix of an MDCT filter bank, with a sine modulated window:
      H(z) = (F_{analysis} x D) x DCT4

      The window function needs to satisfy:
      1. w_n = w_{2N-1-n}        (symmetry)
      2. w_n^2 + w_{n+N}^2 = 1   (Princen-Bradley)
      So, effectively, one can compute that the transformation of the poly-phase matrix weights the input by
         w_{i+N} x_i + \\sqrt{1-w_{i+N}^2} x_{N-i}
      if input x's are maximal (1), then we need to weight with \\sqrt{2}

    :param precompute_dtype     The type of the elements of the resulting tensor.
    :return:                  F_{analysis} x D           [filters_n, filters_n, 2]
    """
    F_analysis = tf.expand_dims(
      self._filter_window_matrix(precompute_dtype), axis=1)             # [filters_n, 1 (=blocks_n), filters_n]
    D = self._delay_matrix(precompute_dtype)                            # [2 (=blocks_n), filters_n, filters_n]
    polyphase_matrix = self._polymatmul(F_analysis, D)                # [filters_n, 2 (=blocks_n), filters_n]

    return tf.transpose(polyphase_matrix, perm=[1, 0, 2])             # [2 (=blocks_n), filters_n, filters_n]

  def _inv_polyphase_matrix(self, precompute_dtype):
    """Decomposed part of inverse poly-phase matrix of an MDCT filter bank, with a sine modulated window:
      G(z) = DCT4 x (D^-1 x F_{synthesis})

    :param precompute_dtype     The type of the elements of the resulting tensor.
    :return:                  D^-1 x F_{synthesis}       [2, filters_n, filters_n]
    """
    # invert Fa matrix for synthesis after removing last dim:
    F_synthesis = tf.expand_dims(
      tf.linalg.inv(
        self._filter_window_matrix(precompute_dtype)), axis=0)               # [1 (=blocks_n), filters_n, filters_n]
    D_inv = self._inverse_delay_matrix(precompute_dtype)                     # [filters_n, 2 (=blocks_n), filters_n]
    inv_polyphase_matrix = self._polymatmul(D_inv, F_synthesis)            # [filters_n, 2 (=blocks_n), filters_n]

    return tf.transpose(inv_polyphase_matrix, perm=[1, 0, 2])              # [2 (=blocks_n), filters_n, filters_n]

  def _filter_window_matrix(self, precompute_dtype):
    """Produces a diamond shaped folding matrix F from the sine window which leads to identical analysis and
    synthesis base-band impulse responses. Hence has det 1 or -1.

    :param precompute_dtype    the type of the elements of the resulting tensor.
    :return:                   F of shape (filters_n, filters_n)
    """
    if self.window_type.lower() == 'sine':
      # Sine window:
      filter_bank_windows = tf.sin(
        math.pi / (2 * self.filters_n) * tf.cast(tf.range(0.5, (3 * self.filters_n) // 2 + 0.5),
                                                 dtype=precompute_dtype))
    elif self.window_type.lower() == 'vorbis':
      filter_bank_windows = tf.sin(
        math.pi / 2. * tf.sin(
          math.pi / (2. * self.filters_n) * tf.cast(tf.range(0.5, (3 * self.filters_n) // 2 + 0.5),
                                                    dtype=precompute_dtype)) ** 2)
    else:
      # no modified window (issues with stop-band attenuation)
      filter_bank_windows = tf.ones(shape=[self.filters_n + self.filters_n // 2], dtype=precompute_dtype)

    # lace window coefficients around diamond matrix
    F_upper_left = tf.reverse(tf.linalg.diag(filter_bank_windows[0:self.filters_n // 2]), axis=[1])
    F_lower_left = tf.linalg.diag(filter_bank_windows[self.filters_n // 2:self.filters_n])
    F_upper_right = tf.linalg.diag(filter_bank_windows[self.filters_n:(self.filters_n + int(self.filters_n / 2))])
    # F matrix is completed via consistency rule (hence no need for filter_bank_windows range to extend to 2filters_n-1
    sym = 1.0  # The kind of symmetry: +-1
    ff = tf.reverse((sym * tf.ones((self.filters_n // 2), dtype=precompute_dtype)
                     - filter_bank_windows[self.filters_n:(3 * self.filters_n) // 2] * filter_bank_windows[self.filters_n - 1:self.filters_n // 2 - 1:-1])
                    / filter_bank_windows[0:self.filters_n // 2], axis=[0])
    # note for sine window:
    # ff entry i (i=0..filters_n/2)
    #    = (1-sin(pi/(2filters_n)(filters_n+i+.5)) * sin(pi/(2filters_n)(filters_n-i-.5))) / sin(pi/(2filters_n)(i+.5))
    #    = sin(pi/(2filters_n) [2filters_n - i+.5])
    F_lower_right = -tf.reverse(tf.linalg.diag(ff), axis=[1])

    return tf.concat([tf.concat([F_upper_left, F_upper_right], axis=1),
                      tf.concat([F_lower_left, F_lower_right], axis=1)], axis=0)

  def _delay_matrix(self, precompute_dtype):
    """Delay matrix D(z), which has delay z^-1 on the upper half of its diagonal
    in a 3D polynomial representation (exponents of z^-1 are in the third dimension)

    :param precompute_dtype:    the type of the elements of the resulting tensor.
    :return:                  delay matrix [2, filters_n, filters_n]
    """
    a = tf.linalg.diag(tf.concat([tf.zeros(self.filters_n // 2, dtype=precompute_dtype),
                                  tf.ones(self.filters_n // 2, dtype=precompute_dtype)], axis=0))
    b = tf.linalg.diag(tf.concat([tf.ones(self.filters_n // 2, dtype=precompute_dtype),
                                  tf.zeros(self.filters_n // 2, dtype=precompute_dtype)], axis=0))
    return tf.stack([a, b], axis=0)

  def _inverse_delay_matrix(self, precompute_dtype):
    """Causal inverse delay matrix D^{-1}(z), which has delays z^-1  on the lower
    half in 3D polynomial representation (exponents of z^-1 are in third dimension)

    :param precompute_dtype:    the type of the elements of the resulting tensor.
    :return:                  inverse delay matrix [filters_n, 2, filters_n]
    """
    a = tf.linalg.diag(tf.concat([tf.ones(self.filters_n // 2, dtype=precompute_dtype),
                                  tf.zeros(self.filters_n // 2, dtype=precompute_dtype)], axis=0))
    b = tf.linalg.diag(tf.concat([tf.zeros(self.filters_n // 2, dtype=precompute_dtype),
                                  tf.ones(self.filters_n // 2, dtype=precompute_dtype)], axis=0))
    return tf.stack([a, b], axis=1)

  def _split_in_blocks(self, x):
    """Split signal in each channel into blocks of size filters_n and folds
    the channel dimension into the batches dimension.
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

    :param x:           multi-channel input signal [batches_n, samples_n, channels_n]
    :return:            multi-channel signal split in blocks [batches_n x channels_n, blocks_n, filter_bands_n]
    :raises             InvalidArgumentError when samples_n is not a multiple of filter_bands_n
    """
    batches_n = tf.shape(x)[0]
    channels_n = x.shape[2]

    x_transpose = tf.transpose(x, perm=[0, 2, 1])
    # 1. fold channels dimension into batches and
    # 2. split in blocks
    x_pp = tf.reshape(x_transpose, shape=[batches_n * channels_n, -1, self.filters_n])

    return x_pp

  def _merge_blocks(self, xp, batches_n: tf.Tensor, channels_n: tf.Tensor):
    """Glues back together the blocks (on axis=1) and un-folds the channels from the batches dimension

    :param xp:  multi-channel signal split in blocks [batches_n x channels_n, blocks_n, filter_bands_n]
    :return:    multi-channel signal [batches_n, samples_n, channels_n]
    """
    # un-fold channels dimension
    xp = tf.reshape(xp, shape=[batches_n, channels_n, -1])
    xp = tf.transpose(xp, perm=[0, 2, 1])

    return xp

  def _dct4(self, samples):
    """Orthogonal DCT4 transformation on axis=1 of samples

        y_k = \\sqrt{2/N} \\sum_{n=0}^{N-1} x_n \\cos( \\pi/N (n+1/2) (k+1/2) )

    The dct4() amplitudes y_k are maximally of the order of
       ~ \\sqrt{2/N} N = \\sqrt{2 N}
    as can be seen from the dct4 formula above.
    Note: DCT4 is its own reverse

    :param samples:     3d array of samples already in shape [batches_n x channels_n, blocks_n, filters_n]
                        axis=-1 are filter_n samples from the signal, on which the DCT4 is performed
    :return:            3-D array where axis=1 is DCT4-transformed, orthonormal with shape [batches_n x channels_n, blocks_n, filters_n]
                        axis=2 contains coefficients of the cosine harmonics which compose the filters_n block signal
    """
    # up-cast to float32 or float64 (required for tf.signal.dct)
    if samples.dtype == tf.float32 or samples.dtype == tf.float64:
      samples_upcast = samples
    else:
      samples_upcast = tf.cast(samples, dtype=tf.float32)

    # up-sample by inserting zeros for all even entries: this allows us to express DCT-IV as a DCT-III
    upsampled = tf.reshape(
      tf.stack([tf.zeros(tf.shape(samples_upcast), dtype=samples_upcast.dtype), samples_upcast], axis=-1),
      shape=[tf.shape(samples_upcast)[0], tf.shape(samples_upcast)[1], 2 * self.filters_n])

    # conversion factor \sqrt{2} is needed to go from orthogonal DCT-III to orthogonal DCT-IV,
    y = tf.signal.dct(upsampled, type=3, axis=-1, norm='ortho')

    # downcast back to original
    if samples.dtype == tf.float32 or samples.dtype == tf.float64:
      y_downcast = y
    else:
      y_downcast = tf.cast(y, dtype=samples.dtype)

    # down-sample
    return tf.cast(tf.sqrt(2.), dtype=samples.dtype) * y_downcast[:, :, 0:self.filters_n]

  def _polymatmul(self, A, F):
    """Matrix multiplication of matrices where each entry is a polynomial.

    :param A: 3D matrix A, with 2nd dimension the polynomial coefficients
    :param F: 3D matrix F, with 1st dimension the polynomial coefficients (filter/kernel)
    :return:  3D matrix C, with 2nd dimension the polynomial coefficients
          C_{b n k} = \\sum_{m=0}^n    \\sum_q A{b m q}           F_{n-m q k}
                    = \\sum_{m=0}^n    \\sum_q A{b m q}           F_flip{f-n+m q k}   with f=tf.shape(F)[0]-1=degree(F)
                    = \\sum_{mm=f-n}^f \\sum_q A{b mm-f+n q}      F_flip{mm q k}      with mm=m+(f-n)
                    = \\sum {mm=f-n}^f \\sum_q A_padded{b mm+n q} F_flip{mm q k}      A padded with degree(F)
                    = tf.nn.convolution(A_padded, F_flip)
        where F plays role of filter (kernel) which is dragged over the padded A.
    """
    F_flip = tf.reverse(F, axis=[0])

    # add padding: degree(F) zeros at start and end along A's polynomial dimension
    F_degree = tf.shape(F)[0] - 1
    A_padded = tf.pad(A, paddings=tf.convert_to_tensor([[0, 0], [F_degree, F_degree], [0, 0]]))

    return tf.nn.convolution(A_padded, F_flip, padding="VALID")
