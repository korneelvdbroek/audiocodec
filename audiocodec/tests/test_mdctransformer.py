import unittest

import numpy as np
import tensorflow as tf

from audiocodec.mdctransformer import MDCTransformer

EPS = 1e-5


def sine_wav(amplitude, frequency, sample_rate=44100, duration_sec=2.0):
  """Create wav which contains sine wave
  """
  wave_data = amplitude * np.sin(2.0 * np.pi * frequency * tf.range(0, sample_rate * duration_sec, dtype=tf.float32) / sample_rate)
  return tf.reshape(wave_data, shape=[1, -1, 1])


class TestMDCT(unittest.TestCase):
  def test_inverse_identity(self):
    """Check that x = MDCT^{-1}(MDCT(x))"""

    # mdct setup
    filters_n = 256
    mdct = MDCTransformer(filters_n)

    # create test signal
    wave_data = sine_wav(0.8, 880, sample_rate=16000, duration_sec=1.)
    wave_data = wave_data[:, 0:filters_n * int(wave_data.shape[1] / filters_n)]

    # transform and go back
    spectrum = mdct.transform(wave_data)
    wave_reproduced = mdct.inverse_transform(spectrum)

    # compare original x with y = MDCT^{-1}(MDCT(x))
    zero = tf.reduce_max(tf.abs(wave_data - wave_reproduced[:, filters_n:-filters_n]))

    self.assertLess(zero, EPS, "Should be zero")

  def test_mdct_calculation(self):
    """Compute mdct on sin function"""
    # mdct setup
    filters_n = 64
    mdct = MDCTransformer(filters_n)

    # create test signal
    wave_data = sine_wav(0.8, 4, sample_rate=64, duration_sec=4.)
    wave_data = wave_data[:, 0:filters_n * int(wave_data.shape[1] / filters_n)]

    # transform and go back
    spectrum = mdct.transform(wave_data)
    correct_spectrum = [-0.000412722176, 0.000430465181, 0.000789350364, -0.000867388735, -0.00275337417,
                        0.0132110268, 0.0193885863, 0.156005412, -0.233544752, -0.0129148215]
    for i, a in enumerate(correct_spectrum):
      self.assertLess(spectrum[0, 1, i, 0] - a, 1e-6)

  def test_mdct_shape(self):
    """Check shape of mdct transform"""
    # mdct setup
    filters_n = 64
    mdct = MDCTransformer(filters_n)

    # create test signal
    batches_n = 128
    blocks_n = 10
    samples_n = blocks_n*filters_n
    channels_n = 2
    wave_data = tf.random.normal(shape=(batches_n, samples_n, channels_n))

    # transform and go back
    spectrum = mdct.transform(wave_data)

    self.assertEqual(tf.shape(spectrum)[0], batches_n)
    self.assertEqual(spectrum.shape[1], blocks_n+1)
    self.assertEqual(spectrum.shape[2], filters_n)
    self.assertEqual(spectrum.shape[3], channels_n)


if __name__ == '__main__':
  unittest.main()
