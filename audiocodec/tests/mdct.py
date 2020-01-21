import unittest

import numpy as np
import tensorflow as tf

from audiocodec.mdct import MDCT

EPS = 1e-6


def sine_wav(amplitude, frequency, sample_rate=44100, duration_sec=2.0):
  """Create wav which contains sine wave
  """
  wave_data = amplitude * np.sin(2.0 * np.pi * frequency * tf.range(0, sample_rate * duration_sec, dtype=tf.float32) / sample_rate)
  return wave_data[np.newaxis, :]


class TestMDCT(unittest.TestCase):
  def test_inverse_identity(self):
    """Check that x = MDCT^{-1}(MDCT(x))"""

    # mdct setup
    filters_n = 256
    mdct = MDCT(filters_n)

    # create test signal
    wave_data = sine_wav(0.8, 880, sample_rate=16000, duration_sec=1.)
    wave_data = wave_data[:, 0:filters_n * int(wave_data.shape[1] / filters_n)]

    # transform and go back
    spectrum = mdct.transform(wave_data)
    wave_reproduced = mdct.inverse_transform(spectrum)

    # compare original x with y = MDCT^{-1}(MDCT(x))
    zero = tf.reduce_max(tf.abs(wave_data - wave_reproduced[:, filters_n:-filters_n]))

    self.assertLess(zero, EPS, "Should be zero")


if __name__ == '__main__':
  unittest.main()
