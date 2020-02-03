import unittest

import tensorflow as tf

from audiocodec.logspectrogram import Spectrogram


EPS = 1e-6


class TestSpectrogram(unittest.TestCase):
  def test_inverse_identity(self):
    """Check that x = note_to_freq(freq_to_note(x))
    """
    filter_bands_n = 90
    blocks_n = 90
    sample_rate = blocks_n*filter_bands_n
    logspectrumconvertor = Spectrogram(sample_rate, filter_bands_n)

    # [batch, block, filter_bands_n]
    spectrum = tf.random.uniform([32, blocks_n, filter_bands_n])
    log_spectrum = logspectrumconvertor.freq_to_note(spectrum)
    self.assertLess(tf.reduce_max(logspectrumconvertor.note_to_freq(log_spectrum) - spectrum), EPS, "Should be zero")


if __name__ == '__main__':
  unittest.main()

