import unittest

import numpy as np
import tensorflow as tf

from audiocodec import psychoacoustic
from audiocodec.mdctransformer import MDCTransformer
from audiocodec.tests import test_mdctransformer as mdct_tests

EPS = 1e-5


class TestPsychoacoustic(unittest.TestCase):
  def test_energy_conservation_W(self):
    """
    Energy (intensity) in any mdct_amplitude vector, should be conserved when transforming to the bark scale.
    We hence multiply self.W with all unit mdct_amplitude vectors (i.e. sum over the rows) and check the result is one
    """
    pa_model = psychoacoustic.PsychoacousticModel(sample_rate=32768, filter_bands_n=64)
    should_be_all_zeros = tf.reduce_sum(pa_model.W, axis=1, keepdims=False) - 1.0
    self.assertLess(tf.reduce_sum(tf.abs(should_be_all_zeros)), 1e-6)

  def test_energy_conservation_W_inv(self):
    """
    Energy (intensity) in any mdct_amplitude vector, should be conserved when transforming to the bark scale.
    We hence multiply self.W with all unit mdct_amplitude vectors (i.e. sum over the rows) and check the result is one
    """
    pa_model = psychoacoustic.PsychoacousticModel(sample_rate=32768, filter_bands_n=64)
    should_be_all_zeros = tf.reduce_sum(pa_model.W_inv, axis=1, keepdims=False) - 1.0
    self.assertLess(tf.reduce_sum(tf.abs(should_be_all_zeros)), 1e-6)

  def test_tonality_tone(self):
    filters_n = 64
    mdct = MDCTransformer(filters_n)
    # create test signal
    wave_data = mdct_tests.sine_wav(0.8, 4, sample_rate=64, duration_sec=5.)
    # transform
    spectrum = mdct.transform(wave_data)

    pa_model = psychoacoustic.PsychoacousticModel(sample_rate=filters_n, filter_bands_n=filters_n)
    tonality = pa_model.tonality(spectrum)
    self.assertEqual(tonality[0, 1], 1.)

  def test_tonality_noise(self):
    filters_n = 64
    blocks_n = 10
    mdct = MDCTransformer(filters_n)
    # create test signal
    batches_n = 10
    channels_n = 2
    wave_data = tf.random.uniform(shape=(batches_n, blocks_n*filters_n, channels_n), minval=-1.0, maxval=1.0)
    # transform
    spectrum = mdct.transform(wave_data)

    pa_model = psychoacoustic.PsychoacousticModel(sample_rate=filters_n, filter_bands_n=filters_n)
    tonality = pa_model.tonality(spectrum)

    # check tonality shape
    self.assertEqual(tonality.shape[0], batches_n)
    self.assertEqual(tonality.shape[1], blocks_n+1)
    self.assertEqual(tonality.shape[2], 1)
    self.assertEqual(tonality.shape[3], channels_n)

    # check tonality is low
    self.assertLess(tf.reduce_mean(tonality[0, 1:-1]), 0.1)


if __name__ == '__main__':
  unittest.main()
