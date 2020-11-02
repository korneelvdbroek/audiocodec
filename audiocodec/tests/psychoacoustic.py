import unittest

import numpy as np
import tensorflow as tf

from audiocodec import psychoacoustic
from audiocodec.mdct import MDCT
from audiocodec.tests import mdct as mdct_tests

import matplotlib.pyplot as plt


EPS = 1e-5


class TestPsychoacoustic(unittest.TestCase):
  def test_inverse_identity(self):
    """Check that x = ampl_to_norm(norm_to_ampl(x))"""
    # compare original x with y = ampl_to_norm(norm_to_ampl(x))
    for mdct_norm in tf.range(-1., 1., 0.2):
      mdct_ampl = psychoacoustic.norm_to_dB_ampl(mdct_norm)

      self.assertLess(psychoacoustic.dB_ampl_to_norm(mdct_ampl) - mdct_norm, EPS, "Should be zero")

    self.assertLess(psychoacoustic.dB_ampl_to_norm(psychoacoustic.norm_to_dB_ampl(1.2)) - 1.2, EPS, "Should be zero")

  def test_masking_threshold(self):
    filters_n = 256
    sample_rate = 64*filters_n
    mdct = MDCT(filters_n)
    # create test signal
    wave_data = mdct_tests.sine_wav(0.8, 1024, sample_rate=sample_rate, duration_sec=5.)
    # transform
    spectrum = mdct.transform(wave_data)

    ampl = psychoacoustic.norm_to_dB_ampl(spectrum)

    y_amplitudes = 10. * tf.math.log(ampl ** 2.0) / tf.math.log(10.)
    plt.plot(y_amplitudes[0, 3, :])

    pa_model = psychoacoustic.PsychoacousticModel(sample_rate=sample_rate, filter_bands_n=filters_n)
    tonality = pa_model.tonality(ampl)
    masking_threshold = pa_model.global_masking_threshold(ampl, tonality)
    quiet_threshold = pa_model._mappingfrombark(pa_model._quiet_threshold_amplitude_in_bark())

    y_masking_threshold = 10. * tf.math.log(masking_threshold ** 2.0) / tf.math.log(10.)
    y_quiet_threshold = 10. * tf.math.log(quiet_threshold ** 2.0) / tf.math.log(10.)
    plt.plot(y_masking_threshold[0, 3, :])
    plt.plot(y_quiet_threshold[0, 0, :])

    mask = psychoacoustic.dB_ampl_to_norm(masking_threshold)
    tf.print(mask, summarize=100)
    plt.show()


  def test_tonality_tone(self):
    filters_n = 64
    mdct = MDCT(filters_n)
    # create test signal
    wave_data = mdct_tests.sine_wav(0.8, 4, sample_rate=64, duration_sec=5.)
    # transform
    spectrum = mdct.transform(wave_data)

    pa_model = psychoacoustic.PsychoacousticModel(sample_rate=filters_n, filter_bands_n=filters_n)
    tonality = pa_model.tonality(spectrum)
    self.assertEqual(tonality[0, 1], 1.)

  def test_tonality_noise(self):
    filters_n = 64
    seconds_n = 10
    mdct = MDCT(filters_n)
    # create test signal
    wave_data = tf.random.uniform(shape=(1, seconds_n*filters_n), minval=-1.0, maxval=1.0)
    # transform
    spectrum = mdct.transform(wave_data)

    pa_model = psychoacoustic.PsychoacousticModel(sample_rate=filters_n, filter_bands_n=filters_n)
    tonality = pa_model.tonality(spectrum)
    self.assertLess(tf.reduce_mean(tonality[0, 1:-1]), 0.1)


if __name__ == '__main__':
  unittest.main()
