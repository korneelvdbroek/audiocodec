import unittest

import numpy as np
import tensorflow as tf

from audiocodec import psychoacoustic

EPS = 1e-5


class TestPsychoacoustic(unittest.TestCase):
  def test_inverse_identity(self):
    """Check that x = ampl_to_norm(norm_to_ampl(x))"""
    # compare original x with y = ampl_to_norm(norm_to_ampl(x))
    for mdct_norm in tf.range(-1., 1., 0.2):
      mdct_ampl = psychoacoustic.norm_to_ampl(mdct_norm)

      self.assertLess(psychoacoustic.ampl_to_norm(mdct_ampl) - mdct_norm, EPS, "Should be zero")

    self.assertLess(psychoacoustic.ampl_to_norm(psychoacoustic.norm_to_ampl(1.2)) - 1.2, EPS, "Should be zero")


if __name__ == '__main__':
  unittest.main()
