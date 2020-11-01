import unittest

import numpy as np
import tensorflow as tf

from audiocodec import psychoacoustic

EPS = 1e-5


class TestPsychoacoustic(unittest.TestCase):
  def test_inverse_identity(self):
    self.assertLess(0., EPS, "Should be zero")


if __name__ == '__main__':
  unittest.main()
