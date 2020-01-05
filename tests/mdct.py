import unittest

import numpy as np
import tensorflow as tf

from audiocodec import mdct

EPS = 1e-6


def sine_wav(amplitude, frequency, sample_rate=44100, duration_sec=2.0):
    """Create wav which contains sine wave
    """
    wave_data = amplitude * np.sin(2.0 * np.pi * frequency * np.arange(0, sample_rate * duration_sec) / sample_rate)
    return wave_data[np.newaxis, :]


class TestMDCT(unittest.TestCase):
    def test_idempotence(self):
        wave_data = sine_wav(0.8, 880, sample_rate=16000, duration_sec=1.)

        filter_bands_n = 256
        mdct_setup = mdct.setup(filter_bands_n)

        wave_data = wave_data[:, 0:filter_bands_n * int(wave_data.shape[1] / filter_bands_n)]
        wave_data = tf.convert_to_tensor(wave_data, dtype=tf.float32)

        spectrum = mdct.transform(wave_data, mdct_setup)
        wave_reproduced = mdct.inverse_transform(spectrum, mdct_setup)

        zero = tf.reduce_max(tf.abs(wave_data - wave_reproduced[:, filter_bands_n:-filter_bands_n]))

        self.assertLess(zero, EPS, "Should be zero")


if __name__ == '__main__':
    unittest.main()
