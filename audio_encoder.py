# !/usr/bin/env python

""" Main program to test the psycho-acoustic encoder/decoder
"""

import os
import winsound
import imageio

import tensorflow as tf
import numpy as np

import scipy.io.wavfile as wav
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator

from audiocodec import codec_utils, codec
from audiocodec.mdct import MDCT
from audiocodec.psychoacoustic import PsychoacousticModel, _dB_MAX
from audiocodec import psychoacoustic as pa
from audiocodec.logspectrogram import Spectrogram


# note: local install https://packaging.python.org/tutorials/installing-packages/#installing-from-a-local-src-tree

# basic guiding principles:
# 1. why do we work in the freq. space? analogy with ear (assuming the resonance theory is correct)
# 2. why mdct not fft? It has real valued coefficients, so we avoid the phase problem and only have a sign to track
# 3. GAN training needs to be progressive (hierarchical) -- need to "blur" audio to reduce dimensions
#    observation that freq. amplitudes in subsequent blocks oscillate
#    --> so 2nd mdct transform and filter our dominant patterns (we again NEED to transform, since direct in time space
#        there is no efficient way to separate important and unimportant)


# todo: mdct2 -- use better blurring (instead of top_k): audio needs to be clean!
#  ideas: 1. keep top_k per freq_bin (although this is not a global song approach...)
#         2. actually blur the spectrum2 (BUT KEEP SIGNS!) --> blur by amplitude circle
#  make sure adjacent bins are both kept (to prevent audio aliasses)
#  not sure we need to clean the audio actually (unless artefacts remain later on...)
# todo: find a time averaging method to smooth over time (and keep chords)


CPU_ONLY = False
DEBUG = False

# Set CPU as available physical device
if CPU_ONLY:
  print('Running Tensorflow on CPU only')
  my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
  tf.config.experimental.set_visible_devices(devices=my_devices)

if DEBUG:
  tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
  # To find out which devices your operations and tensors are assigned to
  tf.debugging.set_log_device_placement(True)


def play_wav(wave_data, sample_rate):
  print("Playing wave data...", end=' ', flush=True)
  audio_filepath = "temp.wav"
  save_wav(audio_filepath, wave_data, sample_rate)
  winsound.PlaySound(audio_filepath, winsound.SND_FILENAME)
  os.remove(audio_filepath)
  print("done")
  return


def sine_wav(amplitude, frequency, sample_rate=44100, duration_sec=2.0):
  """Create wav which contains sine wave
  """
  wave_data = amplitude * np.sin(2.0 * np.pi * frequency * tf.range(0, sample_rate * duration_sec, dtype=tf.float32) / sample_rate)
  return tf.expand_dims(wave_data, axis=0), sample_rate


def create_wav(sample_rate=44100):
  """Create wav which contains sine wave
  """
  frequency = 220
  amplitude = 0.5

  wave_data = []
  for f in [frequency * np.power(2., n) for n in range(4)]:
    wave_data.extend(amplitude * np.sin(2.0 * np.pi * f * np.arange(0, sample_rate) / sample_rate))

  return tf.cast(tf.expand_dims(wave_data, axis=0), dtype=tf.float32), sample_rate


def load_wav(audio_filepath, sample_rate=None):
  """Read in wav file at given sample_rate.

  :param audio_filepath: path and filename of wav file
  :param sample_rate:    sample rate at which audio file needs to be read.
                         If sample_rate is None (default), then original sample rate is preserved
  :return:               raw wave data in range -1..1 (#channels x #audio_samples) and sample rate
  """
  print("Loading audio file {0}...".format(audio_filepath), end=' ', flush=True)
  wave_data, sample_rate = librosa.core.load(audio_filepath, sr=sample_rate, mono=False)

  if wave_data.ndim == 1:
    wave_data = np.reshape(wave_data, [1, wave_data.shape[0]])
  wave_data = tf.convert_to_tensor(wave_data, dtype=tf.float32)
  print('done (sample rate = {})'.format(sample_rate))
  return wave_data, sample_rate


def save_wav(audio_filepath, wave_data, sample_rate):
  wave_data = wave_data.numpy().T
  wave_data = np.clip(2**15 * wave_data, -2 ** 15, 2 ** 15 - 1)  # limit values in the array
  wav.write(audio_filepath, sample_rate, np.int16(wave_data))
  return


def clip_wav(start, stop, wave_data, sample_rate):
  minute_start, second_start = start
  minute_stop, second_stop = stop

  return wave_data[:, (minute_start*60+second_start)*sample_rate:(minute_stop*60+second_stop)*sample_rate]


def test_codec():
  # load audio file
  audio_filepath = './data/'
  audio_filename = 'asot_02_cosmos_sr8100_118_128'
  sample_rate = 90 * 90  # None   # 90*90
  wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)

  # modify signal
  print('Modifying signal...')
  wave_data = codec_utils.modify_signal(wave_data, sample_rate, alpha=0.8)

  # (encode and) plot spectrum
  print('Plotting spectrum...')
  fig, ax = plt.subplots()
  ims = codec_utils.plot_spectrum(ax, wave_data, [0], range(10, 20), sample_rate, alpha=0.6)
  _ = animation.ArtistAnimation(fig, ims, interval=500)
  plt.show()

  # encode
  print('Encoding...')
  encoder_init = codec.encoder_setup(sample_rate, filter_bands_n=1024, bark_bands_n=64, alpha=0.6)
  mdct_amplitudes_quantized, log_mask_thresholds_bark = codec.encoder(wave_data, encoder_init, quality=100)

  # decode
  print('Decoding...')
  wave_data_reconstructed = codec.decoder(mdct_amplitudes_quantized, log_mask_thresholds_bark, encoder_init)

  # cut up wav
  # wave_data_reconstructed = wave_data[:, 44100*(11*60 + 37):44100*(16*60+40)]

  # write back to WAV file
  play_wav(wave_data_reconstructed, sample_rate)
  filepath, ext = os.path.splitext(audio_filepath + audio_filename)
  decoded_filepath = filepath + '_reconstructed.wav'
  save_wav(decoded_filepath, wave_data_reconstructed, sample_rate)

  return


def test_dB_level():
  # setup
  filter_bands_n = 512
  sample_rate = 44100
  mdct = MDCT(filter_bands_n, dB_max=_dB_MAX)
  psychoacoustic = PsychoacousticModel(sample_rate, filter_bands_n, bark_bands_n=24, alpha=0.6)

  # load wav
  audio_filepath = './data/'
  audio_filename = 'asot_02_cosmos'
  wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)
  wave_data = clip_wav((1, 18), (1, 28), wave_data, sample_rate)
  wave_data = wave_data[:, 0:filter_bands_n * int(wave_data.shape[1] / filter_bands_n)]
  # wave_data, sample_rate = sine_wav(1.0, 3.95*787.5, sample_rate, 1.0)

  fig, ax = plt.subplots()
  ims = codec_utils.plot_spectrum(mdct, psychoacoustic, ax, wave_data, channels=[0], blocks=range(10, 20),
                                  sample_rate=sample_rate, filter_bands_n=filter_bands_n)
  _ = animation.ArtistAnimation(fig, ims, interval=500)
  plt.show()


def play_from_im():
  # setup
  filter_bands_n = 90   # needs to be even 44100 = 490 x 90
  sample_rate = 90*90   # try to have +/- 10ms per freq bin (~speed of neurons)
  drown = .90
  mdct = MDCT(filter_bands_n, dB_max=_dB_MAX)

  image_filepath = './data/'
  image_filename = 'asot_02_cosmos'   # 'asot_02_cosmos_sr8100_118_128.wav'
  image_filename_post_fix = '_sr{0:.0f}_118_128_{1:03.0f}_edit1'.format(sample_rate, 100*drown)

  spectrum_modified = codec_utils.read_spectrogram(image_filepath + image_filename + image_filename_post_fix + ".png")
  wave_reproduced = mdct.inverse_transform(spectrum_modified)

  play_wav(wave_reproduced, sample_rate)
  if image_filename is not None:
    save_wav(image_filepath + image_filename + image_filename_post_fix + '_from_image.wav', wave_reproduced, sample_rate)


def test_gradient():
  # setup
  filter_bands_n = 90   # needs to be even 44100 = 490 x 90
  sample_rate = 90*90   # try to have +/- 10ms per freq bin (~speed of neurons)
  drown = 0.0
  mdct = MDCT(filter_bands_n, dB_max=_dB_MAX)
  psychoacoustic = PsychoacousticModel(sample_rate, filter_bands_n, bark_bands_n=24, alpha=0.6)

  # load audio file
  # audio_filename = None
  audio_filepath = './data/'
  audio_filename = 'asot_02_cosmos'   # 'asot_02_cosmos_sr8100_118_128.wav'
  audio_filename_post_fix = '_sr{0:.0f}_118_128_{1:03.0f}'.format(sample_rate, 100*drown)
  wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)
  wave_data = clip_wav((1, 18), (1, 28), wave_data, sample_rate)
  # wave_data, sample_rate = sine_wav(1.0, 3.95*787.5, sample_rate, 1.0)
  wave_data = wave_data[:, 0:filter_bands_n * int(wave_data.shape[1] / filter_bands_n)]

  # play_wav(wave_data, sample_rate)

  # manipulate signal
  mdct_ampl = mdct.transform(wave_data)
  mdct_norm = pa.ampl_to_norm(mdct_ampl)
  mdct_norm = mdct_norm[0:1, :, :]

  # filter (strong beta to make visual difference)
  spectrum_modified = psychoacoustic.lrelu_filter(mdct_norm, drown, beta=0.01)

  # plot
  fig, (ax1, ax2) = plt.subplots(nrows=2)
  codec_utils.plot_spectrogram(ax1, mdct_norm, sample_rate, filter_bands_n)
  codec_utils.plot_spectrogram(ax2, spectrum_modified, sample_rate, filter_bands_n)
  plt.show()

  # vary each of the entries of mdct_amplitude and see impact on chosen entry of total_masking_norm
  channel = 0
  block = 1
  freq_bin = 14
  with tf.GradientTape() as d:
    d.watch(mdct_norm)

    # check gradients!!!!!
    total_masking_norm = psychoacoustic.lrelu_filter(mdct_norm, drown, beta=0.2)

    element = total_masking_norm[channel, block, freq_bin]

  diff_filter = d.gradient(element, mdct_norm)

  tf.print(diff_filter, summarize=20)
  tf.print(tf.shape(mdct_norm))
  tf.print('this one shouldnt be nan ==> ', tf.reduce_sum(diff_filter))
  tf.print('element = ', element)

  return


def test_mdct2():
  # setup
  # higher drown is better is filtering more with k!! (less weird sounds)
  blocks_per_sec = 64
  filter_bands_n = 64
  drown = 0.2
  sample_rate = filter_bands_n * blocks_per_sec
  mdct = MDCT(filter_bands_n, dB_max=_dB_MAX, window_type='vorbis')
  psychoacoustic = PsychoacousticModel(sample_rate, filter_bands_n, bark_bands_n=24, alpha=0.6)

  # load audio file
  audio_filepath = './data/'
  audio_filename = 'asot_02_cosmos'   # 'asot_02_cosmos_sr8100_118_128.wav'
  audio_filename_post_fix = '_sr{0:.0f}_118_128_{1:03.0f}'.format(sample_rate, 100 * drown)
  wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)
  # wave_data = clip_wav((1, 18), (1, 28), wave_data, sample_rate)
  # wave_data, sample_rate = create_wav(sample_rate)
  channel = 0
  wave_data = wave_data[channel:(channel+1), 0:filter_bands_n * int(wave_data.shape[1] / filter_bands_n)]
  # wave_data, sample_rate = sine_wav(1.0, 3.95 * 787.5, sample_rate, 1.0)

  # play input
  # play_wav(wave_data, sample_rate)

  # manipulate signal
  spectrum_ampl = mdct.transform(wave_data)
  spectrum_ampl = pa.ampl_to_norm(spectrum_ampl)

  # psychoacoustic filter (filter out what's in-audible --> shows key patterns) ==> helps later mdct2 filter
  spectrum = psychoacoustic.lrelu_filter(spectrum_ampl, drown, beta=0.000001)

  # mdct on time axis of each freq bucket
  # spectrumX = [#channels = freq_buckets, #blocks+1, filters_n]
  pattern_length = blocks_per_sec
  spectrumX = spectrum[:, ::2, :]  # sample blocks by skipping them!
  spectrumX = tf.transpose(spectrumX[channel, 0:pattern_length * int(spectrumX.shape[1] / pattern_length), :], perm=[1, 0])
  mdct2 = MDCT(pattern_length, dB_max=_dB_MAX, window_type='vorbis')
  spectrum2 = mdct2.transform(spectrumX)
  tf.print(tf.shape(spectrum2))

  # filter (keep only dominant patterns) -- note: we are in ampl space
  # seconds(4) x pattern_width(4) x ampl_freq_components(4)
  if False:
    k = 10000
    cutoffs, _ = tf.math.top_k(tf.reshape(tf.abs(spectrum2), [-1]), k)
    spectrum2_mods = tf.sign(spectrum2) * tf.where(tf.abs(spectrum2) > cutoffs[k-1], tf.abs(spectrum2), tf.zeros(tf.shape(spectrum2)))
  elif False:
    # [#freq_buckets, #blocks, #filters2_n]
    intensity_raw = tf.pow(spectrum2, 2)
    intensity = tf.transpose(intensity_raw, perm=[0, 2, 1])
    intensity = tf.reshape(intensity, [-1, tf.shape(spectrum2)[2]])
    intensity = tf.expand_dims(intensity, axis=-1)
    kernel = 1./10. * tf.ones([10, 1, 1])
    tf.print(tf.shape(intensity))
    tf.print(tf.shape(kernel))
    intensity_blurred = tf.nn.conv1d(intensity, kernel, stride=1, padding='SAME')
    intensity_blurred = intensity_blurred[:, :, 0]
    intensity_blurred = tf.reshape(intensity_blurred, [tf.shape(spectrum2)[0], tf.shape(spectrum2)[2], tf.shape(spectrum2)[1]])
    intensity_blurred = tf.transpose(intensity_blurred, perm=[0, 2, 1])
    tf.print(intensity_raw)
    tf.print(intensity_blurred)
    tf.print(tf.shape(intensity_blurred))
    spectrum2_mods = tf.sign(spectrum2) * tf.pow(tf.maximum(intensity_blurred, pa._EPSILON), 1./2.)
  else:
    spectrum2_mods = spectrum2

  # invert 2nd mdct
  spectrum1_recon = mdct2.inverse_transform(spectrum2_mods)
  # [#channels = freq_buckets, #blocks+2]
  spectrum1_recon = spectrum1_recon[:, pattern_length:-pattern_length]
  spectrum1_recon = tf.transpose(spectrum1_recon, perm=[1, 0])
  spectrum1_recon = tf.expand_dims(spectrum1_recon, axis=0)

  wave_reproduced = mdct.inverse_transform(pa.norm_to_ampl(spectrum1_recon))
  # play and save reconstructed wav
  # play_wav(wave_reproduced, sample_rate)
  save_wav(audio_filepath + audio_filename + audio_filename_post_fix + '_reconstructed.wav', wave_reproduced, sample_rate)

  # plot spectrogram
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)
  # ax1
  codec_utils.plot_spectrogram(ax1, spectrum, sample_rate, filter_bands_n)

  # ax2
  # [#freq_bins, #blocks, #pattern_length]
  spectrum2_plot = tf.reshape(spectrum2, [-1, pattern_length])
  spectrum2_plot = tf.expand_dims(spectrum2_plot, axis=0)
  ax2.imshow(np.flip(np.transpose(spectrum2_plot[0, :, :]), axis=0),
             cmap='gray', interpolation='none', aspect='auto')
  ax2.xaxis.set_major_locator(MultipleLocator(tf.shape(spectrum2)[1].numpy()))

  # ax3
  spectrum2_mods_plot = tf.transpose(spectrum2_mods, perm=[1, 0, 2])
  spectrum2_mods_plot = tf.reshape(spectrum2_mods_plot, [-1, pattern_length])
  spectrum2_mods_plot = tf.expand_dims(spectrum2_mods_plot, axis=0)
  ax3.imshow(np.flip(np.transpose(spectrum2_mods_plot[0, :, :]), axis=0),
             cmap='gray', interpolation='none', aspect='auto')
  ax3.xaxis.set_major_locator(MultipleLocator(tf.shape(spectrum2_mods)[0].numpy()))

  # ax4
  codec_utils.plot_spectrogram(ax4, spectrum1_recon, sample_rate, filter_bands_n)

  plt.show()

  return


def test_octave():
  # setup
  filter_bands_n = 90   # needs to be even 44100 = 490 x 90
  sample_rate = 90*90   # try to have +/- 10ms per freq bin (~speed of neurons)
  drown = .0
  mdct = MDCT(filter_bands_n, dB_max=_dB_MAX)

  psychoacoustic = PsychoacousticModel(sample_rate, filter_bands_n, bark_bands_n=24, alpha=0.6)
  logspectrumconvertor = Spectrogram(sample_rate, filter_bands_n)

  # load audio file
  # audio_filename = None
  if False:
    audio_filepath = './data/'
    audio_filename = 'asot_02_cosmos'   # 'asot_02_cosmos_sr8100_118_128.wav'
    audio_filename_post_fix = '_sr{0:.0f}_118_128_{1:03.0f}'.format(sample_rate, 100*drown)
    wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)
    wave_data = clip_wav((1, 18), (1, 28), wave_data, sample_rate)
  else:
    wave_data, sample_rate = create_wav(sample_rate)
    # wave_data, sample_rate = sine_wav(1.0, 3.95*787.5, sample_rate, 1.0)
  wave_data = wave_data[:, 0:filter_bands_n * int(wave_data.shape[1] / filter_bands_n)]

  # play sound
  play_wav(wave_data, sample_rate)

  # manipulate signal
  mdct_ampl = mdct.transform(wave_data)
  spectrum = pa.ampl_to_norm(mdct_ampl)

  # filter
  spectrum_modified = psychoacoustic.lrelu_filter(spectrum, drown, beta=0.001)
  log_spectrum = logspectrumconvertor.freq_to_note(spectrum_modified)

  if False:
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    codec_utils.plot_spectrogram(ax1, spectrum_modified, sample_rate, filter_bands_n)
    codec_utils.plot_logspectrogram(ax2, log_spectrum, sample_rate, filter_bands_n)
    plt.show()

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
  codec_utils.plot_spectrogram(ax1, spectrum_modified, sample_rate, filter_bands_n)
  codec_utils.plot_logspectrogram(ax2, log_spectrum, sample_rate, filter_bands_n)
  ims = codec_utils.plot_logspectrum(ax3, log_spectrum, sample_rate, filter_bands_n)
  _ = animation.ArtistAnimation(fig, ims, interval=1/90.0*1000*2.0)
  plt.show()

  return


def test_psychoacoustic():
  # setup
  filter_bands_n = 90   # needs to be even 44100 = 490 x 90
  sample_rate = 90*90   # try to have +/- 10ms per freq bin (~speed of neurons)
  drown = .80
  mdct = MDCT(filter_bands_n, dB_max=_dB_MAX)
  psychoacoustic = PsychoacousticModel(sample_rate, filter_bands_n, bark_bands_n=24, alpha=0.6)

  # load audio file
  # audio_filename = None
  audio_filepath = './data/'
  audio_filename = 'asot_02_cosmos'   # 'asot_02_cosmos_sr8100_118_128.wav'
  audio_filename_post_fix = '_sr{0:.0f}_118_128_{1:03.0f}'.format(sample_rate, 100*drown)
  wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)
  wave_data = clip_wav((1, 18), (1, 28), wave_data, sample_rate)
  # wave_data, sample_rate = sine_wav(1.0, 3.95*787.5, sample_rate, 1.0)
  wave_data = wave_data[:, 0:filter_bands_n * int(wave_data.shape[1] / filter_bands_n)]

  # play_wav(wave_data, sample_rate)

  # manipulate signal
  mdct_ampl = mdct.transform(wave_data)
  spectrum = pa.ampl_to_norm(mdct_ampl)

  # filter
  spectrum_modified = psychoacoustic.lrelu_filter(spectrum, drown, beta=0.2)
  codec_utils.save_spectrogram(spectrum_modified, audio_filepath + audio_filename + audio_filename_post_fix + ".png")

  # back to audio signal
  wave_reproduced = mdct.inverse_transform(pa.norm_to_ampl(spectrum_modified))

  # plot both spectrograms
  if True:
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    codec_utils.plot_spectrogram(ax1, spectrum, sample_rate, filter_bands_n)
    codec_utils.plot_spectrogram(ax2, spectrum_modified, sample_rate, filter_bands_n)
    plt.show()

  # plot time-slice
  if False:
    freq_bin_slice = 5
    plt.plot(spectrum_modified[0, :, freq_bin_slice])
    plt.plot(spectrum_modified[0, :, freq_bin_slice + 1])
    plt.plot(spectrum_modified[0, :, freq_bin_slice + 2])
    plt.show()

  play_wav(wave_reproduced, sample_rate)
  if audio_filename is not None:
    save_wav(audio_filepath + audio_filename + audio_filename_post_fix + '_reconstructed.wav', wave_reproduced, sample_rate)

  return


def test_mdct_precision():
  # setup
  filter_bands_n = 90
  sample_rate = 90 * 90  # None   # 90*90
  mdct = MDCT(filter_bands_n, dB_max=_dB_MAX, window_type='vorbis')

  # load audio file
  # audio_filepath = './data/'
  # audio_filename = 'sine'   # 'asot_02_cosmos_sr8100_118_128.wav'
  # wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)

  print("noise_amplitude delta_mean delta_std")
  for noise_size in range(0, 300, 5):
    noise_amplitude = noise_size * 10. ** (-8)

    spectrum = []
    errors = []
    for i in range(100):
      wave_data, sample_rate = sine_wav(1.0, 3.95 * 787.5, sample_rate, 1.0)
      wave_data = wave_data[:, 0:filter_bands_n * int(wave_data.shape[1] / filter_bands_n)]

      # add noise:
      wave_data = wave_data * (1. + noise_amplitude * tf.random.uniform(tf.shape(wave_data), minval=0., maxval=1.))

      # manipulate signal
      spectrum.append(mdct.transform(wave_data))
      if i > 0:
        error = tf.abs(spectrum[-1] - spectrum[0])
        error_on_small = tf.where(error > tf.abs(spectrum[0]), error, 0.)
        errors.append(tf.reduce_max(error_on_small))

    print("{0:17.15e} {1:17.15e} {2:17.15e}".format(noise_amplitude,
                                                    tf.reduce_mean(errors), tf.math.reduce_std(errors)))


def test_mdct():
  # setup
  filter_bands_n = 512
  mdct = MDCT(filter_bands_n, dB_max=_dB_MAX, window_type='vorbis')

  # load audio file
  audio_filepath = './data/'
  audio_filename = 'asot_02_cosmos'   # 'asot_02_cosmos_sr8100_118_128.wav'
  sample_rate = 44100  # None   # 90*90
  wave_data, sample_rate = load_wav(audio_filepath + audio_filename + ".wav", sample_rate)
  wave_data = clip_wav((1, 18), (1, 28), wave_data, sample_rate)
  # wave_data, sample_rate = sine_wav(1.0, 3.95 * 787.5, sample_rate, 1.0)
  wave_data = wave_data[:, 0:filter_bands_n * int(wave_data.shape[1] / filter_bands_n)]

  # play input
  play_wav(wave_data, sample_rate)

  # manipulate signal
  spectrum = mdct.transform(wave_data)
  wave_reproduced = mdct.inverse_transform(spectrum)

  # plot spectrogram
  fig, ax = plt.subplots(nrows=1)
  codec_utils.plot_spectrogram(ax, pa.ampl_to_norm(spectrum), sample_rate, filter_bands_n)
  plt.show()

  # play and save reconstructed wav
  play_wav(wave_reproduced, sample_rate)
  save_wav(audio_filepath + audio_filename + '_reconstructed.wav', wave_reproduced, sample_rate)

  return


def main():
  # test_mdct()
  # test_mdct_precision()
  # test_psychoacoustic()
  # test_gradient()
  # play_from_im()
  # test_dB_level()
  # test_octave()
  test_mdct2()


if __name__ == "__main__":
  main()
