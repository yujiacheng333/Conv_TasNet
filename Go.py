from model_access import ModelAccess
from configmaker import ConfigTables
import soundfile as sf
import tensorflow as tf
from utils import normalize
import librosa
from scipy.io import wavfile
import numpy as np
from itertools import permutations
from tensorflow.python.keras import backend as k
import matplotlib.pyplot as plt


def trim_silence(audio_signal):

    audio_signal = tf.signal.frame(audio_signal, frame_length=256, frame_step=128).numpy()
    power = np.sum(audio_signal**2, axis=1)
    midian = np.sort(power)[len(power)//8]
    mask = power > midian
    audio_signal = audio_signal[mask]
    audio_signal = tf.signal.overlap_and_add(audio_signal, frame_step=128)
    return audio_signal


def concatent(input_list, spk_n, data_l):

    outputlist = []
    input_list = tf.concat(input_list, axis=0)
    frame, spk, time = input_list.shape
    spk_list = np.zeros_like(input_list)  # frame, spk, time
    spk_list[0, :, :] = input_list[0, :, :]
    for frame in range(frame-1):
        local_last = tf.split(spk_list[frame, :, :], 2, axis=-1)[-1]
        local_next = tf.split(input_list[frame + 1, :, :], 2, axis=-1)[0]
        # local last [spk1, ~, time] localnext[~, spk2, time]
        pairwise_dot = k.sum(tf.abs(local_last[..., tf.newaxis, :] - local_next[tf.newaxis]), axis=-1)  # 对称
        print(pairwise_dot)
        prems_ = list(permutations(range(spk)))
        prems = tf.one_hot(prems_, depth=spk)  # [spk!, spk, spk]
        prem_dot = k.sum(prems * pairwise_dot[tf.newaxis], axis=[1, 2])
        prem_dot = k.argmin(prem_dot).numpy()
        rank = prems_[prem_dot]
        next_insert = input_list.numpy()[frame + 1, rank, :]
        spk_list[frame+1, :, :] = next_insert
    for spk in range(spk_n):
        local_data = input_list[:, spk, :]
        local_data = tf.signal.overlap_and_add(local_data, frame_step=data_l//2).numpy()
        outputlist.append(local_data)
    plt.plot(outputlist[0])
    plt.show()
    plt.plot(outputlist[1])
    plt.show()
    return outputlist


if __name__ == '__main__':
    tf.enable_eager_execution()
    model = ModelAccess(False)
    main_config = ConfigTables()
    data_config = main_config.data_config
    data_length = data_config["audio_length"]
    batch_size = main_config.train_config["batch_size"]
    spk_num = main_config.model_config["spk_num"]
    sr = data_config["fs"]
    input_audio = sf.read("f90m120_Mix.wav")
    input_audio = librosa.resample(input_audio[0], input_audio[1], sr)
    input_audio = trim_silence(input_audio)
    input_audio = normalize(input_audio, "max_devide")
    input_audio = tf.signal.frame(input_audio,
                                  frame_length=data_length,
                                  frame_step=data_length//2,
                                  pad_end=True)
    if input_audio.shape[0] % 2 != 0:
        input_audio = np.pad(input_audio, ((0, 1), (0, 0)), "constant")
    target_batch = input_audio.shape[0] // batch_size
    split_size = target_batch
    input_audio = tf.split(input_audio, split_size, axis=0)
    output_array = []
    for i, batch_data in enumerate(input_audio):
        batch_data = batch_data[..., tf.newaxis]
        batch_data = tf.cast(batch_data, tf.float32)
        output = model.convtasnet(batch_data)
        output_array.append(output)
        print("finish {}".format(i))
    output = concatent(output_array, spk_num, data_length)
    for i in range(spk_num):
        wavfile.write("sample_output{}.wav".format(i), sr, output[i]/np.max(output[i]))
