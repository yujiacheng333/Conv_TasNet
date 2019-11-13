import numpy as np
import tensorflow as tf


def pad_audios(audio, normal_type, audio_length):
    """
    cutoff at bidirection
    pad at right side makes latter work easy
    :param audio: speech signal in numpy array 1D
    :param normal_type: normalize in "white", "minmax", "max_devide"
    :param audio_length: Standard audio length in sample points
    :return: normalized audios with length audio_length
    """
    audio = normalize(audio, normal_type)
    if audio.shape[0] > audio_length:
        diff_l = audio.shape[0] - audio_length
        left = np.random.randint(0, diff_l)
        right = diff_l - left
        audio = audio[left:-right]
        padding = 0
    elif audio.shape[0] < audio_length:
        diff_l = np.abs(audio.shape[0] - audio_length)
        audio = np.pad(audio, (0, diff_l), "edge")
        padding = diff_l
    else:
        padding = 0
    return audio, padding


def normalize(x, normal_type="white"):
    normal_type = normal_type.lower()
    assert normal_type.lower() in ["white", "minmax", "max_devide"]
    if normal_type == "white":
        x -= np.mean(x)
        return x / np.std(x)
    if normal_type == "minmax":
        minval = np.min(x)
        maxval = np.max(x)
        return 2 * (x - minval) / (maxval - minval) - .5
    if normal_type == "max_devide":
        x -= np.mean(x)
        return x / np.max(np.abs(x))


def mu_law_encode(audio, quantization_channels):
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return ((signal + 1) / 2 * mu + 0.5).numpy()


def mu_law_decode(output, quantization_channels):
    """
    Recovers waveform from quantized values.
    """
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
        return tf.sign(signal) * magnitude


def mix_audios(audio_array,
               paddings,
               max_mix_num,
               dataexpand_rate,
               mulaw):
    """

    :param audio_array: [Total, time]
    :param paddings: [Total]
    :param max_mix_num: Max source num
    :param dataexpand_rate: not epoch expand
    :param mulaw: use mulaw compress record size
    :return: [Total * dataexpand_rate , time * (maxmix_num + 1)], # mix data first
            ,[Total * dataexpand_rate, max_mix_num]
    """
    org_length = audio_array.shape[0]
    assert org_length == paddings.shape[0], "Data length not same!!"
    output_length = org_length * dataexpand_rate
    iter_audio = []
    iter_padding = []
    for i in range(int(output_length)):
        index = i if i < org_length else i - org_length
        fetchs = []
        while index in fetchs or len(fetchs) == 0:
            fetchs = np.random.randint(low=0, high=org_length, size=[max_mix_num-1])

        fetchs = np.append(fetchs, index)
        fetchs_audio = audio_array[fetchs]
        mixture = np.sum(fetchs_audio, axis=0)[np.newaxis]
        mixture = normalize(mixture, "max_devide")
        output_audios = np.concatenate([mixture, fetchs_audio], axis=0)
        output_paddings = paddings[fetchs]
        if not isinstance(mulaw, type(True)):
            output_audios = mu_law_encode(output_audios, mulaw)
        iter_audio.append(output_audios)
        iter_padding.append(output_paddings)
    if not isinstance(mulaw, type(True)):
        return np.asarray(iter_audio).astype(np.int16), np.asarray(iter_padding).astype(np.int16)
    else:
        return np.asarray(iter_audio).astype(np.float32), np.asarray(iter_padding).astype(np.int16)
