import numpy as np
import librosa
import os
import json
import re
import shutil
import stat
from sphfile import SPHFile
import glob
import tensorflow as tf
from PIT_loss import noise
import soundfile as sf
import pyroomacoustics as pra
tf.enable_eager_execution()


class get_tfrecord(object):

    # Todo WGAN-plenty farme work and add the noise input for the high freq data at the same time reduce the robust
    def __init__(self, org_train=None, org_test=None, mac_r=1, vs=340, max_num_mix=3, min_source_power=0.1):
        if org_train is None:
            self.org_train = "./TIMIT_data/TRAIN/"
        else:
            self.org_train = org_train
        if org_test is None:
            self.org_test = "./TIMIT_data/test/"
        else:
            self.org_test = org_test
        self.r = 1
        self.L = 65536
        self.room = pra.ShoeBox([20, 20], absorption=1.)
        self.max_l = 0
        self.noise_local = noise("./noisex-92/")
        self.train_rate = 0.9
        self.Iter = None
        self.Iter_test = None
        self.record_name = "TIMIT.tfrecords"
        self.test_record_name = "TIMIT_test.tfrecords"
        self.path = "./" + self.record_name
        self.path_test = "./" + self.test_record_name
        self.batch_size = 16
        self.epo = 10
        self.buffer_size = 1000
        self.targ_max2lowrate = 40
        self.targ_dir = "./TIMIT_data/audio_only/"
        self.sub_add = "_n"
        self.fs = 16000
        self.action_rate = {
            "random_mask": 0,
            "random_move": 1,
            "random_noise": 0
        }
        with open("./STFT_params.json") as json_file:
            data = json.load(json_file)
            self.STFT_param = data
            json_file.close()
        self.mac_r = mac_r
        self.vs = vs
        self.mac_seta = np.arange(0, np.pi * 2, np.pi / 3)
        self.max_num_mix = max_num_mix
        self.min_source_power = min_source_power
        self.move_file_2_wav_n()
        self.get_TFRecord()
        with open("./STFT_params.json") as json_file:
            data = json.load(json_file)
            self.STFT_param = data
            json_file.close()
        self.load_tfrecord()
        self.load_tfrecord(mode=1)

    @staticmethod
    def get_dir_frombase(org):
        dirs = os.listdir(org)
        audios = []
        res_dir = []
        for i in dirs:
            train_sub_dir = os.listdir(org + i)
            for j in train_sub_dir:
                res_dir.append(org + i + '/' + j)
        for i in res_dir:
            last_dir = os.listdir(i)
            for j in last_dir:
                if re.match(".*?.WAV", j):
                    audios.append(i + "/" + j)
        return audios

    def move_file_2_wav_n(self, remove_flag=False):
        if not remove_flag:
            train_videos = self.get_dir_frombase(self.org_train)
            print("fine to load train_data's name, length is {}".format(len(train_videos)))
            test_videos = self.get_dir_frombase(self.org_test)
            print("fine to load test_data's name, length is {}".format(len(test_videos)))
            all_videos = train_videos + test_videos
            if os.path.exists(self.targ_dir):
                if len(os.listdir(self.targ_dir)) or "1" + self.sub_add + ".wav" in os.listdir(self.targ_dir):
                    print("The file might be exsist this function {} might not work"
                          .format(self.move_file_2_wav_n.__name__))
                    return
            else:
                os.mkdir(self.targ_dir)
            for i, fp in enumerate(all_videos):
                shutil.copy(fp, self.targ_dir + str(i) + ".WAV")
            for i in range(len(os.listdir(self.targ_dir))):
                fp = self.targ_dir + "/" + str(i) + ".wav"
                sph = SPHFile(fp)
                sph.write_wav(filename=fp.replace(".wav", self.sub_add + ".wav"))
                print("fin {}".format(i))
        else:
            a = input("the dir:{} will be remove".format(self.targ_dir))
            if a:
                try:
                    os.chmod(self.targ_dir, stat.S_IWOTH)
                    os.remove(self.targ_dir)
                except PermissionError:
                    print("Permission is dine,after use chomod try to run with sudo")
        return

    @staticmethod
    def padding_seq(x, max_l):
        if len(x) > max_l:
            return x[0:max_l]
        if max_l > len(x):
            pad_l = int((max_l - len(x)) / 2)
            buffer = np.pad(x, (pad_l, pad_l), "constant", constant_values=(x[0], x[-1]))
            if len(buffer) - max_l != 0:
                buffer = np.append(buffer, x[-1])
            x = buffer
        else:
            if len(x) == max_l:
                return x
            prob = np.random.uniform(0, 1)
            if prob > 0.5:
                x = x[0:max_l]
            else:
                x = x[len(x) - max_l - 1:-1]
                print(len(x))
        return x

    @staticmethod
    def random_noise(x, action_rate):
        prob = np.random.uniform(0, 1)
        if prob < action_rate:
            return x + np.random.normal(loc=0, scale=0.001, size=x.shape)
        else:
            return x

    @staticmethod
    def min_max_normalize(x):
        if len(x) != 1:
            z_o_ser = ((x - np.min(x)) / (np.max(x) - np.min(x)))*2-1
            return z_o_ser
        else:
            return [1]
    @staticmethod
    def whitting(x):
        return (x-np.mean(x))/np.sqrt(np.var(x))
    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def local_2int(x):
        return int(x) if x - int(x) < 0.5 else int(x) + 1

    @staticmethod
    def local_move(x, stride):
        l = x.shape[0]
        if stride > 0:  # 右移
            x = np.concatenate([np.ones([abs(stride), ])*x[0], np.asarray(x)], axis=0)
            x = x[0:l]
        elif stride < 0:  # 左移
            x = np.concatenate([np.asarray(x), np.ones([abs(stride), ])*x[-1]], axis=0)
            x = x[-(l + 1):-1]
        return x


    def get_random_access_input(self, res, record_name="TIMIT.tfrecords"):
        writer = tf.python_io.TFRecordWriter(record_name)
        for counter, base_signal in enumerate(res):
            print("write in to record, {}".format(counter))
            label_signal = base_signal
            self.room = pra.ShoeBox([20, 20], absorption=1.)
            R = pra.beamforming.circular_2D_array(center=[10, 10],
                                                  M=6,
                                                  radius=self.r,
                                                  phi0=0
                                                  )

            R = np.concatenate([R, np.expand_dims(np.asarray([10, 10]), axis=1)], axis=1)
            self.room.add_microphone_array(pra.MicrophoneArray(R, self.room.fs))
            dist = 3
            seta = np.random.randint(0, 360)
            pos_x = dist*np.cos(seta/180*np.pi)
            pos_y = dist*np.sin(seta/180*np.pi)
            self.room.add_source(position=[10+pos_x, 10+pos_y], signal=label_signal)
            self.room.compute_rir()
            self.room.simulate(snr=None, reference_mic=-1)
            recv_sig = self.room.mic_array.signals
            diff_L = int((recv_sig.shape[1] - len(label_signal))/2)
            recv_sig = tf.cast(recv_sig[:, diff_L:-diff_L], tf.float32).numpy()
            label_signal = tf.cast(label_signal, tf.float32).numpy()
            sf.write("haha.wav", recv_sig[:, 0], samplerate=16000)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    "recv_sig_stft": self._bytes_feature(recv_sig.tostring()),
                    "label_signal_stft": self._bytes_feature(label_signal.tostring()),
                    "local_choice": self._int64_feature(seta),
                }
            ))
            writer.write(example.SerializeToString())
        writer.close()

    def get_TFRecord(self,
                     record_name="TIMIT.tfrecords"):

        sr = self.fs
        sub_add = self.sub_add
        tar_dir = self.targ_dir
        if os.path.exists(self.path) and os.path.exists(self.path):
            print("The record file has all ready exsist")
            return "./" + record_name
        # writer = tf.python_io.TFRecordWriter(record_name)
        all_wav_name = glob.glob(tar_dir + "*" + sub_add + ".wav")
        res = []
        resii = []
        for i, dir in enumerate(all_wav_name):
            if i > self.train_rate*len(all_wav_name):
                print("test_data ********finish load{} wavfile".format(dir))
                resii.append(librosa.load(dir, sr)[0])
            print("training data ********* finish load{} wavfile".format(dir))
            res.append(librosa.load(dir, sr)[0])
        for i in res:
            if len(i) > self.max_l:
                self.max_l = len(i)
        max_l_exp = int(np.log2(self.max_l))
        self.max_l = int(np.exp2(max_l_exp))
        for i, audio in enumerate(res):
            res[i] = self.padding_seq(audio, self.max_l)
            print("training*********finish pad {} audios".format(i))
        for i, audio in enumerate(resii):
            resii[i] = self.padding_seq(audio, self.max_l)
            print("test*********finish pad {} audios".format(i))
        self.get_random_access_input(res)
        print("finish prepare training data")
        self.get_random_access_input(resii, record_name=self.test_record_name)
        print("finish prepare test data")

    @staticmethod
    def _extract_fn(data_record):
        features = {
            "local_choice": tf.FixedLenFeature([1], tf.int64),
            "recv_sig_stft": tf.FixedLenFeature([], tf.string),
            "label_signal_stft": tf.FixedLenFeature([], tf.string),
        }
        sample = tf.parse_single_example(data_record, features)
        local_choice = sample["local_choice"]
        recv_sig_sftf = tf.decode_raw(sample["recv_sig_stft"], tf.float32)
        recv_sig_sftf = tf.reshape(recv_sig_sftf, [7, 65536])
        label_signal_stft = tf.decode_raw(sample["label_signal_stft"], tf.float32)
        return [recv_sig_sftf, label_signal_stft, local_choice]

    def load_tfrecord(self, mode=0):
        if mode==0:
            path = self.path
            assert isinstance(path, str)
            dataset = tf.data.TFRecordDataset([path])
            dataset = dataset.map(self._extract_fn)
            dataset = dataset.shuffle(self.buffer_size).repeat(self.epo).batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            self.Iter = iterator
        else:
            path = self.path_test
            assert isinstance(path, str)
            dataset = tf.data.TFRecordDataset([path])
            dataset = dataset.map(self._extract_fn)
            dataset = dataset.shuffle(self.buffer_size).repeat(self.epo).batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            self.Iter_test = iterator


if __name__ == '__main__':

    import scipy.io.wavfile as wf
    iter = get_tfrecord().Iter_test
    buffer = iter.get_next()
    local_data = buffer[0].numpy().astype(np.float32)
    local_data = get_tfrecord().min_max_normalize(local_data[0, :, :])
    wf.write("./haha.wav", 16000, local_data)
