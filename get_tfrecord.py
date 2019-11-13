import tensorflow as tf
import numpy as np
import os
import librosa
from utils import pad_audios, mix_audios
from configmaker import ConfigTables
from sklearn.model_selection import train_test_split
tf.enable_eager_execution()


class RecordMaker(object):
    """
    Todo implemet multichannel mix
    """

    def __init__(self, training):
        self.main_config = ConfigTables()
        self.ioconfig = self.main_config.io_config
        self.dataconfig = self.main_config.data_config
        # IO config
        self.reset = self.ioconfig["reset"]
        self.data_prefix = self.ioconfig["data_prefix"]
        self.train_record_name = self.ioconfig["Train_record_name"]
        self.test_record_name = self.ioconfig["Test_record_name"]
        # data config
        self.fs = self.dataconfig["fs"]
        self.dataexpandrate = self.dataconfig["dataexpandrate"]
        self.mulaw = self.dataconfig["mulaw"]
        self.audio_length = self.dataconfig["audio_length"]
        self.normal_type = self.dataconfig["normal_type"]
        self.max_mix_num = self.dataconfig["max_mix_num"]
        self.train_test_split = self.dataconfig["train_test_split"]
        # local_method
        if self.reset:
            if self.normal_type != "max_devide" and not isinstance(self.mulaw, type(True)):
                raise ValueError("Once mulaw is used normal type must be max_devide," +
                                 "if you ignore this message some further Error might appear")

            self.audios, self.paddings = self.read_audios(self.data_prefix)
            self.audios, self.paddings = mix_audios(self.audios,
                                                    paddings=self.paddings,
                                                    dataexpand_rate=self.dataexpandrate,
                                                    max_mix_num=self.max_mix_num,
                                                    mulaw=self.mulaw)
            # here I split expanded data which might not a good way
            # If you need to use this algorithm to write a paper, please first segment in the expansion
            self.audios_train, self.audios_test, self.paddings_train, self.paddings_test = \
                train_test_split(self.audios,
                                 self.paddings,
                                 test_size=self.train_test_split)
            del self.audios, self.paddings
            self.train_dataset = tf.data.Dataset.from_tensor_slices((self.audios_train, self.paddings_train))
            self.test_dataset = tf.data.Dataset.from_tensor_slices((self.audios_test, self.paddings_test))
            del self.audios_train, self.audios_test, self.paddings_train, self.paddings_test
            writer = tf.io.TFRecordWriter(self.train_record_name)
            for data in self.train_dataset:
                writer.write(self.serialize_example(data[0], data[1]))
            writer.close()
            writer = tf.io.TFRecordWriter(self.test_record_name)
            for data in self.test_dataset:
                writer.write(self.serialize_example(data[0], data[1]))
            writer.close()
            del self.train_dataset, self.test_dataset, writer
        if training:
            raw_dataset_train = tf.data.TFRecordDataset(self.train_record_name)
            self.dataset = raw_dataset_train.map(self._extract_fn)
        else:
            raw_dataset_test = tf.data.TFRecordDataset(self.test_record_name)
            self.dataset = raw_dataset_test.map(self._extract_fn)

    def _extract_fn(self, data_record):
        features = {
            'audios': tf.io.FixedLenFeature([], tf.string),
            'paddings': tf.io.FixedLenFeature([], tf.string)
        }
        sample = tf.io.parse_single_example(data_record, features)
        if not isinstance(self.mulaw, type(True)):
            recv_sig = tf.io.decode_raw(sample["audios"], tf.int16)
        else:
            recv_sig = tf.io.decode_raw(sample["audios"], tf.float32)
        recv_sig = tf.reshape(recv_sig, [self.max_mix_num + 1, -1])
        paddings = tf.io.decode_raw(sample["paddings"], tf.int16)
        return recv_sig, paddings

    def serialize_example(self, signals, paddings):
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        if not isinstance(self.mulaw, type(True)):
            signals = signals.numpy().reshape([-1]).astype(np.int16)
        else:
            signals = signals.numpy().reshape([-1]).astype(np.float32)
        feature = {
            'audios': _bytes_feature(signals.tostring()),
            'paddings': _bytes_feature(paddings.numpy().reshape([-1]).astype(np.int16).tostring())
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def read_audios(self, file_path):
        all_audios = os.listdir(file_path)
        audios = []
        paddings = []
        counter = 0
        for i in all_audios:
            if "_n" in i:
                counter += 1
                if counter > 100:
                    break
                local_audio, padding = pad_audios(librosa.load(file_path+"/"+i, sr=self.fs)[0],
                                                  normal_type=self.normal_type,
                                                  audio_length=self.audio_length)
                audios.append(local_audio)
                paddings.append(padding)
        audios = np.asarray(audios)
        paddings = np.asarray(paddings)
        return audios, paddings
