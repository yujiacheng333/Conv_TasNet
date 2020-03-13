"""
result:mean SISNR =  12.20971027057527 = 12.21
PESQ:2.786857879608475 = 2.79
STOI:0.875448722686369 = 0.88
"""
from utils import stft, istft, mixture, Noise
from noise import noisedataset
from tensorflow.python.keras import backend as k
import tensorflow as tf
import os
import numpy as np
from evaluate import cal_sisnr
from MAUnet_GP import Generator
from get_tfrecord import RecordMaker
from scipy.io import wavfile

tf.enable_eager_execution()
eps = 1e-6


class ModelAccess(object):
    def __init__(self, training, working=False):
        super(ModelAccess, self).__init__()
        self.noise = Noise()
        self.training = training
        self.working = working
        self.audio_length = 16384
        self.lr = 5e-4
        self.batch_size = 64
        self.stft_params = {"frame_length": 256, "frame_step": 128, "fft_size": 256}
        self.optimizer_g = tf.train.RMSPropOptimizer(self.lr, momentum=.9)
        self.ckpt_dir = "./log/recon"
        os.makedirs("./log/Bnvalue", exist_ok=True)
        self.epoches = 32
        self.output_dir = "./output/"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_to_keep = 5
        self.gen = Generator()
        self.checkpoint = tf.train.Checkpoint(optimizer_g=self.optimizer_g,
                                              gen=self.gen,
                                              )
        self.noise = Noise()
        self.ckpt_manager = tf.contrib.checkpoint.CheckpointManager(
            self.checkpoint,
            directory=self.ckpt_dir,
            max_to_keep=self.max_to_keep)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.ckpt_dir))

        if not working:
            self.dataset_train = RecordMaker(True).dataset
            self.dataset_test = RecordMaker(False).dataset
            n = noisedataset()
            self.noise_train = n.dataset_train.repeat(100)
            self.noise_test = n.dataset_test.repeat(100)
            if training:
                self.train_epochs()
            else:
                self.test()
        else:
            print("finish load model")

    def train_epochs(self):

        dataset = self.dataset_train.shuffle(1000).batch(self.batch_size, drop_remainder=True)
        noisedataset = self.noise_train.shuffle(1000).batch(self.batch_size, drop_remainder=True)
        for epoch in range(80):
            local_iter1 = dataset.make_one_shot_iterator()
            local_iter2 = noisedataset.make_one_shot_iterator()
            local_iter = zip(local_iter1, local_iter2)
            for step, data in enumerate(local_iter):
                clean_t, noise_t = data
                clean_t, xb = clean_t
                noise_t, n_type = noise_t
                noise_data = mixture(clean_t, noise_t)
                with tf.GradientTape() as gen_tape:
                    noise_data = stft(noise_data, **self.stft_params)
                    noise_data = tf.concat([tf.real(noise_data), tf.imag(noise_data)], axis=-1)
                    prediction = self.gen(noise_data, training=True)
                    recon = prediction * noise_data
                    recon = tf.cast(recon, tf.complex64)
                    recon = recon[..., 0] + recon[..., 1] * 1j
                    clean_t = stft(clean_t, **self.stft_params)
                    loss = k.mean(k.abs(clean_t - recon)**2)

                    gen_gradients = gen_tape.gradient(loss,
                                                      self.gen.trainable_variables)
                    self.optimizer_g.apply_gradients(zip(gen_gradients, self.gen.trainable_variables))
            self.ckpt_manager.save()
            self.test()

    def test(self):
        dataset = self.dataset_test.repeat(5).batch(256, drop_remainder=True)
        dataset_noise = self.noise_test.batch(256, drop_remainder=True)
        local_iter = zip(dataset.make_one_shot_iterator(), dataset_noise.make_one_shot_iterator())
        recon_sisnr = []
        for data in local_iter:

            content0, content1 = data
            clean_t, xb = content0
            noise_t, ntp = content1
            noise_data_t = mixture(clean_t, noise_t, [0, 0])
            noise_data = stft(noise_data_t, **self.stft_params)[:, :, 1:, tf.newaxis]
            noise_data = tf.concat([tf.real(noise_data), tf.imag(noise_data)], axis=-1)
            prediction = self.gen(noise_data, training=False)
            prediction *= noise_data
            prediction = tf.pad(prediction, ((0, 0), (0, 0), (1, 0), (0, 0)), constant_values=0, mode="constant")
            prediction = tf.cast(prediction, tf.complex64)
            prediction = prediction[..., 0] + 1j * prediction[..., 1]
            recon_stft = istft(prediction, **self.stft_params).numpy()
            recon_stft = recon_stft[:, :self.audio_length]

            for i in range(recon_stft.shape[0]):
                local_clean = clean_t[i].numpy() / np.max(np.abs(clean_t[i]))
                local_noise = noise_data_t[i].numpy() / np.max(np.abs(noise_data_t[i]))
                local_recon = recon_stft[i] / np.max(np.abs(recon_stft[i]))
                wavfile.write(self.output_dir + "recon{}.wav".format(i),
                              8000, local_recon)
                wavfile.write(self.output_dir + "clean{}.wav".format(i),
                              8000, local_clean)
                wavfile.write(self.output_dir + "noise{}.wav".format(i),
                              8000, local_noise)
                # noise_sisnr.append(cal_sisnr(local_clean, local_noise))
                recon_sisnr.append(cal_sisnr(local_clean, local_recon))
            break
        print(np.mean(recon_sisnr))


if __name__ == '__main__':
    ModelAccess(False, working=False)
