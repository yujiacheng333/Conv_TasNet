import tensorflow as tf
import os
import numpy as np
from evaluate import cal_sdri, cal_sisnri
from configmaker import ConfigTables
from conv_tasnet import ConvTasNet
from get_tfrecord import RecordMaker
from utils import mu_law_decode
from PIT_loss import cal_loss
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
tf.enable_eager_execution()


class ModelAccess(object):
    def __init__(self, training):
        super(ModelAccess, self).__init__()
        self.training = training
        self.main_config = ConfigTables()
        self.ioconfig = self.main_config.io_config
        self.dataconfig = self.main_config.data_config
        self.trainconfig = self.main_config.train_config
        self.modelconfig = self.main_config.model_config
        self.mulaw = self.dataconfig["mulaw"]
        self.audio_length = self.dataconfig["audio_length"]
        self.lr = self.trainconfig["lr"]
        self.batch_size = self.trainconfig["batch_size"]
        self.optimizer = self.trainconfig["optimizer"].lower()
        assert self.optimizer in ["sgd", "adma", "rmsprop"], "Not include other optimzier"
        if self.optimizer == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optimizer == "adma":
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        else:
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.ckpt_dir = self.trainconfig["ckpt_dir"]
        self.epoches = self.trainconfig["epoches"]
        self.output_dir = self.dataconfig["output_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.dataset = RecordMaker(training).dataset
        self.max_to_keep = self.trainconfig["max_to_keep"]
        self.filters_e = self.modelconfig["filters_e"]
        self.plot_pertire = self.trainconfig["plot_pertire"]
        self.kernel_size_e = self.modelconfig["kernel_size_e"]
        self.bottle_filter = self.modelconfig["bottle_filter"]
        self.filters_block = self.modelconfig["filters_block"]
        self.kernel_size_block = self.modelconfig["kernel_size_block"]
        self.num_conv_block = self.modelconfig["num_conv_block"]
        self.number_repeat = self.modelconfig["number_repeat"]
        self.spk_num = self.modelconfig["spk_num"]
        self.norm_type = self.modelconfig["norm_type"]
        self.causal = self.modelconfig["causal"]
        self.mask_nonlinear = self.modelconfig["mask_nonlinear"]
        self.savemodel_periter = self.trainconfig["savemodel_periter"]
        self.convtasnet = ConvTasNet(filters_e=self.filters_e,
                                     kernel_size_e=self.kernel_size_e,
                                     bottle_filter=self.bottle_filter,
                                     filters_block=self.filters_block,
                                     kernel_size_block=self.kernel_size_block,
                                     num_conv_block=self.num_conv_block,
                                     number_repeat=self.number_repeat,
                                     spk_num=self.spk_num,
                                     norm_type=self.norm_type,
                                     causal=self.causal,
                                     mask_nonlinear=self.mask_nonlinear,
                                     speech_length=self.audio_length)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              contasnet=self.convtasnet)
        self.ckpt_manager = tf.contrib.checkpoint.CheckpointManager(
            self.checkpoint,
            directory=self.ckpt_dir,
            max_to_keep=self.max_to_keep)

        self.checkpoint.restore(tf.train.latest_checkpoint(self.ckpt_dir))

        if training:
            self.train_epochs()
        else:
            print("finish load model!")

    def train_epochs(self):
        self.dataset = self.dataset.repeat(self.epoches).shuffle(100).batch(self.batch_size, drop_remainder=True)
        for train_step, datas in enumerate(self.dataset):
            audios, paddings = datas
            if not isinstance(self.mulaw, type(True)):
                audios = mu_law_decode(audios, self.mulaw)
            mixture = audios[:, 0, :][..., tf.newaxis]
            clean_speech = audios[:, 1:, :]
            length = self.audio_length - paddings.numpy()
            length = length[:, 0]
            with tf.GradientTape() as tape:
                tasnet_output = self.convtasnet(mixture, self.training)
                loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(clean_speech, tasnet_output, length)
                grd = tape.gradient(loss, self.convtasnet.trainable_variables)
                self.optimizer.apply_gradients(grads_and_vars=zip(grd,
                                                                  self.convtasnet.variables))
            if train_step % self.savemodel_periter == 0:
                self.ckpt_manager.save()
            if train_step % self.plot_pertire == 0:
                self.eval(mixture, length, target_singla=clean_speech, train_step=train_step)

    def eval(self, mixture, length, target_singla, train_step):
        """

        :param mixture: [bs, time]
        :param length: length for non-padding signal
        :param target_singla: [bs, spk, time]
        :param train_step: use to print
        :return: SISNRi, SDRi
        """
        mixture = mixture[:, :, 0]
        bs, time = mixture.shape
        output = self.convtasnet(mixture[..., tf.newaxis], False)
        loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(target_singla, output, length)
        avg_sisnri = []
        avg_sdri = []
        plt.plot(mixture[0, :].numpy())
        plt.show()
        write(self.output_dir + "mixture.wav", rate=8000, data=mixture[0].numpy())
        for i in range(self.spk_num):
            data = reorder_estimate_source[0, i, :].numpy()
            plt.plot(data)
            plt.show()
            plt.plot(target_singla[0, i, :].numpy())
            plt.show()
            data /= np.max(np.abs(data))
            write(self.output_dir + str(i) + ".wav", rate=8000, data=data)
        for i in range(bs):
            local_mixture = mixture[i, :].numpy()
            local_target = target_singla[i].numpy()
            local_estimate = reorder_estimate_source[i].numpy()
            avg_sdri.append(cal_sdri(local_target, local_estimate, local_mixture))
            avg_sisnri.append(cal_sisnri(local_target, local_estimate, local_mixture))
        print("After {} training, loss={}, SDRI={}, SISNRI={}".format(train_step,
                                                                      loss,
                                                                      np.mean(avg_sdri),
                                                                      np.mean(avg_sisnri)))


if __name__ == '__main__':
    ModelAccess(True)
