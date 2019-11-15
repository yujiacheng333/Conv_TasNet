import json
import os
json_path = "./configs"


class ConfigTables(object):
    """
    This class is used to json file IO op
    """
    def __init__(self):
        self.json_dir = json_path
        self.io_config_fp = self.json_dir + "/" + "ioconfig.json"
        self.data_config_fp = self.json_dir + "/" + "data_config.json"
        self.train_config_fp = self.json_dir + "/" + "train_config.json"
        self.model_config_fp = self.json_dir + "/" + "model_config.json"
        with open(self.io_config_fp, "r") as f:
            self.io_config = json.load(f)
        with open(self.data_config_fp, "r") as f:
            self.data_config = json.load(f)
        with open(self.train_config_fp, "r") as f:
            self.train_config = json.load(f)
        with open(self.model_config_fp, "r") as f:
            self.model_config = json.load(f)

    @staticmethod
    def resetconfig():
        os.makedirs(json_path, exist_ok=True)
        io_config = {"reset": False,
                     "data_prefix": "./audio_only",
                     "Train_record_name": "TIMIT_train.tfrecord",
                     "Test_record_name": "TIMIT_test.tfrecord"
                     }

        data_config = {"fs": 8000,
                       "dataexpandrate": 2,
                       "audio_length": 32700//2,
                       "max_mix_num": 2,
                       "normal_type": "max_devide",
                       "mulaw": 256,
                       "train_test_split": .2,
                       "output_dir": "./output/",
                       }
        train_config = {"lr": 1e-4,
                        "optimizer": "adma",
                        "ckpt_dir": './training_checkpoints/',
                        "epoches": 55,
                        "batch_size": 2,
                        "evaluation_size": 1,
                        "max_to_keep": 5,
                        "plot_pertire": 5000,
                        "savemodel_periter": 50
                        }

        model_config = {"filters_e": 256,
                        "kernel_size_e": 20,
                        "bottle_filter": 256,
                        "filters_block": 512,
                        "kernel_size_block": 3,
                        "num_conv_block": 7,
                        "number_repeat": 2,  # 青春版配置~~,可以自己加哦
                        "spk_num": 2,  # This affect the output size different to max_mix_num
                        "norm_type": "cln",
                        "causal": False,
                        "mask_nonlinear": "relu"
                        }
        with open(json_path + "/ioconfig.json", "w") as f:
            f.write(json.dumps(io_config))
        with open(json_path + "/data_config.json", "w") as f:
            f.write(json.dumps(data_config))
        with open(json_path + "/train_config.json", "w") as f:
            f.write(json.dumps(train_config))
        with open(json_path + "/model_config.json", "w") as f:
            f.write(json.dumps(model_config))


if __name__ == '__main__':
    ConfigTables.resetconfig()
