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
        with open(self.io_config_fp, "r") as f:
            self.io_config = json.load(f)
        with open(self.data_config_fp, "r") as f:
            self.data_config = json.load(f)

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
                       "audio_length": 32700,
                       "max_mix_num": 2,
                       "normal_type": "max_devide",
                       "mulaw": 256,
                       "train_test_split": .2,
                       }
        with open(json_path + "/ioconfig.json", "w") as f:
            f.write(json.dumps(io_config))
        with open(json_path + "/data_config.json", "w") as f:
            f.write(json.dumps(data_config))


if __name__ == '__main__':
    ConfigTables.resetconfig()
