## Conv-TasNet
Tensorflow - keras implementation of Conv-TasNet


### Modify according to code of Kaituo Xu


### Modify PIT and rerank method by one hot dot.

## Install
Tensorflow with eager_excution() (1.3.0~2.0.0),Py3,librosa, numpy, ect in import 

## Dataset
TIMIT dataset load method can be find in previous work, I will upload them as fast as possible

## Current equipment requirements
RTX2080Ti(Least) + i9 9900kf + 32GB menmory
## How to
1.Make dir 'audio_only' by modify codes in get_tfrecord_n.py
2.Run config maker.py to get configs dir
3.Run main, onece OOM error, change params in configmaker
4.Run Go method

# There is still some bugs in this code, for better result, see novel conde provided in https://github.com/yujiacheng333/Speech-Experiment/tree/master/TasnetSeries, both DPRNN and ConvTasNet are included.
