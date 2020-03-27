# open-exchange
open-exchange is a model conversion and visualization tool to help users inter-operate among different deep learning frameworks. Convert models between PyTorch and Tensorflow.

## Requirments
- tensorflow==1.8.0
- pytorch==0.4.0
- torchvision==0.2.0
- protobuf>=3.6.1
- python>=3.6
- flask

## How to deploy visualization on Web
If you want to access the deployed web page from an external network, you should first modify the *host* and *port* for the *./visualization/app.py* file.

You can deploy it on Web by running:
```shell
python app.py
```

## Model
Framework | ResNet50 | Inception V3 | ShuffleNet | FCN |
:----------:|:----------:|:----------:|:----------:|:----------:|
TensorFlow | √ | √ | √ | √ | 
PyTorch | √ | √ | √ | √ | 

## Test for Tensorflow and PyTorch
```shell
CUDA_VSIBLE_DEVICES=0 python test.py
```

## Acknowledgements
Thanks to [Microsoft](https://github.com/Microsoft), the initial code of *Tensorflow <-> IR converting* is references to his project [MMdnn](https://github.com/Microsoft/MMdnn).