# open-exchange

# Support Framworks
-   Keras
-   MXNet
-   PyTorch

# Exmaples
### MXNet <-> IR
### Caffe <-> IR
### PyTorch <-> IR
### Keras <-> IR
### Tensorflow <-> IR
- Download resnet_v1_101 data
  ```
  $ wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
  $ tar -xvf resnet_v1_101_2016_08_28.tar.gz
  ```

- Extract tf model files
  Refer the ```common/tensorflow/extractor.py``` to implement your own model extract code.

  ```
  CUDA_VISIBLE_DEVICES=1 python ./common/tensorflow/extractor.py -n resnet_v1_101 -p /home/cgy/AITISA/test/tf-torch/resnet_v1_101.ckpt -o /home/cgy/AITISA/test/tf-torch/resnet_v1_101
  ```

- Convert tf to IR
  ```
  CUDA_VISIBLE_DEVICES=1 python ./scripts/convertToIR.py -s tf -d kit_imagenet -n /home/cgy/AITISA/test/tf-torch/resnet_v1_101/graph_resnet_v1_101.ckpt.ckpt.meta --dstNodeName Squeeze -w /home/cgy/AITISA/test/tf-torch/resnet_v1_101.ckpt
  ```

## Acknowledgements
Thanks to [Microsoft](https://github.com/Microsoft), the initial code of *MXNet -> IR converting* is references to his project [MMdnn](https://github.com/Microsoft/MMdnn).
