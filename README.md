# open-exchange

# Support Framworks
-   Keras
-   MXNet
-   PyTorch
-   Caffe
-   Tensorflow

# Exmaples

### MXNet <-> IR

- Convert MXNet to IR
  ```
  CUDA_VISIBLE_DEVICES=0 python ./scripts/convertToIR.py -s mxnet -d outname -n path/to/network -w path/to/weight/file
  ```

- Convert IR to MXNet
  ```
  CUDA_VISIBLE_DEVICES=0 python ./scripts/IRtoModel.py -f mxnet -d path/to/save/the/destination/model -n path/to/IR/network/structure/file -w path/to/IR/weight/file
  ```

### Caffe <-> IR

- Convert caffe to IR
  ```
  CUDA_VISIBLE_DEVICES=0 python ./scripts/convertToIR.py -s caffe -d outname -n path/to/network -w path/to/weight/file
  ```

- Convert IR to Caffe
  ```
  CUDA_VISIBLE_DEVICES=0 python ./scripts/IRtoModel.py -f caffe -d path/to/save/the/destination/model -n path/to/IR/network/structure/file -w path/to/IR/weight/file
  ```

### PyTorch <-> IR

- Convert pytorch to IR
  ```
  CUDA_VISIBLE_DEVICES=0 python ./scripts/convertToIR.py -s pytorch -d outname -n path/to/network -w path/to/weight/file
  ```

- Convert IR to pytorch
  ```
  CUDA_VISIBLE_DEVICES=0 python ./scripts/IRtoModel.py -f pytorch -d path/to/save/the/destination/model -n path/to/IR/network/structure/file -w path/to/IR/weight/file
  ```

### Keras -> IR

- Convert keras to IR
  ```
  CUDA_VISIBLE_DEVICES=0 python ./scripts/convertToIR.py -s keras -d outname -n path/to/network -w /path/to/weight/file
  ```

### Tensorflow <-> IR

#### Convert tf to IR
- Download resnet_v1_101 data
  ```
  $ wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
  $ tar -xvf resnet_v1_101_2016_08_28.tar.gz
  ```

- Extract tf model files
  Refer the ```common/tensorflow/extractor.py``` to implement your own model extract code.

  ```
  CUDA_VISIBLE_DEVICES=0 python ./common/tensorflow/extractor.py -n resnet_v1_101 -p path/to/network/file -o path/to/outdir
  ```

- Convert tf to IR
  ```
  CUDA_VISIBLE_DEVICES=0 python ./scripts/convertToIR.py -s tf -d outname -n path/to/network --dstNodeName Squeeze -w path/to/weight/file
  ```

#### Convert IR to tf

- Convert tf to IR
  ```
  CUDA_VISIBLE_DEVICES=0 python ./scripts/IRtoModel.py --phase test/train -f tf -d path/to/save/the/destination/model -n path/to/IR/network/structure/file -w path/to/IR/weight/file
  ```


## Acknowledgements
Thanks to [Microsoft](https://github.com/Microsoft), the initial code of *MXNet -> IR converting* is references to his project [MMdnn](https://github.com/Microsoft/MMdnn).
