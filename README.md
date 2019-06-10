# open-exchange

open-exchange is a model conversion and visualization tool to help users inter-operate among different deep learning frameworks. Convert models between Keras, MXNet, PyTorch, Caffe and Tensorflow.

# Requirments
- tensorflow==1.13.0
- pycaffe
- keras==2.4.1
- pytorch==0.4.1
- mxnet==1.0.0
- protobuf==3.6.1

## How to deploy visualization on Web
If you want to access the deployed web page from an external network, you should first modify the *host* and *port* for the *./visualization/app.py* file.

You can deploy it on Web by running:
```shell
python app.py
```

# Exmaples

### MXNet <-> IR

```shell
cd scripts/
```

- Convert MXNet to IR
  ```shell
  CUDA_VISIBLE_DEVICES=0 python convertToIR.py -s mxnet -d outname -n path/to/network -w path/to/weight/file
  ```

- Convert IR to MXNet
  ```shell
  CUDA_VISIBLE_DEVICES=0 python IRtoModel.py -f mxnet -d path/to/save/the/destination/model -n path/to/IR/network/structure/file -w path/to/IR/weight/file
  ```

### Caffe <-> IR

```shell
cd scripts/
```

- Convert caffe to IR
  ```shell
  CUDA_VISIBLE_DEVICES=0 python convertToIR.py -s caffe -d outname -n path/to/network -w path/to/weight/file
  ```

- Convert IR to Caffe
  ```shell
  CUDA_VISIBLE_DEVICES=0 python IRtoModel.py -f caffe -d path/to/save/the/destination/model -n path/to/IR/network/structure/file -w path/to/IR/weight/file
  ```
  ```The resulting conversion failure is dependent on the protobuf version inconsistency.```

### PyTorch <-> IR

```shell
cd scripts/
```

- Convert pytorch to IR
  ```shell
  CUDA_VISIBLE_DEVICES=0 python convertToIR.py -s pytorch -d outname -n path/to/network -w path/to/weight/file
  ```

- Convert IR to pytorch
  ```shell
  CUDA_VISIBLE_DEVICES=0 python IRtoModel.py -f pytorch -d path/to/save/the/destination/model -n path/to/IR/network/structure/file -w path/to/IR/weight/file
  ```

### Keras -> IR

```shell
cd scripts/
```

- Convert keras to IR
  ```shell
  CUDA_VISIBLE_DEVICES=0 python convertToIR.py -s keras -d outname -n path/to/network -w /path/to/weight/file
  ```

### Tensorflow <-> IR

#### Convert tf to IR
- Download resnet_v1_101 data
  ```shell
  $ wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
  $ tar -xvf resnet_v1_101_2016_08_28.tar.gz
  ```

- Extract tf model files
  Refer the ```common/tensorflow/extractor.py``` to implement your own model extract code.

  ```shell
  CUDA_VISIBLE_DEVICES=0 python ./common/tensorflow/extractor.py -n resnet_v1_101 -p path/to/network/file -o path/to/outdir
  ```

- Convert tf to IR
  ```shell
  CUDA_VISIBLE_DEVICES=0 python convertToIR.py -s tf -d outname -n path/to/network --dstNodeName Squeeze -w path/to/weight/file
  ```

#### Convert IR to tf

- Convert tf to IR
  ```shell
  cd scripts/
  CUDA_VISIBLE_DEVICES=0 python ./scripts/IRtoModel.py --phase test/train -f tf -d path/to/save/the/destination/model -n path/to/IR/network/structure/file -w path/to/IR/weight/file
  ```

<!-- 
## Acknowledgements
Thanks to [Microsoft](https://github.com/Microsoft), the initial code of *MXNet -> IR converting* is references to his project [MMdnn](https://github.com/Microsoft/MMdnn). -->
