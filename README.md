# Committee of NAS-based models

<p align="center">
  <img src="img/darts.png" alt="darts" width="48%">
</p>
The algorithm is based on continuous relaxation and gradient descent in the architecture space. It is able to efficiently design high-performance convolutional architectures for image classification (on CIFAR-10 and ImageNet) and recurrent architectures for language modeling (on Penn Treebank and WikiText-2). Only a single GPU is required.

## Requirements
```
Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0
```
NOTE: PyTorch 0.4 is not supported at this moment and would lead to OOM.

## Datasets
Instructions for acquiring imagenette can be found [here](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz). CIFAR-10 can be automatically downloaded by torchvision.

## Architecture search (using small proxy models)
To carry out architecture search using 2nd-order approximation, run
```
cd cnn_space1 && python train_search.py      # for conv cells on CIFAR-10
cd cnn_space2 && python train_search.py      # for conv cells on CIFAR-10
cd cnn_space3 && python train_search.py      # for conv cells on CIFAR-10
```
## Test all models
The easist way to get started is to evaluate our DARTS models.
```
cd cnn_space1 && python test_all.py      # for conv cells on CIFAR-10
cd cnn_space2 && python test_all.py      # for conv cells on CIFAR-10
cd cnn_space3 && python test_all.py     # for conv cells on CIFAR-10
```
## To run the ensemble
```
python test_ensemble.py
```
## To run transfer learning in imagenette
```
cd darts_imagenette/cnn && python train_imagenette.py      # for conv cells on CIFAR-10
cd darts_imagenette/cnn_space1 && python train_imagenette.py      # for conv cells on CIFAR-10
cd darts_imagenette/cnn_space2 && python train_imagenette.py      # for conv cells on CIFAR-10
cd darts_imagenette/cnn_space3 && python train_imagenette.py      # for conv cells on CIFAR-10
```

## To run the ensemble in imagenette
```
cd darts_imagenette && python test_ensemble_imagenette.py
```




