# tiny-torch

<!-- | Conv-Net                               | MLP                                   |
|:-------------------------------------:|:-------------------------------------:|
| <img src="generated/conv_net.svg" width="400" height="1200"/> | <img src="generated/mlp.svg" width="400" height="300" style="position: relative; top: -450px;"/> | -->

## Conv-Net

<img src="generated/conv_net.svg" width="500" height="1000">

```sh
rm generated/*
# define Conv-Net and train on GPU
# (you can change DEVICE macro to run on CPU instead)
nvcc models/conv_net.cu && ./a.out
# optionally, visualize model graph
cat generated/graph.txt | dot -Tsvg -o generated/conv_net.svg
# optionally, run the generated test code
python generated/codegen.py
```

## MLP

<img src="generated/mlp.svg" width="400" height="250"/>

```sh
rm generated/*
# define MLP and train on CPU 
# (you can change DEVICE macro to run on GPU instead)
nvcc models/mlp.cu && ./a.out
# optionally, visualize model graph
cat generated/graph.txt | dot -Tsvg -o generated/mlp.svg
# optionally, run the generated test code
python generated/codegen.py 
```

<!-- ## Miscellaneous commands -->
## Setup commands

```sh
# setup evn to run the generated test code
conda create -n tiny_torch_tests python=3.11 && conda activate tiny_torch_tests
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

```sh
# download data
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P ../data
tar -xvzf ../data/cifar-10-binary.tar.gz
```
