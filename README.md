# tiny-torch

<!-- | Conv-Net                               | MLP                                   |
|:-------------------------------------:|:-------------------------------------:|
| <img src="generated/conv_net.svg" width="400" height="1200"/> | <img src="generated/mlp.svg" width="400" height="300" style="position: relative; top: -450px;"/> | -->

## Conv-Net

<img src="generated/conv_net.svg" width="500" height="1000">

```sh
rm generated/*
# define and train Conv-Net
nvcc models/conv_net.cu && ./a.out
# visualize model graph
cat generated/graph.txt | dot -Tsvg -o generated/conv_net.svg
```

## MLP

<img src="generated/mlp.svg" width="400" height="250"/>

```sh
rm generated/*
# define and train MLP
nvcc models/mlp.cu && ./a.out
# visualize model graph
cat generated/graph.txt | dot -Tsvg -o generated/mlp.svg
```
