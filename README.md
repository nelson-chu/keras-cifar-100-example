# CIFAR-100 Classification with SE-Residual and Dense Blocks

This repository contains an implementation of a neural network model that combines SE-Residual Blocks, Dense Blocks, and Transition Layers to train on the CIFAR-100 dataset using Keras. The model achieves a top-1 accuracy of 0.70819 and a top-5 accuracy of 0.91829 on the test set.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Setup](#setup)
- [Training and Evaluation](#training)
- [Docker Usage](#docker-usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The CIFAR-100 dataset consists of 100 classes containing 600 images each. This project implements a deep learning model using SE-Residual Blocks, Dense Blocks, and Transition Layers to efficiently learn and classify the images.

## Model Architecture

The model architecture integrates:
- **SE-Residual Blocks**: Enhance residual blocks with Squeeze-and-Excitation networks to improve feature recalibration.
- **Dense Blocks**: Connect each layer to every other layer in a feed-forward fashion.
- **Transition Layers**: Reduce the number of feature maps and dimensions.

## Setup

To set up the environment, ensure you have Docker installed. Clone the repository and build the Docker image:

```bash
git clone https://github.com/nelson-chu/keras-cifar-100-example.git
cd keras-cifar-100-example
docker build -t cifar100-model .
```

## Training and Evaluation

To train the model, run the following Docker command:

```bash
docker run --gpus all cifar100-model python cifar-100-example.py
```


## Docker Usage

The project includes a Dockerfile for easy setup and execution. Ensure Docker is installed on your machine, then use the provided commands to build and run the container.

## Results

The model achieves:
- **Top-1 Accuracy**: 0.70819
- **Top-5 Accuracy**: 0.91829

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
