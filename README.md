# MNIST Autoencoder for Image Denoising

This project implements and compares three autoencoder architectures trained to **remove noise from handwritten digit images** using the MNIST dataset. Each model learns to take a noisy digit image as input and reconstruct the original, clean image.

## Project Overview

An autoencoder is a neural network trained to compress an input into a smaller representation (encoding) and then reconstruct an output from that representation (decoding). In a **denoising autoencoder**, the twist is:

- **Input to the model:** a noisy version of the image (random Gaussian noise added, then clipped to a valid pixel range)
- **Target for the loss:** the original, clean image

This forces the network to learn the actual structure of a digit rather than simply copying pixels, since it never sees the clean image directly as input — only as the target it's being trained toward.

## Dataset

- **Source:** [MNIST Dataset (Kaggle)](https://www.kaggle.com/datasets/awsaf49/mnist-dataset)
- **Format:** PNG images organized into folders by digit class (`0`–`9`), split into `training/` and `testing/` directories
- **Size:** 60,000 training images, 10,000 test images
- Loaded using `torchvision.datasets.ImageFolder`, with images converted to single-channel grayscale tensors

## Models

Three autoencoder architectures were built and trained, each using **Mean Squared Error (MSE)** loss between the reconstructed output and the original clean image:

| Model | Architecture | Description |
|---|---|---|
| **Model 1 — FFNN Autoencoder** | Fully connected (linear) layers | Flattens each 28×28 image into a 784-length vector; single linear encoder and decoder layer |
| **Model 2 — Transpose CNN Autoencoder** | Convolutional + Transposed Convolutional | Uses `Conv2d` + `MaxPool2d` to encode, and `ConvTranspose2d` layers to upsample and decode |
| **Model 3 — Upsampled CNN Autoencoder** | Convolutional + Nearest-Neighbor Upsampling | Uses `Conv2d` + `MaxPool2d` to encode, and nearest-neighbor upsampling followed by `Conv2d` layers to decode (avoids the checkerboard artifacts sometimes caused by transposed convolutions) |

All models use a sigmoid activation on the final output layer, since pixel values are normalized between 0 and 1.

## Training Details

- **Loss function:** `nn.MSELoss()`
- **Optimizer:** Adam
- **Epochs:** 20
- **Batch size:** 20
- **Train/validation split:** 80% training / 20% validation (drawn from the training set)
- **Noise injection:** Gaussian noise (`noise_factor=0.5`) added to inputs during both training and validation, with pixel values clipped back to `[0, 1]`
- The best-performing model (lowest validation loss) is checkpointed and saved to a `.pth` file after each improving epoch

### Learning Rates Used
- Model 1 (FFNN): `lr = 0.01`
- Model 2 (Transpose CNN): `lr = 0.01`
- Model 3 (Upsampled CNN): `lr = 0.001` — a lower learning rate was needed here to avoid the model collapsing to a constant/background output early in training

## Results

Each trained model is evaluated on a batch of test images, visualized as three rows:

1. **Noisy input** — the corrupted image fed into the model
2. **Model reconstruction** — the model's denoised output
3. **Original clean image** — ground truth for comparison

All three models successfully learn to remove noise and recover recognizable digit shapes, with the Upsampled CNN Autoencoder achieving the lowest validation loss of the three.

## Requirements

```
torch
torchvision
numpy
matplotlib
```

## How to Run

1. Download the [MNIST Dataset from Kaggle](https://www.kaggle.com/datasets/awsaf49/mnist-dataset) and extract it locally.
2. Update the `data_root` path in the notebook to point to your extracted `mnist_png` folder (containing `training/` and `testing/` subfolders).
3. Run the notebook cells in order:
   - Load and prepare the data
   - Define model architectures
   - Train all three models
   - Load the best saved weights
   - Visualize denoising results on test images

## File Structure

```
├── autoencoder_mnist.ipynb          # Main notebook (data loading, models, training, testing)
├── F_Auto_MNIST_model.pth           # Saved weights for FFNN Autoencoder
├── Tran_conv_Auto_MNIST_model.pth   # Saved weights for Transpose CNN Autoencoder
├── upsamp_conv_Auto_MNIST_model.pth # Saved weights for Upsampled CNN Autoencoder
└── README.md
```

## Acknowledgements

Base autoencoder architectures adapted from [NvsYashwanth/MNIST-Autoecncoder](https://github.com/NvsYashwanth/MNIST-Autoecncoder), extended here to perform image denoising rather than plain reconstruction.
