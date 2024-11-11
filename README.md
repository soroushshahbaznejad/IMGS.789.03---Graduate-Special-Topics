# IMGS.789.03---Graduate-Special-Topics
Assignment 2
# Generative Models for Image Generation and Anomaly Detection

This repository provides implementations of Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) for image generation, latent space interpolation, and anomaly detection. The models are trained on datasets like MNIST and CIFAR-10 to demonstrate key concepts in generative modeling.

## Table of Contents

- [Overview](#overview)
- [File Descriptions](#file-descriptions)
- [Acknowledgments](#acknowledgments)

---

## Overview

### Generative Adversarial Networks (GANs)

GANs consist of two neural networks, a generator and a discriminator, that compete in a zero-sum game. The generator creates synthetic images, while the discriminator tries to distinguish between real and generated images. By training both networks simultaneously, GANs can generate realistic images.

- **Basic GAN for MNIST**: Generates images of digits.
- **DCGAN for CIFAR-10**: Uses deep convolutional networks to improve generation quality.
- **Latent Space Interpolation**: Interpolates between latent vectors to explore how smoothly GANs transition between images.

### Variational Autoencoders (VAEs)

VAEs are probabilistic generative models that learn a distribution over the input data in a latent space. By sampling from this latent space, VAEs can generate new, similar images. They are also useful for anomaly detection by evaluating reconstruction errors.

- **VAE for MNIST**: Learns to reconstruct and generate new MNIST digits.
- **Anomaly Detection**: Detects anomalies by analyzing reconstruction errors.
- **Latent Space Visualization**: Plots the 2D latent space of digit classes to examine clustering patterns.

---

## File Descriptions

### GANs

1. **`gan_mnist.py`**  
   Implements a simple GAN with fully connected generator and discriminator networks to generate MNIST images.
   - **Tasks**:
     - Train GAN on MNIST.
     - Plot loss curves for generator and discriminator.
     - Visualize generated images at different training epochs.

2. **`dcgan_cifar10.py`**  
   Implements a Deep Convolutional GAN (DCGAN) for generating CIFAR-10 images.
   - **Tasks**:
     - Use strided convolutions instead of pooling layers.
     - Apply batch normalization and specific activation functions for stability.
     - Visualize and compare generated images with CIFAR-10 images.

3. **`latent_interpolation.py`**  
   Implements latent space interpolation between two generated images for GANs.
   - **Tasks**:
     - Select two latent vectors.
     - Interpolate between them and visualize the transition to analyze smoothness and realism.

### VAEs

1. **`vae_mnist.py`**  
   Implements a Variational Autoencoder for MNIST image generation.
   - **Tasks**:
     - Train the VAE on MNIST.
     - Reconstruct input images and generate new images.
     - Plot ELBO (Evidence Lower Bound) loss and KL-divergence during training.

2. **`vae_anomaly_detection.py`**  
   Implements a VAE for anomaly detection using the MNIST dataset.
   - **Tasks**:
     - Train VAE and calculate reconstruction errors.
     - Classify images as "normal" or "anomalous" based on reconstruction error distribution.
     - Visualize reconstruction error distributions and set a threshold for anomaly detection.

3. **`vae_latent_space_visualization.py`**  
   Visualizes the VAE's latent space by creating 2D scatter plots of the latent vectors.
   - **Tasks**:
     - Color code the scatter plot by digit class to observe clustering.
     - Analyze latent space organization based on digit classes.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/generative-models.git
   cd generative-models
