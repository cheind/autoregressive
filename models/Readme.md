# Autoregressive Pretrained Models

The following pre-trained models are provided (R=receptive field, Q=number of quantization channels):
- `mnist_q2`: WaveNet trained on MNIST with R=699, Q=2
- `mnist_q256`: WaveNet trained on MNIST with R=699, Q=256
- `fourier`: WaveNet trained on random Fourier dataset.

Each model has roughly 4M parameters. Please note that the models have not been tuned with few parameters in mind.