# Autoregressive Pretrained Models

Autoregressive library comes with a set of pre-trained models. Neither of models was carefully tuned. In the following
 - `R` refers to the models' receptive field,
 - `Q` to the number of input/output quantization channels.

### Model `mnist_q256`
WaveNet trained on MNIST. R=699, Q=256, conditioned on the digit class.

### Model `mnist_q2`
WaveNet trained on binarized MNIST. R=699, Q=2, conditioned on the digit class.

### Model `fseries_q127`
WaveNet trained on random Fourier series dataset. R=699, Q=127, conditioned on the periodicity of the signal. Training parameters allow periods (integer) between 5-10secs which maps to conditions [0..4].

For usage see [main](../) Readme.

