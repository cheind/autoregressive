---
marp: true
size: 16:9
paginate: true
theme: default
#backgroundColor: #fff
#color: #33
math: katex
---
<!-- 
_class: lead
_footer: 'https://github.com/cheind/autoregressive'
_paginate: false
 -->
<style>
section { 
    font-size: 20px; 
}
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>
<style scoped>section { font-size: 30px; }</style>
# Autoregressive Models
**Christoph Heindl**
12/2021

---

# Autoregressive

Given a set of random variables $\mathbf{x}=\{x_1,x_2,x_3...,x_T\}$, we represent their joint distribution as
$$
\begin{align*}
p(\mathbf{x}) &= \prod_{i=1}^Tp(x_i\mid \mathbf{x}_{j<i})\\
&=p(x_1)p(x_2 \mid x_1)p(x_3 \mid x_2, x_1)\ldots,
\end{align*}
$$
which is always possible (chain rule).

---

# Dilated Convolutions
<!--_footer: Note, how each input (orange) within the receptive field is used exactly once.-->
Receptive field of dilated convolutions grows exponentially while parameters increase only linearly.

![center](wavenet-dilated-convolutions.svg)

In general, each layer with dilation factor $D_i$ and kernel size $K_i$ adds
$$
 r_i = (K_i-1)D_i
$$
to the receptive field $R=\sum_i r_i$.

---

# Causal Padding


Causal padding (left-padding) ensures that convoluted features do not depend on future values. Two possibilities: input-padding (left), layer-padding (right)

![](wavenet-causal-padding.svg) ![](wavenet-causal-padding2.svg)

In general, a total $P=R-1$ paddings is required.

<!--_footer: Autoregressive library uses layer-padding, WaveNet paper suggest input padding.-->
---

![bg fit right:50%](compare_curves_train_unroll.svg)

# Train-Unrolling Results

N-step forecast comparison between two models trained with and without unrolling on Fourier-series dataset with up to 4 terms.

## Conclusion
(+) Decreases generative drift
(+) Improves recreation of higher frequency patterns
(-) Increases training time (rolling origin)
(-) Sparser losses

---

![bg fit right:50%](compare_curves_noise.svg)

# Train-Unrolling Results

N-step prediction based on noisy observations - comparison between two models trained with and without unrolling on a clean Fourier series dataset with up to 4 terms.

## Conclusion
(+) Both models capture global trends
(-) Accuracy of both modes decreases

---

![bg fit right:45%](compare_val_acc.svg)

# Train-Unrolling Results

8-step rolling origin validation comparison between models trained with and without unrolling on Fourier-series dataset with up to 4 terms.

## Conclusion
(+) Generally higher validation acc. at earlier training epochs.
(+) Similar picture if validation unrolling > train unrolling steps.

---

# Generative Results

The following graph shows four samples drawn from the models' prior distribution.

![center w:1024](prior_samples.svg)

---

![bg fit right:50%](benchmark_generators.svg)

# Runtime Performance Results

The plot to the left shows default (blue) and fast (orange) sample generation using 64 wave-channels, 8 quantization levels and 32 batch-size.

## Conclusion
(+) Fast method avoids exponential inference time as layer depth increases.
(-) Code overhead is considerable.

