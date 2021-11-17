---
marp: true
size: 16:9
paginate: true
theme: default
backgroundColor: #fff
color: #222
math: katex
---
<!-- 
_class: lead
_footer: https://github.com/cheind/autoregressive
 -->
<style>
section { 
    font-size: 25px; 
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

Given a set of random variables $\mathbf{x}=\{x_1,x_2,x_3...,x_T\}$, we model their joint
$$
\begin{align*}
p(\mathbf{x}) &= \prod_{i=1}^Tp(x_i\mid \mathbf{x}_{j<i})\\
&=p(x_1)p(x_2 \mid x_1)p(x_3 \mid x_2, x_1)\ldots,
\end{align*}
$$
always ok (chain rule).

---

# WaveNets

Here comes some text, then the image

![center](wavenets.svg) 

Then again, text.