# HZ-0I sampled-softmax probe

Implemented an optional sampled-vocabulary loss (`reference/hz0i_sampled_softmax.py`)
to reduce the 24,576-way output projection during pretraining. On the current
15M-width CPU probe, full CE took 63.0ms for five steps while 1,024-negative
sampling took 208.7ms. The gather/scatter implementation is 3.3x slower here,
so it is rejected as a default optimization. It remains available for future
large-width GPU/Metal kernels where the projection may dominate.
