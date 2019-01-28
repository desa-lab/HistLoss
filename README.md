# Histogram Loss
A fast implementation of the histogram loss in pytorch, and the original paper can be found here:
* [Learning Deep Embeddings with Histogram Loss] - https://arxiv.org/pdf/1611.00822.pdf

## Getting started
Both forward and backward functions are implemented, so it can be used as a loss function in your own work. This version is rather stable on both CPUs and GPUs as no outstanding errors occurred during tests.

### Implementation
This implementation is based on two pieces of information available online about pytorch:
* [torch.bincount](https://pytorch.org/docs/stable/torch.html?highlight=bincount#torch.bincount) - The very fast `bincount` function in pytorch
* [Extending Pytorch](https://pytorch.org/docs/stable/notes/extending.html) - Writing your own customised layer with both forward and backward functions.


### Prerequisites
```
pytorch >= v0.4.1
```

### Running
Import the function into python
```
from hist_loss import HistogramLoss
```
Initialise an instance of the function
```
func_loss = HistogramLoss()
```
Forward computation
```
loss = func_loss(sim_pos, sim_neg, n_bins, w_pos, w_neg)
```
Backward computation
```
loss.backward()
```

### Contact
* [shuaitang93@ucsd.edu](mailto:shuaitang93.ucsd.edu) - Email
* [@Shuai93Tangr](https://twitter.com/Shuai93Tang) - Twitter
* [shuaitang](http://shuaitang.github.io/) - Homepage
