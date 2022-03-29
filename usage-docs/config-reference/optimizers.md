# OPTIMIZERS

The optimizer builder is set in `EXECUTION.OPTIMIZER_BUILDER`. The optimizer manages the optimizers for the model. Different architectures have different optimization strategies; e.g. for GANs, encoders and decoders might be trained separately. So they many need multiple optimizers. These controls are managed by the optimizer builder.

Options are:

- `ClassificationOptimizer`: The optimizer for CoLabel classifier models. This is a basic optimizer builder that uses a conventional optimizer to update all trainable parameters of a model.




# `OPTIMIZER` options

For the `OPTIMIZER` section, here are the following options.


## `OPTIMIZER_NAME`

Any PyTorch optimizer can be used. They include (from [this list](https://pytorch.org/docs/stable/optim.html)):

- `Adadelta`
- `Adagrad`
- `Adam`
- `AdamW`
- `SparseAdam`
- `Adamax`
- `ASGD`
- `LBFGS`
- `NAdam`
- `RAdam`
- `RMSprop`
- `Rprop`
- `SGD`



## `OPTIMIZER_KWARGS`

These are optimizer-specific keyword arguments for each optimizer. They are available in their respective optimizer documentation on PyTorch (see link above).

<span style="color:teal; font-weight:bold">NOTE:</span> These parameters do not include learning rate parameters below, as they are passed separately.

## `BASE_LR`

This is the starting learning rate parameter. Usually `0.0001` for most experiments.

## `LR_BIAS_FACTOR`

The learning rate bias factor. Usually `1.0`. This adjusts a multiplicative factor for the learning rate.

## `WEIGHT_DECAY`

Weight decay parameter for Adam and AdamW. Usually `0.0005`


## `WEIGHT_BIAS_FACTOR`

Weight bias factor for decay parameters. Usually `0.0005`


## `FP16`

Whether to use mixed precision. Unused at this time. **Boolean**