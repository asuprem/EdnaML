# `LOSS`

The loss builder builds loss functions for our model. There may be multiple loss functions for each output, as well as multiple collections of loss functions, one for each output. So, the `LOSS` section is an array of `LOSSES`, each array itself comprising an array of losses for a single output of the model, as follows

```
LOSS:
  - LOSSES: ['SoftmaxLogitsLoss']
    KWARGS: [{}]
    LAMBDAS: [1.0]
    NAME: out1
  - LOSSES: ['SoftmaxLogitsLoss', 'CenterLoss']
    KWARGS: [{}, {"center_loss_args":"center_loss_values"}]
    LAMBDAS: [0.5, 0.5]
```


# `LOSS` array

Each array corresponds to a single output of the model and comprises of 4 parameters:

- `LOSSES`: array of losses used for this output. Currently, only `SoftmaxLogitsLoss` is supported for this.
- `KWARGS`: array of arguments for each loss in `LOSS[i].LOSSES`
- `LAMBDAS`: The weight of each loss for this output. These are normalized, so values of `[1,5,3]` are applicable.
- `NAME`: Optional. The name for this output that this loss array corresponds to.



# `LOSS` array order

Since each entry in the loss array corresponds to a single output, you need to check the model you are building. The model's `forward` function gives the order of outputs. 


