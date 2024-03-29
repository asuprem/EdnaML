from typing import List
from torch import nn
import warnings
from ednaml.utils.LabelMetadata import LabelMetadata


class LossBuilder(nn.Module):
    LOSS_PARAMS = {}
    loss: nn.ModuleList
    loss_labelname: str
    loss_label: str
    loss_classes_metadata: LabelMetadata
    loss_lambda: List[int]

    def __init__(self, loss_functions, loss_lambda, loss_kwargs, **kwargs):
        super().__init__()
        self.loss = nn.ModuleList([])

        fn_len = len(loss_functions)
        lambda_len = len(loss_lambda)
        kwargs_len = len(loss_kwargs)

        self.loss_label = kwargs.get("label", "color")

        self.loss_labelname = kwargs.get("name", None)
        if self.loss_labelname is None:
            warnings.warn(
                "No name given for this loss, using label as name %s"
                % self.loss_label
            )
            self.loss_labelname = self.loss_label

        self.loss_classes_metadata: LabelMetadata = kwargs.get("metadata")

        # Set up the logger
        self.logger = kwargs.get("logger")
        # Sanity check
        if fn_len != lambda_len:
            raise ValueError(
                "Loss function list length is %i. Expected %i length"
                " loss_lambdas, got %i" % (fn_len, fn_len, lambda_len)
            )
        if fn_len != kwargs_len:
            raise ValueError(
                "Loss function list length is %i. Expected %i length"
                " loss_kwargs, got %i" % (fn_len, fn_len, kwargs_len)
            )
        # Add the loss functions with the correct features. Lambda is applied later
        for idx, loss_fn_name in enumerate(loss_functions):
            self.loss.append(
                self.LOSS_PARAMS[loss_fn_name]["fn"](
                    lossname=self.loss_label,
                    metadata=self.loss_classes_metadata,
                    **loss_kwargs[idx]
                )
            )
            self.logger.info(
                "Added {loss} with lambda = {lamb} and loss arguments {largs}"
                .format(
                    loss=loss_fn_name,
                    lamb=loss_lambda[idx],
                    largs=str(loss_kwargs[idx]),
                )
            )

        lambda_sum = sum(loss_lambda)
        loss_lambda = [float(item) / float(lambda_sum) for item in loss_lambda]
        self.loss_lambda = loss_lambda
        self.loss_fn = loss_functions

    def forward(self, **kwargs):
        """Call operator of the loss builder.

        This returns the sum of each individual loss provided in the initialization, multiplied by their respective loss_lambdas.
        TODO update this + CarZam base model forward to deal with logits as well if necessary

        Args (kwargs only):
            labels: Torch tensor of shape (batch_size, 1). The class labels.
            features: Torch tensor of shape (batch_size, embedding_dimensions). The feature embeddings generated by the ReID model.
        """
        loss = 0.0
        for idx, loss_fn in enumerate(self.loss):
            # loss += self.loss_lambda[idx] * fn(kwargs.get(self.LOSS_PARAMS[self.loss_fn[idx]]['args'][0]), kwargs.get(self.LOSS_PARAMS[self.loss_fn[idx]]['args'][1]), kwargs.get(self.LOSS_PARAMS[self.loss_fn[idx]]['args'][2]))
            loss += self.loss_lambda[idx] * loss_fn(
                *[
                    kwargs.get(arg_name, None)
                    for arg_name in self.LOSS_PARAMS[self.loss_fn[idx]]["args"]
                ]
            )

        return loss


from .ClassificationLossBuilder import ClassificationLossBuilder
