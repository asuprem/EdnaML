import os, json
import torch
from ednaml.deploy.BaseDeploy import BaseDeploy
from ednaml.models.Albert import AlbertPreTrainedModel, AlbertModel, AlbertOnlyMLMHead
from ednaml.models.Albert import AlbertConfig
from torch.nn import CrossEntropyLoss
from ednaml.models.ModelAbstract import ModelAbstract
from torch import nn

from ednaml.utils.albert import AlbertEmbeddingAverage, AlbertPooledOutput, AlbertRawCLSOutput
import ednaml.core.decorators  as edna



@edna.register_model
class FNCFilter(ModelAbstract):
    model_name = "FNCFilter"
    """FNCFilter takes in the raw FNC data and keeps text documents that contain 
    specific keywords.

    Documents with these keywords are saved in fnc-filtered.json, and documents without these keywords are saved in fnc-unfiltered.json

    Args:
        ModelAbstract (_type_): _description_
    """

    def model_attributes_setup(self, **kwargs):
        self.filter_list = kwargs.get("filter_list", ["covid", "corona"])

    def model_setup(self, **kwargs):
        self.filter_func = lambda x : 1 if len([item for item in self.filter_list if item in x]) > 0 else 0

    def forward_impl(self, x, **kwargs): # x is list of objects in batch
        # we are provided a batch of objects as a list
        filter_batch = []
        unfilter_batch = []
        for item in x:
            if self.filter_func(item["full_text"].lower()):
                filter_batch.append(item)
            else:
                unfilter_batch.append(item)

        return filter_batch, [], unfilter_batch

        

@edna.register_deployment
class FNCFilterDeployment(BaseDeploy):
    def deploy_step(self, batch):   # batch should be list of dicts
        filter_batch, _, unfilter_batch = self.model(batch)

        return filter_batch, None, unfilter_batch

    def move_to_device(self, batch):
        return batch

    def output_setup(self, **kwargs):
        filter_file = kwargs.get("filtered_output")
        unfilter_file = kwargs.get("unfiltered_output")
        basename = kwargs.get("basename")
        filter_name = "-".join([basename, filter_file])+".json"
        unfilter_name = "-".join([basename, unfilter_file])+".json"

        self.filter_obj = open(filter_name, "w")
        self.unfilter_obj = open(unfilter_name, "w")

    def output_step(self, logits, features, secondary):
        #logits is filter_batch
        # secondary is unfilter_batch
        for item in logits:
            self.filter_obj.write(json.dumps(item)+"\n")
        for item in secondary:
            self.unfilter_obj.write(json.dumps(item)+"\n")

    def end_of_deployment(self):
        self.filter_obj.close()
        self.unfilter_obj.close()
