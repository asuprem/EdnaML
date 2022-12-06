import h5py
import torch

import ednaml.core.decorators as edna
from ednaml.deploy import HFMLMSequenceDeploy


@edna.register_deployment
class SemanticMaskingFeaturesGenerator(HFMLMSequenceDeploy):
    def output_setup(self, **kwargs):
      output_file = kwargs.get("feature_file", "training_features")
      self.output_file = output_file + ".h5"
      self.writer = h5py.File(self.output_file, "w")  #we will delete old training features file
      self.written = False
      self.prev_idx = -1
    
    def output_step(self, logits, features: torch.LongTensor, secondary): 
      if len(features.shape) == 3:
        features = features[:,0,:]
      if self.written:
          feats = features.cpu().numpy()
          self.writer["features"].resize((self.writer["features"].shape[0] + feats.shape[0]), axis=0)
          self.writer["features"][-feats.shape[0]:] = feats        
      else:   # First time writing -- we will need to create the dataset.
          self.writer.create_dataset("features", data=features.cpu().numpy(), compression = "gzip", chunks=True, maxshape=(None,features.shape[1]))
          self.written = True

      if self.writer["features"].shape[0]%5000 == 0:
          self.logger.debug("Chunked %i lines in deployment output %s"%(self.writer["features"].shape[0], self.output_file))
          self.prev_idx = self.writer["features"].shape[0]%5000

    def end_of_epoch(self, epoch: int):
        self.writer.close()

    def end_of_deployment(self):
        pass