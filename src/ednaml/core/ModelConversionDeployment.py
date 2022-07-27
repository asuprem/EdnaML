"""
ModelConversionDeployment is a built-in Edna class to convert models between different ModelAbstract implementations.

The use case for such a ModelConversionDeployment is to change how a model functions after it has already been trained, such as adding additional outputs like moden confidence or self-diagnoses, adjusting how a model manipulates its input, including a post-facto reject option, etc.

Prerequisites for using ModelConversionDeployment:

- a trained model's saved weights

Corequisites for using ModelConversionDeployment:

- a model class, inherited from ModelAbstract
    - ModelAbstract already contains methods to load weights from a .pth file where the keys do not exactly match, with partial_load. This would normally be used to load the saved weights of the older model into the new model
- a config file that links to the old weights through the Edna section OR the save section
- NOTE --> we still need to deal with one small tiny issue!!!!!!! when providing multiple configs, the latter replace the older on same keys. So will the deployment.yml's SAVE section replace the model.yml's SAVE section????
- a config file with an updated Save section to push the new model and weights
- any helper files -- if not provided, ModelConversionDeployment will use the classes from the original models' implementation, unless a new Crawler is provided
- 


Usage Instructions:


edna ModelConversionDeployment new_model_file.py model_config.yml deployment_config.yml

"""

from ednaml.deploy import BaseDeploy
# So we need the load() to be partial load, possibly...
# WE NEED TO ADJUST HOW MODEL LOADS ARE HANDLED IN EDNAML AND EDNADEPLOY

class ModelConversionDeployment(BaseDeploy):
  def deploy_step(self, batch):
    batch = tuple(item.cuda() for item in batch)
    all_input_ids, all_attention_mask, all_token_type_ids, all_masklm, all_labels = batch
    # outputs is a tuple of output, features, secondary_output --> which has the plugin components. Maybe we will split plugin outputs separately...
    import pdb
    pdb.set_trace()
    outputs = self.model(all_input_ids, token_type_ids = all_token_type_ids, attention_mask=all_attention_mask)
    logit = outputs[0].detach().cpu()
    logit_labels = torch.argmax(logit, dim=1)
    #print(str(logit_labels[0]), "\t", ",".join([str(item) for item in all_input_ids[0].cpu()[:10].tolist()]))
    
    # What we need to return...
    # - predicted label, raw logit probability, model L-score, L-score perturbation, Model L-score, cosine distance to nearest proxy in model, euclidean distance to nearest proxy in model
    # So, we need the models themselves to return some of this stuff...
    # Models have already been trained. We basically want the model to be equipped with some plugins
    return logit_labels, logit, None

  def output_setup(self, **kwargs):
    file_path = kwargs.get("file_name", "deploy-out.txt")
    self.fobj = open(file_path, "w")

  def output_step(self, logits, features, secondary):
    self.fobj.write("\n".join([str(item) for item in logits.tolist()])+"\n")