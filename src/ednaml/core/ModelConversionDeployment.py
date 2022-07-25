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