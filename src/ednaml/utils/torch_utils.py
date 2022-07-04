from typing import Type
import torch
from collections import OrderedDict

# https://gist.github.com/the-bass/0bf8aaa302f9ba0d26798b11e4dd73e3
def rename_state_dict_keys(source, key_transformation, target=None):
    """Rename state dict keys

    Args:
        source             -> Path to the saved state dict.
        key_transformation -> Function that accepts the old key names of the state
                            dict as the only argument and returns the new key name.
        target (optional)  -> Path at which the new state dict should be saved
                            (defaults to `source`)
        Example:
        Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
        other keys.

    Example:
        ```
        def key_transformation(old_key):
            if old_key == "layer.0.weight":
                return "layer.1.weight"
            return old_key
        rename_state_dict_keys(state_dict_path, key_transformation)
        ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)

from ednaml.utils.LabelMetadata import LabelMetadata
from ednaml.models.ModelAbstract import ModelAbstract
def build_model_and_load_weights(
    config_file: str,
    model_class: Type[ModelAbstract] = None,
    epoch: int = 0,
    custom_metadata: LabelMetadata = None,
    add_filehandler: bool = False,
    add_streamhandler: bool = True,
):
    """Generates a model using a configuration file, and loads a specific saved epoch

    Args:
        config_file (str): _description_
        model_class (Type[ModelAbstract], optional): _description_. Defaults to None.
        epoch (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    from ednaml.core import EdnaML

    eml = EdnaML(
        config=config_file,
        mode="test",
        test_only=True,
        add_filehandler=add_filehandler,
        add_streamhandler=add_streamhandler,
    )
    eml.labelMetadata = custom_metadata  # TODO this needs to be fixed with the actual label metadata...or tell users to not infer things :|
    if model_class is not None:
        eml.addModelClass(model_class=model_class)
    eml.buildModel()
    if epoch is None:
        # means we get the most recent epoch available...
        epoch = eml.getPreviousStop()
    eml.loadEpoch(epoch=epoch)
    return eml.model