import torch
from torch import nn
import os
from ednaml.config import BaseConfig
from ednaml.utils.web import cached_path


class AlbertPreTrainedConfig(BaseConfig):
    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.torchscript = kwargs.pop("torchscript", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})


class PreTrainedAlbertModel(nn.Module):
    r"""Base class for all models.
    :class:`~pytorch_transformers.PreTrainedModel` takes care of storing the configuration of the models and handles methods for loading/downloading/saving models
    as well as a few methods commons to all models to (i) resize the input embeddings and (ii) prune heads in the self-attention heads.
    Class attributes (overridden by derived classes):
        - ``config_class``: a class derived from :class:`~pytorch_transformers.PretrainedConfig` to use as configuration class for this model architecture.
        - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names` (string) as keys and `url` (string) of associated pretrained weights as values.
        - ``load_tf_weights``: a python ``method`` for loading a TensorFlow checkpoint in a PyTorch model, taking as arguments:
            - ``model``: an instance of the relevant subclass of :class:`~pytorch_transformers.PreTrainedModel`,
            - ``config``: an instance of the relevant subclass of :class:`~pytorch_transformers.PretrainedConfig`,
            - ``path``: a path (string) to the TensorFlow checkpoint.
        - ``base_model_prefix``: a string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
    """
    config_class = None
    pretrained_model_archive_map = {}
    load_tf_weights = lambda model, config, path: None
    base_model_prefix = ""

    def __init__(self, *inputs, **kwargs):
        super().__init__()
        # if not isinstance(config, PretrainedConfig):
        #    raise ValueError(
        #        "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
        #        "To create a model from a pretrained model use "
        #        "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
        #            self.__class__.__name__, self.__class__.__name__
        #        ))
        # Save config in model
        self.config = AlbertPreTrainedConfig(**kwargs)

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end
        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[
            :num_tokens_to_copy, :
        ] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    def _tie_or_clone_weights(self, first_module, second_module):
        """Tie or clone module weights depending of weither we are using TorchScript or not"""

        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

        if hasattr(first_module, "bias") and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(
                first_module.bias.data,
                (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
                "constant",
                0,
            )

    def resize_token_embeddings(self, new_num_tokens=None):
        """Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.
        Arguments:
            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
                If not provided or None: does nothing and just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.
        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(
            self, self.base_model_prefix, self
        )  # get the base model if needed
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens

        # Tie weights again if needed
        if hasattr(self, "tie_weights"):
            self.tie_weights()

        return model_embeds

    def init_weights(self):
        """Initialize and prunes weights if needed."""
        # Initialize weights
        self.apply(self._init_weights)

        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

    def prune_heads(self, heads_to_prune):
        """Prunes heads of the base model.
        Arguments:
            heads_to_prune: dict with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`).
            E.g. {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        base_model = getattr(
            self, self.base_model_prefix, self
        )  # get the base model if needed

        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(
                heads
            )
            self.config.pruned_heads[layer] = list(
                union_heads
            )  # Unfortunately we have to store it as list for JSON

        base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """
        Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~pytorch_transformers.PreTrainedModel.from_pretrained`` class method.
        """
        import os

        assert os.path.isdir(save_directory), (
            "Saving path should be a directory where the model and"
            " configuration can be saved"
        )

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")

        torch.save(model_to_save.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, **kwargs
    ):
        """
        Instantiate a pretrained pytorch model from a pre-trained model configuration.
        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``
        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.
        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.
        Parameters:
            pretrained_model_name_or_path: either:
                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method
            config: (`optional`) instance of a class derived from :class:`~pytorch_transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.
            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and :func:`~pytorch_transformers.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.
            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.
            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.
            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:
                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~pytorch_transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.
        Examples::
            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)

        # Load config
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                **kwargs
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[
                pretrained_model_name_or_path
            ]
        elif os.path.isdir(pretrained_model_name_or_path):
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, "model.ckpt" + ".index"
                )
            else:
                archive_file = os.path.join(
                    pretrained_model_name_or_path, "pytorch_model.bin"
                )
        else:
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
            )
        except EnvironmentError as e:
            if (
                pretrained_model_name_or_path
                in cls.pretrained_model_archive_map
            ):
                print(
                    "Couldn't reach server at '{}' to download pretrained"
                    " weights.".format(archive_file)
                )
            else:
                print(
                    "Model name '{}' was not found in model name list ({}). We"
                    " assumed '{}' was a path or url but couldn't find any file"
                    " associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ", ".join(cls.pretrained_model_archive_map.keys()),
                        archive_file,
                    )
                )
            raise e
        if resolved_archive_file == archive_file:
            print("loading weights file {}".format(archive_file))
        else:
            print(
                "loading weights file {} from cache at {}".format(
                    archive_file, resolved_archive_file
                )
            )

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location="cpu")
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            return cls.load_tf_weights(
                model, config, resolved_archive_file[:-6]
            )  # Remove the '.index'

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = (
                {} if metadata is None else metadata.get(prefix[:-1], {})
            )
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            print(
                "Weights of {} not initialized from pretrained model: {}"
                .format(model.__class__.__name__, missing_keys)
            )
        if len(unexpected_keys) > 0:
            print(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

        if hasattr(model, "tie_weights"):
            model.tie_weights()  # make sure word embedding weights are still tied

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model
