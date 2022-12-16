"""

pass storages into config
Only storages

eq = EdnaQuery(config = [])


eq.getERSKeyWhere



What we want to query on:
        experiment_key.model_core_name
        experiment_key.model_version
        experiment_key.model_backbone
        experiment_key.model_qualifier
        experiment_key.run
        experiment_key.epoch
        experiment_key.step
        experiment_key.artifact
        experiment_key.metrics.[metric_name] <loss, class_loss, acc, soft_acc, etc, etc>
        experiment_key.config.[config_key]


NOTES:
    since all metrics and artifacts can be saved canonically OR with ers-key, this makes querying a little bit more complicated

For FS query -- maybe we just...dont? We'll leave it for someone else to set up...


We should focus on our mongo and other systems!



select experiment_key where experiment_key.model_core_name = "mnist_resnet" AND experiment_key.model_qualifier = "cifar10"



"""