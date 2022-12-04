# EdnaML

EdnaML is a declarative API and framework for deep learning to run reproducible ML experiments. EdnaML is currently designed for PyTorch for ease of adoption. 

## Design

EdnaML's bottom-up design comprises of a layered API with basic building blocks. With the bottom-up design, we provide the basic building blocks for an ML pipeline, such as abstractions for data, model, training, and deployments. Then, we can build higher-level APIs on top of the building blocks to create a layered API that allows for provenance and tracking in the higher-level API while allowing access to the lower-level building blocks. 

![EdnaML Design](https://i.redd.it/ls10r2spbzz91.jpg)

The lowest-level are basic building blocks comprising of existing libraries that are already heavily optimized for major machine learning tasks, such as `PyTorch` for model design and data loading, `HuggingFace` API for transformers and diffusers, and `scikit-learn` for statistical learning. We build pipeline abstractions corresponding to pipeline stages (e.g. data crawling, batch generation, model building, and model deployment) on top of these low-level building blocks. Abstractions can be used as-is or extended to use any other building blocks the user may wish. For example, the `ModelAbstract` abstraction (in `/src/ednaml/models/ModelAbstract`) for an ML classifier is built on top of PyTorch’s nn.Module; however, it can be replaced with a custom `ModelAbstract` as long as the API is itself preserved.

Finally, we build a declarative API for instantiating, executing, reproducing, evaluating, and deploying an ML pipeline. The declarative API, accessed through the `ednaml.core.EdnaML()` class, allows for relatively easy deployment of experiments from configuration files, while still allowing access to the low-level building blocks to extend or replace any abstraction in the pipeline. 
